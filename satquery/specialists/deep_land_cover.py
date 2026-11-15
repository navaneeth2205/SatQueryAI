"""
Deep Multimodal Land Cover Classification Specialist for SatQuery AI.

Wraps the trained Dual-Branch ResNet-34 deep learning specialist:
- Loads model checkpoint (CB-BCE or ASL) and validation-tuned per-class thresholds.
- Accepts raw paired Sentinel-1 SAR and Sentinel-2 Optical observations
  (as TIFF paths, in-memory numpy arrays, or torch tensors).
- Normalizes inputs using the frozen protocol statistics.
- Returns SpecialistResult with predicted classes, probabilities, and execution metadata.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import torch
import tifffile

from satquery.taxonomy import TaskType, InputType
from satquery.specialists.base import BaseSpecialist, SpecialistResult, SpecialistStatus
from satquery.specialists.models.multimodal_resnet import MultimodalResNet34
from satquery.specialists.dataset_land_cover import normalize_optical, normalize_sar


class DeepLandCoverClassifier(BaseSpecialist):
    """
    Deep Multimodal Land Cover Specialist powered by Dual-Branch ResNet-34.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        classes_path: Optional[Union[str, Path]] = None,
        loss_type: str = "cb_bce",
        device: Optional[str] = None
    ):
        super().__init__(
            name=f"DeepLandCoverClassifier_{loss_type.upper()}",
            supported_tasks=[TaskType.LAND_COVER_CLASSIFICATION],
            required_input_type=InputType.SINGLE_IMAGE
        )

        repo_root = Path(__file__).resolve().parent.parent.parent
        self.loss_type = loss_type.lower()
        artifacts_dir = repo_root / "artifacts" / "specialists" / f"deep_land_cover_{self.loss_type}"

        if checkpoint_path is None:
            checkpoint_path = artifacts_dir / f"best_model_{self.loss_type}.pt"
        self.checkpoint_path = Path(checkpoint_path)

        if config_path is None:
            config_path = artifacts_dir / "threshold_config.json"
        self.config_path = Path(config_path)

        if classes_path is None:
            classes_path = repo_root / "artifacts" / "specialists" / "land_cover" / "classes.json"
            if not classes_path.exists():
                classes_path = repo_root / "artifacts" / "specialists" / "land_cover_phase2" / "classes.json"
        self.classes_path = Path(classes_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load classes
        with open(self.classes_path, "r", encoding="utf-8") as f:
            self.classes = json.load(f)
        self.num_classes = len(self.classes)

        # Load threshold configuration if available
        self.thresholds = np.full(self.num_classes, 0.50, dtype=np.float32)
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                self.thresholds = np.array(cfg.get("tuned_class_thresholds", [0.50] * self.num_classes), dtype=np.float32)

        # Initialize model architecture
        self.model = MultimodalResNet34(
            num_classes=self.num_classes,
            optical_channels=10,
            sar_channels=2,
            pretrained=False,
            channel_order="b02_first"
        ).to(self.device)

        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
        else:
            self.is_loaded = False

    def supports_input(self, input_data: Any) -> Tuple[bool, Optional[str]]:
        """Validate input payload."""
        if input_data is None:
            return False, "Input data is None. Must provide paired S1 and S2 observation data."

        if isinstance(input_data, dict):
            # Form A: File paths
            if "s1_path" in input_data and "s2_path" in input_data:
                s1_p = Path(input_data["s1_path"])
                s2_p = Path(input_data["s2_path"])
                if not s1_p.exists() or not s2_p.exists():
                    return False, f"Supplied TIFF paths do not exist: {s1_p}, {s2_p}"
                return True, None

            # Form B: In-memory numpy arrays
            if "s1_array" in input_data and "s2_array" in input_data:
                s1 = input_data["s1_array"]
                s2 = input_data["s2_array"]
                if not (isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray)):
                    return False, "s1_array and s2_array must be numpy ndarrays."
                if s1.ndim != 3 or s1.shape[0] != 2:
                    return False, f"s1_array must have shape (2, H, W), got {s1.shape}."
                if s2.ndim != 3 or s2.shape[0] != 10:
                    return False, f"s2_array must have shape (10, H, W), got {s2.shape}."
                return True, None

        return False, "Unsupported input. Provide dict with 's1_path'/'s2_path' or 's1_array'/'s2_array'."

    def predict(self, input_data: Any) -> SpecialistResult:
        """
        Execute deep multimodal land cover prediction and return standardized SpecialistResult.
        """
        valid, err_msg = self.supports_input(input_data)
        if not valid:
            return SpecialistResult(
                status=SpecialistStatus.INCOMPATIBLE_INPUT,
                task=TaskType.LAND_COVER_CLASSIFICATION.value,
                unavailable_reason=err_msg,
                required_inputs_description="Paired Sentinel-1 SAR (2 bands) and Sentinel-2 Optical (10 bands) imagery."
            )

        if not self.is_loaded:
            return SpecialistResult(
                status=SpecialistStatus.UNAVAILABLE,
                task=TaskType.LAND_COVER_CLASSIFICATION.value,
                unavailable_reason=f"Model checkpoint not found at {self.checkpoint_path}.",
                required_inputs_description="Trained Dual-Branch ResNet-34 checkpoint."
            )

        try:
            # 1. Extract raw arrays
            if "s1_path" in input_data:
                raw_s1 = tifffile.imread(str(input_data["s1_path"]))
                raw_s2 = tifffile.imread(str(input_data["s2_path"]))
            else:
                raw_s1 = input_data["s1_array"]
                raw_s2 = input_data["s2_array"]

            # 2. Normalize inputs
            norm_opt = normalize_optical(raw_s2)
            norm_sar = normalize_sar(raw_s1)

            opt_t = torch.from_numpy(norm_opt).unsqueeze(0).float().to(self.device)
            sar_t = torch.from_numpy(norm_sar).unsqueeze(0).float().to(self.device)

            # 3. Model forward pass
            with torch.no_grad():
                logits, emb = self.model.forward_with_embeddings(opt_t, sar_t)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            # 4. Apply per-class thresholds
            pred_binary = (probs >= self.thresholds).astype(int)
            predicted_classes = [self.classes[i] for i, val in enumerate(pred_binary) if val == 1]

            # Sort predicted classes descending by probability
            prob_dict = {self.classes[i]: round(float(probs[i]), 4) for i in range(self.num_classes)}
            predicted_classes.sort(key=lambda c: prob_dict[c], reverse=True)
            thresholds_dict = {self.classes[i]: round(float(self.thresholds[i]), 4) for i in range(self.num_classes)}

            meta = {
                "specialist_name": self.name,
                "architecture": "DualBranch_ResNet34_CrossAttention",
                "loss_formulation": self.loss_type,
                "embedding_dim": 512,
                "embedding_norm": round(float(torch.norm(emb).item()), 4),
                "device": str(self.device),
                "num_classes_evaluated": self.num_classes
            }

            return SpecialistResult(
                status=SpecialistStatus.SUCCESS,
                task=TaskType.LAND_COVER_CLASSIFICATION.value,
                predictions=predicted_classes,
                probabilities=prob_dict,
                thresholds_applied=thresholds_dict,
                execution_metadata=meta
            )

        except Exception as e:
            return SpecialistResult(
                status=SpecialistStatus.UNAVAILABLE,
                task=TaskType.LAND_COVER_CLASSIFICATION.value,
                unavailable_reason=f"Inference error in {self.name}: {str(e)}"
            )
