"""
BigEarthNet-14k Multimodal Dataset for Land Cover Classification.

Implements the Milestone M3.0 / M3.1 specification (Deliverable F.1):
- Paired Sentinel-1 SAR (2 bands: [VH, VV], float32 dB) and
  Sentinel-2 Optical (10 bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]).
- Optical Normalization: DN / 10,000.0, channel-wise standardized with training split statistics:
    means = [0.136, 0.118, 0.106, 0.128, 0.203, 0.257, 0.272, 0.273, 0.187, 0.121]
    stds  = [0.082, 0.076, 0.074, 0.089, 0.098, 0.108, 0.113, 0.112, 0.095, 0.078]
- SAR Normalization: clipped to [-35.0, +5.0] dB and linearly scaled to [0.0, 1.0] via (dB + 35.0) / 40.0.
- Synchronized Spatial Augmentations for Training:
  Random horizontal flips, random vertical flips, and random 90-degree rotations.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import io
import json
import os
import zipfile
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

# Frozen training set statistics (Deliverable F.1, lines 1065-1066)
TRAIN_OPTICAL_MEANS = np.array([
    0.136, 0.118, 0.106, 0.128, 0.203, 0.257, 0.272, 0.273, 0.187, 0.121
], dtype=np.float32)

TRAIN_OPTICAL_STDS = np.array([
    0.082, 0.076, 0.074, 0.089, 0.098, 0.108, 0.113, 0.112, 0.095, 0.078
], dtype=np.float32)

SAR_DB_MIN = -35.0
SAR_DB_MAX = 5.0
SAR_DB_SPAN = 40.0


def normalize_optical(raw_s2: np.ndarray) -> np.ndarray:
    """
    Standardize 10-band Sentinel-2 array (10, H, W).
    1. Rescale DN / 10,000.0.
    2. Channel-wise standardization with frozen training means/stds.
    """
    refl = raw_s2.astype(np.float32) / 10000.0
    means = TRAIN_OPTICAL_MEANS[:, None, None]
    stds = TRAIN_OPTICAL_STDS[:, None, None]
    return (refl - means) / stds


def normalize_sar(raw_s1: np.ndarray) -> np.ndarray:
    """
    Normalize 2-band Sentinel-1 SAR array (2, H, W).
    Clip to [-35.0, +5.0] dB and linearly scale to [0.0, 1.0].
    """
    s1_clipped = np.clip(raw_s1.astype(np.float32), SAR_DB_MIN, SAR_DB_MAX)
    return (s1_clipped - SAR_DB_MIN) / SAR_DB_SPAN


class BigEarthNetDataset(Dataset):
    """
    PyTorch Dataset for BigEarthNet-14k multimodal paired observations.
    """

    def __init__(
        self,
        metadata_path: Union[str, Path],
        archive_path: Union[str, Path],
        classes_path: Optional[Union[str, Path]] = None,
        augment: bool = False,
        subset_size: Optional[int] = None,
        seed: int = 42
    ):
        super().__init__()
        self.metadata_path = Path(metadata_path)
        self.archive_path = Path(archive_path)
        self.augment = augment

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive file not found: {self.archive_path}")

        # Load metadata parquet
        self.df = pd.read_parquet(self.metadata_path)
        if subset_size is not None and subset_size < len(self.df):
            self.df = self.df.sample(n=subset_size, random_state=seed).reset_index(drop=True)

        # Load class taxonomy and fit MultiLabelBinarizer
        if classes_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            classes_path = repo_root / "artifacts" / "specialists" / "land_cover" / "classes.json"
        
        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        self.mlb = MultiLabelBinarizer(classes=self.classes)
        self.targets = self.mlb.fit_transform(self.df["labels"]).astype(np.float32)

        # Lazily opened per worker / process
        self._zip = None
        self._zip_pid = None

    def _get_zip(self) -> zipfile.ZipFile:
        current_pid = os.getpid()
        if self._zip is None or self._zip_pid != current_pid:
            self._zip = zipfile.ZipFile(self.archive_path, "r")
            self._zip_pid = current_pid
        return self._zip

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            opt_tensor: Normalized optical tensor (10, 120, 120)
            sar_tensor: Normalized SAR tensor (2, 120, 120)
            target_tensor: Multi-hot binary labels (16,)
            patch_id: String identifier
        """
        row = self.df.iloc[idx]
        split = row["split"]
        s1_name = row["s1_name"]
        patch_id = row["patch_id"]

        s1_rel = f"BEN_14k/BigEarthNet-S1/{split}/{s1_name}.tif"
        s2_rel = f"BEN_14k/BigEarthNet-S2/{split}/{patch_id}.tif"

        z = self._get_zip()
        s1_bytes = z.read(s1_rel)
        s2_bytes = z.read(s2_rel)

        raw_s1 = tifffile.imread(io.BytesIO(s1_bytes))  # (2, 120, 120) float32
        raw_s2 = tifffile.imread(io.BytesIO(s2_bytes))  # (10, 120, 120) uint16

        # Normalization
        norm_opt = normalize_optical(raw_s2)
        norm_sar = normalize_sar(raw_s1)

        # Spatial augmentations applied synchronously
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                norm_opt = np.flip(norm_opt, axis=2).copy()
                norm_sar = np.flip(norm_sar, axis=2).copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                norm_opt = np.flip(norm_opt, axis=1).copy()
                norm_sar = np.flip(norm_sar, axis=1).copy()
            # Random 90-degree rotations
            k = np.random.randint(0, 4)
            if k > 0:
                norm_opt = np.rot90(norm_opt, k, axes=(1, 2)).copy()
                norm_sar = np.rot90(norm_sar, k, axes=(1, 2)).copy()

        opt_tensor = torch.from_numpy(norm_opt).float()
        sar_tensor = torch.from_numpy(norm_sar).float()
        target_tensor = torch.from_numpy(self.targets[idx]).float()

        return opt_tensor, sar_tensor, target_tensor, patch_id

    def get_class_counts(self) -> np.ndarray:
        """
        Return positive class counts sum over this split.
        """
        return np.sum(self.targets, axis=0)
