"""
Dataset and Data Loading Utilities for OSCD Change Detection.
Milestone M3.2: Deep Bitemporal Change-Detection Specialist.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ImageNet / Sentinel-2 RGB normalization parameters
S2_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
S2_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_rgb(img_arr: np.ndarray) -> np.ndarray:
    """Normalize uint8 RGB image to zero-mean, unit-variance float32 tensor array."""
    # (H, W, 3) in [0, 1]
    norm = (img_arr.astype(np.float32) / 255.0 - S2_RGB_MEAN) / S2_RGB_STD
    # Convert to (3, H, W)
    return np.transpose(norm, (2, 0, 1))


class OSCDPatchDataset(Dataset):
    """
    Extracts fixed-size (patch_size x patch_size) patches from full satellite scenes.
    Supports regular grid tiling and data augmentation.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        patch_size: int = 256,
        stride: int = 128,
        augment: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment

        self.patches: List[Dict] = []
        self._prepare_patches()

    def _prepare_patches(self):
        scene_dirs = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        for scene_dir in scene_dirs:
            t1_img = np.array(Image.open(scene_dir / "t1.png"))
            t2_img = np.array(Image.open(scene_dir / "t2.png"))
            cm_img = np.array(Image.open(scene_dir / "cm.png"))

            h, w = cm_img.shape[:2]

            # Generate sliding window coordinates
            y_coords = list(range(0, h - self.patch_size + 1, self.stride))
            if y_coords[-1] + self.patch_size < h:
                y_coords.append(h - self.patch_size)

            x_coords = list(range(0, w - self.patch_size + 1, self.stride))
            if x_coords[-1] + self.patch_size < w:
                x_coords.append(w - self.patch_size)

            for y in y_coords:
                for x in x_coords:
                    t1_patch = t1_img[y : y + self.patch_size, x : x + self.patch_size]
                    t2_patch = t2_img[y : y + self.patch_size, x : x + self.patch_size]
                    cm_patch = cm_img[y : y + self.patch_size, x : x + self.patch_size]

                    self.patches.append({
                        "scene": scene_dir.name,
                        "y": y,
                        "x": x,
                        "t1": normalize_rgb(t1_patch),
                        "t2": normalize_rgb(t2_patch),
                        "cm": (cm_patch > 0).astype(np.float32)[np.newaxis, ...],  # (1, H, W)
                    })

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.patches[idx]
        t1 = torch.from_numpy(item["t1"].copy())
        t2 = torch.from_numpy(item["t2"].copy())
        cm = torch.from_numpy(item["cm"].copy())

        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                t1 = torch.flip(t1, dims=[-1])
                t2 = torch.flip(t2, dims=[-1])
                cm = torch.flip(cm, dims=[-1])
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                t1 = torch.flip(t1, dims=[-2])
                t2 = torch.flip(t2, dims=[-2])
                cm = torch.flip(cm, dims=[-2])
            # Random 90-degree rotation
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                t1 = torch.rot90(t1, k=k, dims=[-2, -1])
                t2 = torch.rot90(t2, k=k, dims=[-2, -1])
                cm = torch.rot90(cm, k=k, dims=[-2, -1])

        return t1, t2, cm


class OSCDFullScene:
    """Container for full-scene inference with overlapping sliding window reconstruction."""

    def __init__(self, scene_dir: Path):
        self.scene_dir = Path(scene_dir)
        self.scene_name = self.scene_dir.name
        self.t1_raw = np.array(Image.open(self.scene_dir / "t1.png"))
        self.t2_raw = np.array(Image.open(self.scene_dir / "t2.png"))
        self.cm_true = (np.array(Image.open(self.scene_dir / "cm.png")) > 0).astype(np.uint8)

        self.height, self.width = self.cm_true.shape
        self.t1_norm = normalize_rgb(self.t1_raw)
        self.t2_norm = normalize_rgb(self.t2_raw)

    def predict_full_scene(
        self,
        model: torch.nn.Module,
        device: torch.device,
        patch_size: int = 256,
        stride: int = 128,
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Runs sliding window inference across the full scene with Gaussian-weighted overlap blending.
        Handles scenes smaller than patch_size via reflection padding.
        Returns full-resolution probability map (H, W) in [0.0, 1.0].
        """
        model.eval()

        # Handle scenes smaller than patch_size (e.g. campinas with H=241)
        pad_h = max(0, patch_size - self.height)
        pad_w = max(0, patch_size - self.width)

        if pad_h > 0 or pad_w > 0:
            t1_padded = np.pad(self.t1_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            t2_padded = np.pad(self.t2_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            h_eval, w_eval = t1_padded.shape[1], t1_padded.shape[2]
        else:
            t1_padded = self.t1_norm
            t2_padded = self.t2_norm
            h_eval, w_eval = self.height, self.width

        prob_map = np.zeros((h_eval, w_eval), dtype=np.float32)
        weight_map = np.zeros((h_eval, w_eval), dtype=np.float32)

        # Coordinate windows
        y_coords = list(range(0, h_eval - patch_size + 1, stride))
        if len(y_coords) == 0 or y_coords[-1] + patch_size < h_eval:
            y_coords.append(h_eval - patch_size)

        x_coords = list(range(0, w_eval - patch_size + 1, stride))
        if len(x_coords) == 0 or x_coords[-1] + patch_size < w_eval:
            x_coords.append(w_eval - patch_size)

        # Center weight matrix (smooth transitions at patch boundaries)
        center_weight = np.ones((patch_size, patch_size), dtype=np.float32)
        for i in range(patch_size):
            for j in range(patch_size):
                di = min(i, patch_size - 1 - i) / (patch_size / 4)
                dj = min(j, patch_size - 1 - j) / (patch_size / 4)
                center_weight[i, j] = min(1.0, di) * min(1.0, dj)
        center_weight = np.clip(center_weight, 0.1, 1.0)

        # Batch patches
        batch_t1, batch_t2, batch_coords = [], [], []

        with torch.no_grad():
            for y in y_coords:
                for x in x_coords:
                    p1 = t1_padded[:, y : y + patch_size, x : x + patch_size]
                    p2 = t2_padded[:, y : y + patch_size, x : x + patch_size]
                    batch_t1.append(p1)
                    batch_t2.append(p2)
                    batch_coords.append((y, x))

                    if len(batch_t1) == batch_size:
                        t1_t = torch.from_numpy(np.stack(batch_t1)).to(device)
                        t2_t = torch.from_numpy(np.stack(batch_t2)).to(device)
                        logits = model(t1_t, t2_t)
                        probs = torch.sigmoid(logits).cpu().numpy()[:, 0, :, :]

                        for b in range(len(batch_coords)):
                            by, bx = batch_coords[b]
                            prob_map[by : by + patch_size, bx : bx + patch_size] += probs[b] * center_weight
                            weight_map[by : by + patch_size, bx : bx + patch_size] += center_weight

                        batch_t1, batch_t2, batch_coords = [], [], []

            # Remaining batch
            if len(batch_t1) > 0:
                t1_t = torch.from_numpy(np.stack(batch_t1)).to(device)
                t2_t = torch.from_numpy(np.stack(batch_t2)).to(device)
                logits = model(t1_t, t2_t)
                probs = torch.sigmoid(logits).cpu().numpy()[:, 0, :, :]

                for b in range(len(batch_coords)):
                    by, bx = batch_coords[b]
                    prob_map[by : by + patch_size, bx : bx + patch_size] += probs[b] * center_weight
                    weight_map[by : by + patch_size, bx : bx + patch_size] += center_weight

        # Normalize by overlapping weights
        prob_map = prob_map / np.maximum(weight_map, 1e-6)

        # Un-pad back to original scene dimensions
        prob_map = prob_map[: self.height, : self.width]
        return np.clip(prob_map, 0.0, 1.0)
