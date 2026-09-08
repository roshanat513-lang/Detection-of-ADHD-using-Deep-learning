"""PyTorch Dataset wrapping EEG epochs, plus helpers to load from npy or edf."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import load_edf_directory


class EEGDataset(Dataset):
    """Wraps EEG epochs (X) and labels (y) for use with a DataLoader.

    X: (n_epochs, n_channels, n_timesteps) float32
    y: (n_epochs,) int64, 0=control, 1=adhd
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), "X and y must have the same number of samples"
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # Add channel dim for CNN input: (1, n_channels, n_timesteps)
        x = self.X[idx].unsqueeze(0)
        return x, self.y[idx]


def load_dataset(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, groups according to config['data']['format'].

    `groups` identifies which subject each epoch came from, and must be used
    for subject-wise (not random epoch-wise) splitting to avoid data leakage.
    """
    data_cfg = config["data"]

    if data_cfg["format"] == "npy":
        X = np.load(data_cfg["npy_x_path"])
        y = np.load(data_cfg["npy_y_path"])
        groups_path = data_cfg.get("npy_groups_path")
        if groups_path and os.path.exists(groups_path):
            groups = np.load(groups_path, allow_pickle=True)
        else:
            # Fall back to treating every epoch as its own group (no leakage
            # protection) — only reasonable if epochs are already
            # subject-independent, e.g., one epoch per subject.
            groups = np.arange(len(y)).astype(object)
        return X, y, groups

    if data_cfg["format"] == "edf":
        common_kwargs = dict(
            sfreq=data_cfg["sampling_rate"],
            n_channels=data_cfg["n_channels"],
            epoch_length_sec=data_cfg["epoch_length_sec"],
            overlap=data_cfg["epoch_overlap"],
            bandpass_low=data_cfg["bandpass_low"],
            bandpass_high=data_cfg["bandpass_high"],
            notch_freq=data_cfg["notch_freq"],
        )
        X_adhd, y_adhd, g_adhd = load_edf_directory(
            data_cfg["edf_adhd_dir"], label=1, **common_kwargs
        )
        X_ctrl, y_ctrl, g_ctrl = load_edf_directory(
            data_cfg["edf_control_dir"], label=0, **common_kwargs
        )
        X = np.concatenate([X_adhd, X_ctrl], axis=0)
        y = np.concatenate([y_adhd, y_ctrl], axis=0)
        groups = np.concatenate([g_adhd, g_ctrl], axis=0)
        return X, y, groups

    raise ValueError(f"Unknown data format: {data_cfg['format']}")


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights, for use with CrossEntropyLoss."""
    classes, counts = np.unique(y, return_counts=True)
    weights = counts.sum() / (len(classes) * counts)
    weight_tensor = torch.zeros(len(classes), dtype=torch.float32)
    for cls, w in zip(classes, weights):
        weight_tensor[int(cls)] = w
    return weight_tensor
