"""EEG preprocessing: filtering, epoching, normalization.

Works on raw EEG loaded via MNE (for .edf inputs) or directly on numpy
arrays (for pre-epoched .npy inputs).
"""
from __future__ import annotations

import glob
import os
from typing import List, Tuple

import numpy as np

try:
    import mne
    mne.set_log_level("WARNING")
except ImportError:  # mne only required for the .edf code path
    mne = None


def bandpass_notch_filter(
    raw_data: np.ndarray,
    sfreq: float,
    low: float = 0.5,
    high: float = 40.0,
    notch: float = 50.0,
) -> np.ndarray:
    """Apply band-pass + notch filtering to a (n_channels, n_samples) array."""
    if mne is None:
        raise ImportError("mne is required for filtering. `pip install mne`.")
    from mne.filter import filter_data, notch_filter

    filtered = filter_data(raw_data, sfreq=sfreq, l_freq=low, h_freq=high, verbose=False)
    filtered = notch_filter(filtered, Fs=sfreq, freqs=notch, verbose=False)
    return filtered


def epoch_signal(
    data: np.ndarray, sfreq: float, epoch_length_sec: float, overlap: float = 0.5
) -> np.ndarray:
    """Slice a continuous (n_channels, n_samples) signal into overlapping epochs.

    Returns array of shape (n_epochs, n_channels, epoch_samples).
    """
    epoch_samples = int(epoch_length_sec * sfreq)
    step = int(epoch_samples * (1 - overlap))
    step = max(step, 1)

    n_channels, n_samples = data.shape
    epochs = []
    start = 0
    while start + epoch_samples <= n_samples:
        epochs.append(data[:, start:start + epoch_samples])
        start += step

    if not epochs:
        return np.empty((0, n_channels, epoch_samples))
    return np.stack(epochs, axis=0)


def zscore_normalize(epochs: np.ndarray) -> np.ndarray:
    """Per-channel, per-epoch z-score normalization.

    epochs: (n_epochs, n_channels, n_timesteps)
    """
    mean = epochs.mean(axis=-1, keepdims=True)
    std = epochs.std(axis=-1, keepdims=True) + 1e-8
    return (epochs - mean) / std


def load_edf_directory(
    directory: str,
    label: int,
    sfreq: float,
    n_channels: int,
    epoch_length_sec: float,
    overlap: float,
    bandpass_low: float,
    bandpass_high: float,
    notch_freq: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load every .edf file in a directory, filter, epoch, and label it.

    Each file is treated as one subject; the returned `groups` array lets
    downstream code do subject-wise (not epoch-wise) train/test splits,
    which prevents leakage between epochs of the same recording.
    """
    if mne is None:
        raise ImportError("mne is required to read .edf files. `pip install mne`.")

    files = sorted(glob.glob(os.path.join(directory, "*.edf")))
    all_epochs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_groups: List[np.ndarray] = []

    for subject_idx, fpath in enumerate(files):
        raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
        raw.resample(sfreq)
        picks = raw.ch_names[:n_channels]
        data = raw.get_data(picks=picks)  # (n_channels, n_samples)

        filtered = bandpass_notch_filter(
            data, sfreq, low=bandpass_low, high=bandpass_high, notch=notch_freq
        )
        epochs = epoch_signal(filtered, sfreq, epoch_length_sec, overlap)
        epochs = zscore_normalize(epochs)

        if epochs.shape[0] == 0:
            continue

        all_epochs.append(epochs)
        all_labels.append(np.full(epochs.shape[0], label, dtype=np.int64))
        all_groups.append(np.full(epochs.shape[0], f"{os.path.basename(fpath)}_{subject_idx}"))

    if not all_epochs:
        n_timesteps = int(epoch_length_sec * sfreq)
        return (
            np.empty((0, n_channels, n_timesteps)),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=object),
        )

    return (
        np.concatenate(all_epochs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_groups, axis=0),
    )
