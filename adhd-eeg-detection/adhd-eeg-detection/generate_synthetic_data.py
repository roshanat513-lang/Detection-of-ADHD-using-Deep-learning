"""Generates synthetic EEG-like epochs so the full pipeline (train/eval) can
be smoke-tested without a real dataset. NOT representative of real EEG or
real ADHD signal characteristics — for pipeline testing only.

Usage:
    python generate_synthetic_data.py --n-subjects 40 --config configs/config.yaml
"""
import argparse
import os

import numpy as np
import yaml


def make_synthetic_epochs(n_subjects_per_class, n_epochs_per_subject, n_channels, n_timesteps, sfreq, seed=42):
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    subject_id = 0

    for label, class_name in enumerate(["control", "adhd"]):
        for _ in range(n_subjects_per_class):
            # Base 1/f-like noise plus a class-dependent oscillatory component
            # (purely synthetic — a stand-in signal, not a clinical model)
            t = np.arange(n_timesteps) / sfreq
            for _ in range(n_epochs_per_subject):
                pink_noise = rng.normal(0, 1, (n_channels, n_timesteps)).cumsum(axis=1)
                pink_noise = (pink_noise - pink_noise.mean()) / (pink_noise.std() + 1e-8)

                # theta/beta ratio is a classic (if debated) ADHD EEG marker;
                # bump theta power a bit for the "adhd" synthetic class only
                # to give the demo model something learnable.
                theta_freq = 6.0
                theta_amp = 1.5 if label == 1 else 0.5
                oscillation = theta_amp * np.sin(2 * np.pi * theta_freq * t)

                epoch = pink_noise + oscillation[None, :]
                X.append(epoch.astype(np.float32))
                y.append(label)
                groups.append(f"{class_name}_subj{subject_id}")
            subject_id += 1

    return np.stack(X), np.array(y, dtype=np.int64), np.array(groups, dtype=object)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--n-subjects", type=int, default=20, help="subjects per class")
    parser.add_argument("--n-epochs-per-subject", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sfreq = config["data"]["sampling_rate"]
    n_channels = config["data"]["n_channels"]
    n_timesteps = int(config["data"]["epoch_length_sec"] * sfreq)

    X, y, groups = make_synthetic_epochs(
        n_subjects_per_class=args.n_subjects,
        n_epochs_per_subject=args.n_epochs_per_subject,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        sfreq=sfreq,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)
    np.save(os.path.join(args.out_dir, "groups.npy"), groups)

    print(f"Synthetic dataset written to {args.out_dir}/")
    print(f"  X.npy shape: {X.shape}")
    print(f"  y.npy shape: {y.shape}, class balance: {np.bincount(y)}")
    print(f"  groups.npy: {len(np.unique(groups))} unique subjects")


if __name__ == "__main__":
    main()
