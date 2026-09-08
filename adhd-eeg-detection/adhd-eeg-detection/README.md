# ADHD-EEG-Net: Early Detection of ADHD from EEG Signals using Deep Learning

A deep learning pipeline for classifying ADHD vs. Control subjects from resting-state / task EEG recordings, using **EEGNet** and a **CNN-LSTM** hybrid model implemented in PyTorch.

> ⚠️ **Disclaimer**: This project is for **research and educational purposes only**. It is **not a certified medical diagnostic tool**. ADHD diagnosis must only be performed by qualified clinicians using validated clinical criteria (e.g., DSM-5). Do not use this software to make real-world clinical decisions.

---

## ✨ Features

- 📥 EEG data loading for `.edf` / `.csv` / `.npy` formats via [MNE-Python](https://mne.tools/)
- 🧹 Preprocessing pipeline: band-pass filtering, notch filtering, artifact rejection, epoching, per-channel z-score normalization
- 🧠 Two model architectures:
  - **EEGNet** (Lawhern et al., 2018) — compact CNN designed specifically for EEG
  - **CNN-LSTM hybrid** — spatial-temporal feature extraction + sequence modeling
- 📊 Training with early stopping, class-weighted loss (handles imbalanced data), k-fold cross-validation
- 📈 Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, per-fold reports
- 🧪 Synthetic data generator so the whole pipeline runs out-of-the-box with no real dataset
- ✅ Unit tests + GitHub Actions CI
- 🔧 Fully config-driven (`configs/config.yaml`)

## 📁 Project Structure

```
adhd-eeg-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/
│   └── config.yaml          # all hyperparameters & paths
├── generate_synthetic_data.py  # creates dummy EEG data to test the pipeline
├── src/
│   ├── utils.py             # seeding, logging, checkpointing
│   ├── data/
│   │   ├── preprocessing.py # filtering, epoching, normalization
│   │   └── dataset.py       # PyTorch Dataset + loaders (edf/csv/npy)
│   ├── models/
│   │   ├── eegnet.py        # EEGNet architecture
│   │   └── cnn_lstm.py      # CNN-LSTM hybrid architecture
│   ├── train.py             # training loop w/ cross-validation
│   └── evaluate.py          # standalone evaluation on a held-out set
├── tests/
│   └── test_model.py
└── .github/workflows/ci.yml
```

## 🧬 Expected Data Format

The pipeline supports two input modes, set via `configs/config.yaml -> data.format`:

1. **Raw EEG (`edf`)** — one `.edf` file per subject, placed in `data/adhd/` and `data/control/`. Loaded and epoched automatically with MNE.
2. **Pre-epoched arrays (`npy`)** — a single `X.npy` of shape `(n_epochs, n_channels, n_timesteps)` and `y.npy` of shape `(n_epochs,)` with labels `0=control, 1=adhd`.

Public datasets you can use (not included in this repo — download separately and follow each source's license/usage terms):
- IEEE DataPort "EEG Data for ADHD / Control Children"
- ADHD-200 (fMRI, would need a different pipeline)
- Any clinical EEG dataset with resting-state or Go/No-Go / CPT task recordings

## 🚀 Quickstart

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/adhd-eeg-detection.git
cd adhd-eeg-detection
pip install -r requirements.txt

# 2. (No real data yet?) Generate synthetic EEG data to test the pipeline end-to-end
python generate_synthetic_data.py

# 3. Train
python -m src.train --config configs/config.yaml

# 4. Evaluate the best checkpoint
python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best_model.pt
```

## ⚙️ Configuration

All settings live in `configs/config.yaml`: sampling rate, channel count, epoch length, model choice (`eegnet` or `cnn_lstm`), learning rate, batch size, number of folds, etc. Edit this file rather than the code to run new experiments.

## 🧠 Model Notes

- **EEGNet**: depthwise + separable convolutions make it lightweight (~2-3k params for typical configs) and well-suited to small clinical EEG datasets where overfitting is a major risk.
- **CNN-LSTM**: 1D-CNN extracts local spatial-temporal features per window, then a bidirectional LSTM models longer-range temporal dependencies across the EEG epoch — often stronger when longer recordings/epochs are available.

Select the model in `config.yaml`:
```yaml
model:
  name: eegnet   # or: cnn_lstm
```

## 📊 Reproducibility

- Fixed seeds for `numpy`, `torch`, and `random` (see `src/utils.py`)
- Stratified k-fold cross-validation to guard against small-dataset variance
- All metrics logged per fold and averaged, with std reported

## 🩺 Ethical & Clinical Considerations

- ADHD is a heterogeneous, clinically-diagnosed condition; EEG-based classifiers are an active **research** area, not a replacement for clinical assessment.
- Always validate on data from your own population before drawing conclusions — models trained on one age group, clinic, or EEG device often do not generalize.
- Be mindful of class imbalance, subject-level data leakage (never split epochs from the same subject across train/test), and small-sample overfitting — this repo defaults to subject-wise splitting.

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙏 References

- Lawhern et al., 2018. *EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.*
- Various works on ADHD EEG classification (theta/beta ratio features, deep learning approaches on resting-state and CPT-task EEG).
