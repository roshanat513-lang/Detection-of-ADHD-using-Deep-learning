"""Basic smoke tests: model forward passes, shapes, and preprocessing utils."""
import numpy as np
import torch

from src.models.eegnet import EEGNet
from src.models.cnn_lstm import CNNLSTM
from src.data.preprocessing import epoch_signal, zscore_normalize
from src.data.dataset import compute_class_weights


N_CHANNELS = 19
N_TIMESTEPS = 512
BATCH_SIZE = 4


def test_eegnet_forward_shape():
    model = EEGNet(n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS, num_classes=2)
    x = torch.randn(BATCH_SIZE, 1, N_CHANNELS, N_TIMESTEPS)
    out = model(x)
    assert out.shape == (BATCH_SIZE, 2)


def test_cnn_lstm_forward_shape():
    model = CNNLSTM(n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS, num_classes=2)
    x = torch.randn(BATCH_SIZE, 1, N_CHANNELS, N_TIMESTEPS)
    out = model(x)
    assert out.shape == (BATCH_SIZE, 2)


def test_epoch_signal_shapes():
    sfreq = 128
    data = np.random.randn(N_CHANNELS, sfreq * 10)  # 10 seconds
    epochs = epoch_signal(data, sfreq=sfreq, epoch_length_sec=2.0, overlap=0.5)
    assert epochs.ndim == 3
    assert epochs.shape[1] == N_CHANNELS
    assert epochs.shape[2] == 256  # 2 sec * 128 Hz


def test_zscore_normalize():
    epochs = np.random.randn(5, N_CHANNELS, 100) * 10 + 3
    normed = zscore_normalize(epochs)
    assert np.allclose(normed.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(normed.std(axis=-1), 1, atol=1e-5)


def test_class_weights_balanced():
    y = np.array([0, 0, 1, 1])
    weights = compute_class_weights(y)
    assert torch.allclose(weights, torch.tensor([1.0, 1.0]))


def test_class_weights_imbalanced():
    y = np.array([0, 0, 0, 1])
    weights = compute_class_weights(y)
    # minority class (1) should get a higher weight
    assert weights[1] > weights[0]
