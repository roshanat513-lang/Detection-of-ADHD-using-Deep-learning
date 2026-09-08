"""EEGNet (Lawhern et al., 2018) — a compact CNN for EEG classification.

Input shape expected: (batch, 1, n_channels, n_timesteps)
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_timesteps: int,
        num_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps

        # Block 1: temporal convolution + depthwise spatial convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: separable convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),  # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Figure out flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timesteps)
            out = self.block2(self.block1(dummy))
            flat_dim = out.numel()

        self.classifier = nn.Linear(flat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_channels, n_timesteps)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
