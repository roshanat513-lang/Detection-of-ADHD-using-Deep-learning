"""CNN-LSTM hybrid model: 1D-CNN spatial-temporal feature extraction
followed by a (bi-)LSTM to capture longer-range temporal dependencies.

Input shape expected: (batch, 1, n_channels, n_timesteps)
"""
from typing import List

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_timesteps: int,
        num_classes: int = 2,
        cnn_channels: List[int] = (16, 32, 64),
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        conv_layers = []
        in_ch = n_channels
        for out_ch in cnn_channels:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*conv_layers)

        # Figure out the sequence length after CNN downsampling
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_timesteps)
            seq_len = self.cnn(dummy).shape[-1]

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_channels, n_timesteps) -> drop the dummy channel dim
        x = x.squeeze(1)  # (batch, n_channels, n_timesteps)
        x = self.cnn(x)   # (batch, cnn_channels[-1], seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_channels[-1])

        lstm_out, (h_n, _) = self.lstm(x)
        # Use the final hidden state(s); concat directions if bidirectional
        if self.lstm.bidirectional:
            final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final = h_n[-1]

        return self.classifier(final)
