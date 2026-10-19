"""Minimal AG-CAG estimator example for a causal history window.

This file contains the network architecture only. It has no measurement files,
feature-generation code, fitted scalers, trained weights, or paper results.
"""

from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Shared MLP applied to average and maximum temporal descriptors."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        bottleneck = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(),
            nn.Linear(bottleneck, channels, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, channels, time]")
        average = self.avg_pool(features).squeeze(-1)
        maximum = self.max_pool(features).squeeze(-1)
        weights = torch.sigmoid(
            self.shared_mlp(average) + self.shared_mlp(maximum)
        ).unsqueeze(-1)
        return features * weights


class AGCAG(nn.Module):
    """CNN, channel attention, GRU, and gated local/history fusion."""

    def __init__(self, input_dim: int = 24, hidden_dim: int = 128) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 2:
            raise ValueError("input_dim must be positive and hidden_dim at least two")
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.attention = ChannelAttention(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return one bias-proxy estimate per past-to-present input window.

        `history` has shape [batch, window_length, input_dim]. It must contain
        only samples available through the prediction instant. The centered
        convolutions operate within that already causal window; no future
        measurement outside the window is supplied to the estimator.
        """
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError("history must have shape [batch, time, input_dim]")
        encoded = self.attention(
            self.encoder(history.transpose(1, 2))
        ).transpose(1, 2)
        local = encoded[:, -1, :]
        temporal, _ = self.gru(encoded)
        memory = temporal[:, -1, :]
        alpha = self.gate(torch.cat((local, memory), dim=-1))
        fused = alpha * memory + (1.0 - alpha) * local
        return self.head(fused)


def synthetic_smoke_test() -> None:
    torch.manual_seed(7)
    model = AGCAG(input_dim=24, hidden_dim=128)
    example = torch.randn(2, 200, 24)
    output = model(example)
    loss = nn.SmoothL1Loss(beta=0.05)(output, torch.zeros_like(output))
    loss.backward()
    print(f"Synthetic shape: {tuple(output.shape)}; finite: {bool(torch.isfinite(output).all())}")


if __name__ == "__main__":
    synthetic_smoke_test()
