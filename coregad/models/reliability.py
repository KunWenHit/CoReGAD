from __future__ import annotations

import torch
from torch import nn


class SpectralResidualEnergyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.LayerNorm(16),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    def forward(self, controlled_spectral_residual: torch.Tensor) -> torch.Tensor:
        if controlled_spectral_residual.ndim != 2 or controlled_spectral_residual.shape[1] != 2:
            raise ValueError("energy head requires exactly two residual channels")
        return self.network(controlled_spectral_residual).reshape(-1)


class StructuralReliabilityGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, structural_statistics: torch.Tensor) -> torch.Tensor:
        if structural_statistics.ndim != 2 or structural_statistics.shape[1] != 3:
            raise ValueError("reliability gate requires exactly three structural channels")
        return self.network(structural_statistics).reshape(-1)
