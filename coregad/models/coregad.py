from __future__ import annotations

import math

import torch
from torch import nn

from .reliability import SpectralResidualEnergyHead, StructuralReliabilityGate


class CoReGAD(nn.Module):
    """Frozen public residual head for CoReGAD."""

    def __init__(self) -> None:
        super().__init__()
        self.spectral_residual_energy = SpectralResidualEnergyHead()
        self.structural_reliability_gate = StructuralReliabilityGate()
        self.gamma_raw = nn.Parameter(
            torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32)
        )

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_raw)

    def forward(
        self,
        controlled_spectral_residual: torch.Tensor,
        structural_statistics: torch.Tensor,
        base_anomaly_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        spectral_residual_energy = self.spectral_residual_energy(
            controlled_spectral_residual
        )
        structural_reliability = self.structural_reliability_gate(
            structural_statistics
        )
        graph_correction = (
            self.gamma
            * structural_reliability
            * torch.tanh(spectral_residual_energy)
        )
        final_anomaly_logit = base_anomaly_logit + graph_correction
        return {
            "controlled_spectral_residual": controlled_spectral_residual,
            "structural_statistics": structural_statistics,
            "spectral_residual_energy": spectral_residual_energy,
            "structural_reliability": structural_reliability,
            "gamma": self.gamma,
            "graph_correction": graph_correction,
            "base_anomaly_logit": base_anomaly_logit,
            "base_anomaly_score": torch.sigmoid(base_anomaly_logit),
            "final_anomaly_logit": final_anomaly_logit,
            "final_anomaly_score": torch.sigmoid(final_anomaly_logit),
        }
