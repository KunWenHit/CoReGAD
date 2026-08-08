import math

import torch
from torch import nn

from coregad.models.coregad import CoReGAD


class FrozenReference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.energy = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.LayerNorm(16), nn.Linear(16, 1), nn.Softplus()
        )
        self.gate = nn.Sequential(nn.Linear(3, 8), nn.GELU(), nn.Linear(8, 1), nn.Sigmoid())
        self.gamma_raw = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32))

    def forward(self, spectral: torch.Tensor, structural: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
        energy = self.energy(spectral).reshape(-1)
        reliability = self.gate(structural).reshape(-1)
        correction = torch.sigmoid(self.gamma_raw) * reliability * torch.tanh(energy)
        final_logit = base + correction
        return {"energy": energy, "reliability": reliability, "correction": correction, "score": torch.sigmoid(final_logit)}


def test_behavior_parity() -> None:
    torch.manual_seed(5)
    reference = FrozenReference()
    public = CoReGAD()
    public.spectral_residual_energy.network.load_state_dict(reference.energy.state_dict())
    public.structural_reliability_gate.network.load_state_dict(reference.gate.state_dict())
    public.gamma_raw.data.copy_(reference.gamma_raw.data)
    spectral, structural, base = torch.randn(37, 2), torch.randn(37, 3), torch.randn(37)
    old = reference(spectral, structural, base)
    new = public(spectral, structural, base)
    assert torch.equal(old["energy"], new["spectral_residual_energy"])
    assert torch.equal(old["reliability"], new["structural_reliability"])
    assert torch.equal(old["correction"], new["graph_correction"])
    assert torch.equal(old["score"], new["final_anomaly_score"])
