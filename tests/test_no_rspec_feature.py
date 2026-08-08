import torch

from coregad.models.coregad import CoReGAD
from coregad.models.spectral_reference import SpectralDiscrepancy


def test_only_two_spectral_features() -> None:
    embeddings = torch.randn(8, 4)
    reference = torch.randn(8, 4)
    head = torch.nn.Linear(4, 1)
    output = SpectralDiscrepancy()(embeddings, reference, head)
    assert output["spectral_discrepancy"].shape == (8, 2)
    assert CoReGAD().spectral_residual_energy.network[0].in_features == 2
