import pytest
import torch

from coregad.models.reliability import SpectralResidualEnergyHead


def test_energy_input_contract() -> None:
    head = SpectralResidualEnergyHead()
    assert head(torch.randn(7, 2)).shape == (7,)
    with pytest.raises(ValueError):
        head(torch.randn(7, 3))
