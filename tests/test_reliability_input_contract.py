import pytest
import torch

from coregad.models.reliability import StructuralReliabilityGate


def test_reliability_input_contract() -> None:
    gate = StructuralReliabilityGate()
    assert gate(torch.randn(7, 3)).shape == (7,)
    with pytest.raises(ValueError):
        gate(torch.randn(7, 2))
