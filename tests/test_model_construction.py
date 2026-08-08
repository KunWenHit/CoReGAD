from coregad.models.coregad import CoReGAD


def test_model_construction() -> None:
    model = CoReGAD()
    assert sum(parameter.numel() for parameter in model.parameters()) == 139
    assert model.spectral_residual_energy.network[0].in_features == 2
    assert model.spectral_residual_energy.network[0].out_features == 16
    assert model.structural_reliability_gate.network[0].in_features == 3
    assert model.structural_reliability_gate.network[0].out_features == 8
