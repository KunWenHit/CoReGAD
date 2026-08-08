import torch

from coregad.models.spectral_reference import GlobalSpectralReference


def test_spectral_reference() -> None:
    adjacency = torch.eye(4).to_sparse()
    embeddings = torch.randn(4, 6)
    output = GlobalSpectralReference()(adjacency, embeddings)
    assert torch.equal(output["low_spectral_component"], embeddings)
    assert torch.equal(output["band_spectral_component"], torch.zeros_like(embeddings))
    assert torch.equal(output["high_spectral_component"], torch.zeros_like(embeddings))
    expected = output["global_spectral_weights"][0] * embeddings
    assert torch.allclose(output["spectral_reference"], expected)
