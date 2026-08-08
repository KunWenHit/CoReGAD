import torch

from coregad.models.spectral_reference import structural_statistics


def test_structural_statistics() -> None:
    adjacency = torch.eye(5).to_sparse()
    embeddings = torch.randn(5, 3)
    degree = torch.zeros(5)
    support = torch.zeros(5)
    output = structural_statistics(adjacency, embeddings, degree, support)
    assert output.shape == (5, 3)
    assert torch.equal(output, torch.zeros_like(output))
