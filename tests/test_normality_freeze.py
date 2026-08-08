import torch

from coregad.models.normality_core import CrossFittedNormalityCore


def test_normality_freeze() -> None:
    core = CrossFittedNormalityCore(7)
    frozen = core.freeze()
    frozen.train(True)
    assert not frozen.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    output = frozen(torch.randn(5, 7))
    assert output["node_embeddings"].shape == (5, 64)
