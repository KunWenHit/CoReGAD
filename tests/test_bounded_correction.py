import torch

from coregad.models.coregad import CoReGAD


def test_bounded_correction() -> None:
    output = CoReGAD()(torch.randn(64, 2), torch.randn(64, 3), torch.randn(64))
    correction = output["graph_correction"]
    assert bool(torch.all(correction >= 0.0))
    assert bool(torch.all(correction <= output["gamma"]))
    assert torch.equal(
        output["final_anomaly_logit"],
        output["base_anomaly_logit"] + correction,
    )
