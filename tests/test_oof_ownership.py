import numpy as np

from coregad.data.oof import assemble_oof_scores
from coregad.data.splits import build_outer_fold_manifest


def test_oof_ownership() -> None:
    manifest = build_outer_fold_manifest(np.arange(5), np.arange(5, 105))
    outputs = []
    observed = []
    for fold in range(5):
        _, training, heldout = manifest.fold(fold)
        assert np.intersect1d(training, heldout).size == 0
        observed.extend(heldout.tolist())
        outputs.append((heldout, heldout.astype(np.float64)))
    assert sorted(observed) == list(range(5, 105))
    scores = assemble_oof_scores(105, outputs)
    assert np.isfinite(scores[5:]).all()
