from __future__ import annotations

import numpy as np


ALLOWED_SCORE_MEANINGS = {"higher_is_anomaly", "higher_is_normal"}


def orient_anomaly_scores(scores: np.ndarray, native_score_meaning: str) -> np.ndarray:
    """Convert a declared native direction without consulting evaluation labels."""

    if native_score_meaning not in ALLOWED_SCORE_MEANINGS:
        raise ValueError("native score meaning must be declared in the registry")
    values = np.asarray(scores, dtype=np.float64)
    return values.copy() if native_score_meaning == "higher_is_anomaly" else -values
