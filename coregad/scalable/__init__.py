"""Memory-bounded execution engine for the frozen CoReGAD equations."""

from .chunked_spectral import (
    SCALABLE_TRAIN_BATCH_SIZE,
    SPECTRAL_ROW_CHUNK,
    ScalableSpectralArtifacts,
    compute_chunked_graph_features,
)
from .sparse_operator import SparseGraphOperator, build_sparse_graph_operator

__all__ = [
    "SCALABLE_TRAIN_BATCH_SIZE",
    "SPECTRAL_ROW_CHUNK",
    "ScalableSpectralArtifacts",
    "SparseGraphOperator",
    "build_sparse_graph_operator",
    "compute_chunked_graph_features",
]
