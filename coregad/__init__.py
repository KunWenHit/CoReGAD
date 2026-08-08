"""CoReGAD: normal-only graph anomaly detection."""

from .models.coregad import CoReGAD
from .models.graph_context import GraphContextShaping
from .models.normality_core import CrossFittedNormalityCore, FrozenNormalityCore
from .models.spectral_reference import GlobalSpectralReference, SpectralDiscrepancy
from .models.structural_residualization import (
    ControlledStructuralResidualizer,
    CrossFittedNuisanceEstimator,
)

__all__ = [
    "CoReGAD",
    "ControlledStructuralResidualizer",
    "CrossFittedNormalityCore",
    "CrossFittedNuisanceEstimator",
    "FrozenNormalityCore",
    "GlobalSpectralReference",
    "GraphContextShaping",
    "SpectralDiscrepancy",
]

__version__ = "0.1.0"
