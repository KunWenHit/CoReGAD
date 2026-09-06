"""CoReGAD: normal-only graph anomaly detection."""

from .models.coregad import (
    FULL_F2,
    LEGACY_CONTROLLED_TIGHT,
    MODEL_VARIANTS,
    CoReGAD,
)
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
    "FULL_F2",
    "LEGACY_CONTROLLED_TIGHT",
    "MODEL_VARIANTS",
]

__version__ = "0.2.0"
