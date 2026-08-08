from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


INNER_FOLDS = 5
INNER_HASH_SEED = 7319
RFF_DIM = 256
RFF_SEED_BASE = 7319
RIDGE_LAMBDA = 1.0e-2
RFF_CHUNK_SIZE = 65536
RFF_SAMPLE_LIMIT = 8192
STRUCTURAL_RESIDUAL_STRENGTH = 0.75


def robust_center_scale(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or values.shape[1] not in (2, 3):
        raise ValueError("values must have two or three channels")
    center = values.median(dim=0).values
    mad = (values - center).abs().median(dim=0).values * 1.4826
    scale = torch.where(mad > 1.0e-6, mad, torch.ones_like(mad))
    return center, scale


def controlled_center_scale(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("controlled residual must have two channels")
    center = values.median(dim=0).values
    mad = (values - center).abs().median(dim=0).values
    return center, torch.clamp(mad, min=1.0e-6)


def hash64(values: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64) + np.uint64(seed)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def inner_fold_ownership(
    node_ids: np.ndarray,
    seed: int = INNER_HASH_SEED,
) -> np.ndarray:
    return (
        hash64(np.asarray(node_ids, dtype=np.uint64), int(seed))
        % np.uint64(INNER_FOLDS)
    ).astype(np.int8)


def deterministic_sample_indices(
    node_ids: np.ndarray,
    limit: int,
    seed: int,
) -> np.ndarray:
    ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    if ids.size <= int(limit):
        return np.arange(ids.size, dtype=np.int64)
    hashed = hash64(ids.astype(np.uint64), int(seed))
    selected = np.argpartition(hashed, int(limit) - 1)[: int(limit)]
    return selected[np.argsort(hashed[selected], kind="mergesort")].astype(np.int64)


def rbf_bandwidth(
    structural: torch.Tensor,
    node_ids: np.ndarray,
    seed: int,
    sample_limit: int = RFF_SAMPLE_LIMIT,
) -> float:
    indices = deterministic_sample_indices(
        node_ids, min(int(sample_limit), int(structural.shape[0])), int(seed)
    )
    selected = structural[torch.from_numpy(indices).to(structural.device)]
    with torch.no_grad():
        distances = torch.pdist(selected.to(dtype=torch.float32), p=2)
        nonzero = distances[distances > 0]
        sigma = float(nonzero.median().item()) if int(nonzero.numel()) else 1.0
    return sigma if math.isfinite(sigma) and sigma > 1.0e-6 else 1.0


def rff_parameters(
    sigma: float,
    dim: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    weights = torch.randn((3, int(dim)), generator=generator, dtype=torch.float32)
    weights = weights / float(max(sigma, 1.0e-6))
    offsets = torch.rand((int(dim),), generator=generator, dtype=torch.float32)
    offsets = offsets * (2.0 * math.pi)
    return weights.to(device), offsets.to(device)


def rff_design(
    structural: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    features = math.sqrt(2.0 / float(weights.shape[1])) * torch.cos(
        structural @ weights + offsets.reshape(1, -1)
    )
    ones = torch.ones(
        (structural.shape[0], 1),
        dtype=structural.dtype,
        device=structural.device,
    )
    return torch.cat([ones, features], dim=1)


def solve_ridge(
    gram: np.ndarray,
    cross: np.ndarray,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> np.ndarray:
    matrix = np.asarray(gram, dtype=np.float64)
    target = np.asarray(cross, dtype=np.float64)
    regularized = matrix + float(ridge_lambda) * np.eye(
        matrix.shape[0], dtype=np.float64
    )
    try:
        return np.linalg.solve(regularized, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(regularized, target, rcond=None)[0]


@dataclass
class NuisanceState:
    weights: torch.Tensor
    offsets: torch.Tensor
    inner_coefficients: np.ndarray
    full_coefficients: np.ndarray
    ownership: np.ndarray
    sigma: float
    seed: int


class CrossFittedNuisanceEstimator:
    """Five-fold cross-fitted RBF random-Fourier ridge estimator."""

    def __init__(
        self,
        *,
        rff_dim: int = RFF_DIM,
        ridge_lambda: float = RIDGE_LAMBDA,
        inner_folds: int = INNER_FOLDS,
        seed: int = RFF_SEED_BASE,
        chunk_size: int = RFF_CHUNK_SIZE,
    ) -> None:
        if int(rff_dim) != RFF_DIM or float(ridge_lambda) != RIDGE_LAMBDA:
            raise ValueError("released nuisance hyperparameters are fixed")
        if int(inner_folds) != INNER_FOLDS:
            raise ValueError("released nuisance cross-fitting uses five folds")
        self.rff_dim = int(rff_dim)
        self.ridge_lambda = float(ridge_lambda)
        self.inner_folds = int(inner_folds)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.state: NuisanceState | None = None

    def fit(
        self,
        structural_statistics: torch.Tensor,
        spectral_discrepancy: torch.Tensor,
        node_ids: np.ndarray,
    ) -> "CrossFittedNuisanceEstimator":
        if structural_statistics.ndim != 2 or structural_statistics.shape[1] != 3:
            raise ValueError("structural statistics must have three channels")
        if spectral_discrepancy.ndim != 2 or spectral_discrepancy.shape[1] != 2:
            raise ValueError("spectral discrepancy must have two channels")
        ownership = inner_fold_ownership(node_ids)
        sigma = rbf_bandwidth(
            structural_statistics, node_ids, self.seed, RFF_SAMPLE_LIMIT
        )
        weights, offsets = rff_parameters(
            sigma, self.rff_dim, self.seed, structural_statistics.device
        )
        width = 1 + self.rff_dim
        global_gram = np.zeros((width, width), dtype=np.float64)
        global_cross = np.zeros((width, 2), dtype=np.float64)
        fold_gram = np.zeros((self.inner_folds, width, width), dtype=np.float64)
        fold_cross = np.zeros((self.inner_folds, width, 2), dtype=np.float64)
        for begin in range(0, int(structural_statistics.shape[0]), self.chunk_size):
            end = min(int(structural_statistics.shape[0]), begin + self.chunk_size)
            design = rff_design(structural_statistics[begin:end], weights, offsets)
            response = spectral_discrepancy[begin:end]
            global_gram += (design.T @ design).detach().cpu().numpy().astype(np.float64)
            global_cross += (design.T @ response).detach().cpu().numpy().astype(np.float64)
            for inner in range(self.inner_folds):
                mask_array = ownership[begin:end] == inner
                if not bool(mask_array.any()):
                    continue
                mask = torch.from_numpy(mask_array).to(design.device)
                selected_design = design[mask]
                selected_response = response[mask]
                fold_gram[inner] += (
                    selected_design.T @ selected_design
                ).detach().cpu().numpy().astype(np.float64)
                fold_cross[inner] += (
                    selected_design.T @ selected_response
                ).detach().cpu().numpy().astype(np.float64)
        full_coefficients = solve_ridge(
            global_gram, global_cross, self.ridge_lambda
        )
        inner_coefficients = np.stack(
            [
                solve_ridge(
                    global_gram - fold_gram[inner],
                    global_cross - fold_cross[inner],
                    self.ridge_lambda,
                )
                for inner in range(self.inner_folds)
            ],
            axis=0,
        )
        self.state = NuisanceState(
            weights=weights.detach(),
            offsets=offsets.detach(),
            inner_coefficients=inner_coefficients,
            full_coefficients=full_coefficients,
            ownership=ownership,
            sigma=sigma,
            seed=self.seed,
        )
        return self

    def predict_oof(self, structural_statistics: torch.Tensor) -> torch.Tensor:
        state = self._require_state()
        output = torch.empty(
            (structural_statistics.shape[0], 2),
            dtype=structural_statistics.dtype,
            device=structural_statistics.device,
        )
        for inner in range(self.inner_folds):
            indices = np.flatnonzero(state.ownership == inner).astype(np.int64)
            for begin in range(0, int(indices.size), self.chunk_size):
                chosen = torch.from_numpy(
                    indices[begin : begin + self.chunk_size]
                ).to(structural_statistics.device)
                design = rff_design(
                    structural_statistics[chosen], state.weights, state.offsets
                )
                coefficients = torch.from_numpy(
                    state.inner_coefficients[inner].astype(np.float32)
                ).to(structural_statistics.device)
                output[chosen] = design @ coefficients
        return output.detach()

    def predict(self, structural_statistics: torch.Tensor) -> torch.Tensor:
        state = self._require_state()
        coefficients = torch.from_numpy(
            state.full_coefficients.astype(np.float32)
        ).to(structural_statistics.device)
        output = torch.empty(
            (structural_statistics.shape[0], 2),
            dtype=structural_statistics.dtype,
            device=structural_statistics.device,
        )
        with torch.no_grad():
            for begin in range(0, int(structural_statistics.shape[0]), self.chunk_size):
                end = min(int(structural_statistics.shape[0]), begin + self.chunk_size)
                output[begin:end] = rff_design(
                    structural_statistics[begin:end], state.weights, state.offsets
                ) @ coefficients
        return output

    def _require_state(self) -> NuisanceState:
        if self.state is None:
            raise RuntimeError("nuisance estimator has not been fitted")
        return self.state


class ControlledStructuralResidualizer:
    """Apply the fixed controlled structural residualization contract."""

    residual_strength = STRUCTURAL_RESIDUAL_STRENGTH

    def __init__(self, *, seed: int = RFF_SEED_BASE) -> None:
        self.nuisance_estimator = CrossFittedNuisanceEstimator(seed=seed)
        self.spectral_center: torch.Tensor | None = None
        self.spectral_scale: torch.Tensor | None = None
        self.structural_center: torch.Tensor | None = None
        self.structural_scale: torch.Tensor | None = None
        self.controlled_center: torch.Tensor | None = None
        self.controlled_scale: torch.Tensor | None = None

    def fit_transform(
        self,
        spectral_discrepancy: torch.Tensor,
        structural_statistics: torch.Tensor,
        node_ids: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        self.spectral_center, self.spectral_scale = robust_center_scale(
            spectral_discrepancy
        )
        self.structural_center, self.structural_scale = robust_center_scale(
            structural_statistics
        )
        spectral_normalized = self._normalize_spectral(spectral_discrepancy)
        structural_normalized = self._normalize_structural(structural_statistics)
        self.nuisance_estimator.fit(
            structural_normalized, spectral_normalized, node_ids
        )
        predictable = self.nuisance_estimator.predict_oof(structural_normalized)
        controlled = spectral_normalized - self.residual_strength * predictable
        self.controlled_center, self.controlled_scale = controlled_center_scale(controlled)
        controlled_normalized = self._normalize_controlled(controlled)
        return {
            "spectral_discrepancy_normalized": spectral_normalized,
            "structural_statistics_normalized": structural_normalized,
            "structure_predictable_spectral_component": predictable,
            "controlled_spectral_residual_raw": controlled,
            "controlled_spectral_residual": controlled_normalized,
        }

    def transform(
        self,
        spectral_discrepancy: torch.Tensor,
        structural_statistics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        spectral_normalized = self._normalize_spectral(spectral_discrepancy)
        structural_normalized = self._normalize_structural(structural_statistics)
        predictable = self.nuisance_estimator.predict(structural_normalized)
        controlled = spectral_normalized - self.residual_strength * predictable
        return {
            "spectral_discrepancy_normalized": spectral_normalized,
            "structural_statistics_normalized": structural_normalized,
            "structure_predictable_spectral_component": predictable,
            "controlled_spectral_residual_raw": controlled,
            "controlled_spectral_residual": self._normalize_controlled(controlled),
        }

    def _normalize_spectral(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self._required(self.spectral_center).reshape(1, 2)) / self._required(
            self.spectral_scale
        ).reshape(1, 2)

    def _normalize_structural(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self._required(self.structural_center).reshape(1, 3)) / self._required(
            self.structural_scale
        ).reshape(1, 3)

    def _normalize_controlled(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self._required(self.controlled_center).reshape(1, 2)) / self._required(
            self.controlled_scale
        ).reshape(1, 2)

    @staticmethod
    def _required(value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise RuntimeError("residualizer has not been fitted")
        return value
