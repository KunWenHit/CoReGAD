from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


F2_ALPHA = 0.75
F2_RHO = 0.80
F2_WIDE_TEMPERATURE = 2.0
ROUTING_EPSILON = 1.0e-8


@dataclass(frozen=True)
class StrictLowerNormalPercentile:
    """Empirical percentile using count(reference < value) / N."""

    sorted_values: torch.Tensor
    normal_count: int

    @classmethod
    def fit(
        cls,
        values: torch.Tensor,
        *,
        normal_mask: torch.Tensor,
        heldout_mask: torch.Tensor | None = None,
    ) -> "StrictLowerNormalPercentile":
        flat = values.detach().reshape(-1)
        normal = normal_mask.detach().reshape(-1).to(
            dtype=torch.bool, device=flat.device
        )
        if flat.shape != normal.shape:
            raise ValueError("normal calibration values and mask must align")
        if heldout_mask is not None:
            heldout = heldout_mask.detach().reshape(-1).to(
                dtype=torch.bool, device=flat.device
            )
            if heldout.shape != normal.shape:
                raise ValueError("heldout calibration mask must align")
            if bool((normal & heldout).any()):
                raise ValueError("normal calibration rejects held-out membership")
        reference = flat[normal]
        if reference.numel() == 0 or not bool(torch.isfinite(reference).all()):
            raise ValueError("normal support must be non-empty and finite")
        return cls(
            sorted_values=torch.sort(reference).values.detach().clone(),
            normal_count=int(reference.numel()),
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        flat = values.detach().reshape(-1)
        reference = self.sorted_values.to(device=flat.device, dtype=flat.dtype)
        ranks = torch.searchsorted(reference, flat, right=False)
        return (ranks.to(values.dtype) / float(self.normal_count)).reshape(
            values.shape
        )


# The scientific lineage used this longer name. Keep it as a public alias.
TieSafeNormalSupportPercentile = StrictLowerNormalPercentile


def tail_ramp(values: torch.Tensor, rho: float = F2_RHO) -> torch.Tensor:
    if not 0.0 < float(rho) < 1.0:
        raise ValueError("tail threshold must be in (0,1)")
    return ((values - float(rho)) / (1.0 - float(rho))).clamp(0.0, 1.0)


def geometric_three(
    left: torch.Tensor, middle: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    product = (
        left.clamp(0.0, 1.0)
        * middle.clamp(0.0, 1.0)
        * right.clamp(0.0, 1.0)
    )
    return torch.pow(product.clamp_min(0.0), 1.0 / 3.0)


def suppression_ratio(
    raw_energy: torch.Tensor, controlled_energy: torch.Tensor
) -> torch.Tensor:
    ratio = (raw_energy - controlled_energy) / (raw_energy + ROUTING_EPSILON)
    return ratio.clamp(0.0, 1.0)


@dataclass(frozen=True)
class RoutingBatch:
    raw_evidence: torch.Tensor
    controlled_evidence: torch.Tensor
    structural_statistics: torch.Tensor
    base_anomaly_logit: torch.Tensor
    normal_mask: torch.Tensor
    normal_channel_support: torch.Tensor
    normal_channel_embedding: torch.Tensor
    normal_channel_decision: torch.Tensor


def fit_normal_channel_percentiles(
    raw_evidence: torch.Tensor, normal_mask: torch.Tensor
) -> tuple[StrictLowerNormalPercentile, StrictLowerNormalPercentile]:
    if raw_evidence.ndim != 2 or raw_evidence.shape[1] != 2:
        raise ValueError("raw graph evidence must have shape [N,2]")
    return (
        StrictLowerNormalPercentile.fit(
            raw_evidence[:, 0].abs(), normal_mask=normal_mask
        ),
        StrictLowerNormalPercentile.fit(
            raw_evidence[:, 1].abs(), normal_mask=normal_mask
        ),
    )


def build_routing_batch(
    *,
    raw_evidence: torch.Tensor,
    controlled_evidence: torch.Tensor,
    structural_statistics: torch.Tensor,
    base_anomaly_logit: torch.Tensor,
    normal_mask: torch.Tensor,
    normal_channel_percentiles: tuple[
        StrictLowerNormalPercentile, StrictLowerNormalPercentile
    ],
) -> RoutingBatch:
    if raw_evidence.ndim != 2 or raw_evidence.shape[1] != 2:
        raise ValueError("raw graph evidence must have shape [N,2]")
    if controlled_evidence.shape != raw_evidence.shape:
        raise ValueError("raw and controlled graph evidence must align")
    count = raw_evidence.shape[0]
    if structural_statistics.shape != (count, 3):
        raise ValueError("structural reliability input must have shape [N,3]")
    if base_anomaly_logit.shape != (count,) or normal_mask.shape != (count,):
        raise ValueError("node-aligned vectors must have shape [N]")
    tensors = (
        raw_evidence,
        controlled_evidence,
        structural_statistics,
        base_anomaly_logit,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("routing inputs must be finite")
    embedding = normal_channel_percentiles[0].transform(raw_evidence[:, 0].abs())
    decision = normal_channel_percentiles[1].transform(raw_evidence[:, 1].abs())
    return RoutingBatch(
        raw_evidence=raw_evidence.detach(),
        controlled_evidence=controlled_evidence.detach(),
        structural_statistics=structural_statistics.detach(),
        base_anomaly_logit=base_anomaly_logit.detach(),
        normal_mask=normal_mask.detach().to(dtype=torch.bool),
        normal_channel_support=torch.minimum(embedding, decision).detach(),
        normal_channel_embedding=embedding.detach(),
        normal_channel_decision=decision.detach(),
    )


@dataclass(frozen=True)
class NormalRouteCalibration:
    evidence_composite: StrictLowerNormalPercentile
    energy: StrictLowerNormalPercentile
    bound_composite: StrictLowerNormalPercentile


def fit_route_calibration(
    *,
    reliability: torch.Tensor,
    channel_support: torch.Tensor,
    suppression: torch.Tensor,
    raw_energy: torch.Tensor,
    controlled_energy: torch.Tensor,
    normal_mask: torch.Tensor,
) -> NormalRouteCalibration:
    evidence = geometric_three(reliability, channel_support, suppression)
    evidence_percentile = StrictLowerNormalPercentile.fit(
        evidence, normal_mask=normal_mask
    )
    maximum_energy = torch.maximum(raw_energy, controlled_energy)
    energy_percentile = StrictLowerNormalPercentile.fit(
        maximum_energy, normal_mask=normal_mask
    )
    energy_rank = energy_percentile.transform(maximum_energy)
    bound = geometric_three(reliability, channel_support, energy_rank)
    bound_percentile = StrictLowerNormalPercentile.fit(
        bound, normal_mask=normal_mask
    )
    return NormalRouteCalibration(
        evidence_composite=evidence_percentile,
        energy=energy_percentile,
        bound_composite=bound_percentile,
    )


def transform_routes(
    calibration: NormalRouteCalibration,
    *,
    reliability: torch.Tensor,
    channel_support: torch.Tensor,
    suppression: torch.Tensor,
    raw_energy: torch.Tensor,
    controlled_energy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    evidence = geometric_three(reliability, channel_support, suppression)
    evidence_rank = calibration.evidence_composite.transform(evidence)
    maximum_energy = torch.maximum(raw_energy, controlled_energy)
    energy_rank = calibration.energy.transform(maximum_energy)
    bound = geometric_three(reliability, channel_support, energy_rank)
    bound_rank = calibration.bound_composite.transform(bound)
    evidence_route = tail_ramp(evidence_rank).detach()
    bound_route = tail_ramp(bound_rank).detach()
    return evidence_route, bound_route, {
        "evidence_composite": evidence.detach(),
        "evidence_rank": evidence_rank.detach(),
        "maximum_energy": maximum_energy.detach(),
        "energy_rank": energy_rank.detach(),
        "bound_composite": bound.detach(),
        "bound_rank": bound_rank.detach(),
    }


class NormalCalibratedFactorizedRouter(nn.Module):
    """Parameter-free evidence/bound router calibrated on normal references."""

    def forward(
        self,
        *,
        reliability: torch.Tensor,
        channel_support: torch.Tensor,
        raw_energy: torch.Tensor,
        controlled_energy: torch.Tensor,
        reference_reliability: torch.Tensor,
        reference_channel_support: torch.Tensor,
        reference_raw_energy: torch.Tensor,
        reference_controlled_energy: torch.Tensor,
        reference_normal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        reference_suppression = suppression_ratio(
            reference_raw_energy, reference_controlled_energy
        )
        calibration = fit_route_calibration(
            reliability=reference_reliability,
            channel_support=reference_channel_support,
            suppression=reference_suppression,
            raw_energy=reference_raw_energy,
            controlled_energy=reference_controlled_energy,
            normal_mask=reference_normal_mask,
        )
        suppression = suppression_ratio(raw_energy, controlled_energy)
        evidence_route, bound_route, components = transform_routes(
            calibration,
            reliability=reliability,
            channel_support=channel_support,
            suppression=suppression,
            raw_energy=raw_energy,
            controlled_energy=controlled_energy,
        )
        components["suppression"] = suppression.detach()
        return evidence_route, bound_route, components


def correction_endpoints(
    *,
    gamma: torch.Tensor,
    reliability: torch.Tensor,
    raw_energy: torch.Tensor,
    controlled_energy: torch.Tensor,
    wide_temperature: float = F2_WIDE_TEMPERATURE,
) -> dict[str, torch.Tensor]:
    common = gamma * reliability
    temperature = float(wide_temperature)
    if temperature <= 0.0:
        raise ValueError("wide temperature must be positive")
    return {
        "controlled_tight": common * torch.tanh(controlled_energy),
        "controlled_wide": common
        * temperature
        * torch.tanh(controlled_energy / temperature),
        "raw_tight": common * torch.tanh(raw_energy),
        "raw_wide": common * temperature * torch.tanh(raw_energy / temperature),
    }


def factorized_correction(
    *,
    controlled_tight: torch.Tensor,
    controlled_wide: torch.Tensor,
    raw_tight: torch.Tensor,
    raw_wide: torch.Tensor,
    evidence_route: torch.Tensor,
    bound_route: torch.Tensor,
) -> torch.Tensor:
    evidence = evidence_route.detach()
    bound = bound_route.detach()
    return (
        (1.0 - evidence) * (1.0 - bound) * controlled_tight
        + (1.0 - evidence) * bound * controlled_wide
        + evidence * (1.0 - bound) * raw_tight
        + evidence * bound * raw_wide
    )
