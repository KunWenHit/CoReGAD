from __future__ import annotations

import math

import torch
from torch import nn

from .reliability import SpectralResidualEnergyHead, StructuralReliabilityGate
from .routing import (
    NormalCalibratedFactorizedRouter,
    RoutingBatch,
    correction_endpoints,
    factorized_correction,
)


FULL_F2 = "FULL_F2"
WO_M1_GRAPH_RESIDUAL_EVIDENCE = "WO_M1_GRAPH_RESIDUAL_EVIDENCE"
WO_M2_RELIABILITY = "WO_M2_RELIABILITY"
WO_M3_FACTORIZED_ROUTING = "WO_M3_FACTORIZED_ROUTING"
LEGACY_CONTROLLED_TIGHT = "LEGACY_CONTROLLED_TIGHT"
MODEL_VARIANTS = (
    FULL_F2,
    WO_M1_GRAPH_RESIDUAL_EVIDENCE,
    WO_M2_RELIABILITY,
    WO_M3_FACTORIZED_ROUTING,
    LEGACY_CONTROLLED_TIGHT,
)


def normalize_model_variant(variant: str) -> str:
    aliases = {
        "f2": FULL_F2,
        "full_f2": FULL_F2,
        "legacy": LEGACY_CONTROLLED_TIGHT,
        "legacy_controlled_tight": LEGACY_CONTROLLED_TIGHT,
        "wo_m1": WO_M1_GRAPH_RESIDUAL_EVIDENCE,
        "wo_m2": WO_M2_RELIABILITY,
        "wo_m3": WO_M3_FACTORIZED_ROUTING,
    }
    normalized = aliases.get(
        str(variant).strip().lower(), str(variant).strip().upper()
    )
    if normalized not in MODEL_VARIANTS:
        raise ValueError(f"unknown CoReGAD model variant: {variant}")
    return normalized


class CoReGAD(nn.Module):
    """CoReGAD-F2 residual detector with parameter-free factorized routing."""

    def __init__(self, *, model_variant: str = FULL_F2) -> None:
        super().__init__()
        self.model_variant = normalize_model_variant(model_variant)
        self.spectral_residual_energy = SpectralResidualEnergyHead()
        self.structural_reliability_gate: StructuralReliabilityGate | None
        if self.model_variant == WO_M2_RELIABILITY:
            self.structural_reliability_gate = None
        else:
            self.structural_reliability_gate = StructuralReliabilityGate()
        self.factorized_router = NormalCalibratedFactorizedRouter()
        self.gamma_raw = nn.Parameter(
            torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32)
        )

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_raw)

    @property
    def m3_trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.factorized_router.parameters()
        )

    def _energies_and_endpoints(
        self, batch: RoutingBatch
    ) -> dict[str, torch.Tensor]:
        raw_energy = self.spectral_residual_energy(batch.raw_evidence)
        controlled_energy = self.spectral_residual_energy(batch.controlled_evidence)
        if self.structural_reliability_gate is None:
            reliability = torch.ones_like(raw_energy)
        else:
            reliability = self.structural_reliability_gate(
                batch.structural_statistics
            )
        return {
            "raw_energy": raw_energy,
            "controlled_energy": controlled_energy,
            "reliability": reliability,
            **correction_endpoints(
                gamma=self.gamma,
                reliability=reliability,
                raw_energy=raw_energy,
                controlled_energy=controlled_energy,
            ),
        }

    def forward_batch(
        self,
        batch: RoutingBatch,
        *,
        calibration_batch: RoutingBatch | None = None,
        model_variant: str | None = None,
    ) -> dict[str, torch.Tensor | str]:
        variant = normalize_model_variant(model_variant or self.model_variant)
        if (
            variant == WO_M2_RELIABILITY
            and self.structural_reliability_gate is not None
        ):
            raise ValueError(
                "WO_M2 requires a detector constructed without reliability"
            )
        if (
            variant != WO_M2_RELIABILITY
            and self.structural_reliability_gate is None
        ):
            raise ValueError("this detector was constructed without reliability")

        if variant == WO_M1_GRAPH_RESIDUAL_EVIDENCE:
            zero = torch.zeros_like(batch.base_anomaly_logit)
            one = torch.ones_like(batch.base_anomaly_logit)
            final_logit = batch.base_anomaly_logit
            return {
                "model_variant": variant,
                "raw_spectral_evidence": batch.raw_evidence,
                "controlled_spectral_residual": batch.controlled_evidence,
                "structural_statistics": batch.structural_statistics,
                "raw_energy": zero,
                "controlled_energy": zero,
                "spectral_residual_energy": zero,
                "reliability": one,
                "structural_reliability": one,
                "controlled_tight": zero,
                "controlled_wide": zero,
                "raw_tight": zero,
                "raw_wide": zero,
                "evidence_route": zero,
                "bound_route": zero,
                "gamma": self.gamma,
                "graph_correction": zero,
                "base_anomaly_logit": batch.base_anomaly_logit,
                "base_anomaly_score": torch.sigmoid(batch.base_anomaly_logit),
                "final_anomaly_logit": final_logit,
                "final_anomaly_score": torch.sigmoid(final_logit),
            }

        endpoints = self._energies_and_endpoints(batch)
        if variant in {WO_M3_FACTORIZED_ROUTING, LEGACY_CONTROLLED_TIGHT}:
            evidence_route = torch.zeros_like(endpoints["raw_energy"])
            bound_route = torch.zeros_like(endpoints["raw_energy"])
            route_components: dict[str, torch.Tensor] = {}
        else:
            reference_batch = (
                calibration_batch if calibration_batch is not None else batch
            )
            reference = (
                endpoints
                if reference_batch is batch
                else self._energies_and_endpoints(reference_batch)
            )
            evidence_route, bound_route, route_components = self.factorized_router(
                reliability=endpoints["reliability"],
                channel_support=batch.normal_channel_support,
                raw_energy=endpoints["raw_energy"],
                controlled_energy=endpoints["controlled_energy"],
                reference_reliability=reference["reliability"],
                reference_channel_support=reference_batch.normal_channel_support,
                reference_raw_energy=reference["raw_energy"],
                reference_controlled_energy=reference["controlled_energy"],
                reference_normal_mask=reference_batch.normal_mask,
            )
        graph_correction = factorized_correction(
            controlled_tight=endpoints["controlled_tight"],
            controlled_wide=endpoints["controlled_wide"],
            raw_tight=endpoints["raw_tight"],
            raw_wide=endpoints["raw_wide"],
            evidence_route=evidence_route,
            bound_route=bound_route,
        )
        final_logit = batch.base_anomaly_logit + graph_correction
        return {
            "model_variant": variant,
            "raw_spectral_evidence": batch.raw_evidence,
            "controlled_spectral_residual": batch.controlled_evidence,
            "structural_statistics": batch.structural_statistics,
            **endpoints,
            **route_components,
            "spectral_residual_energy": endpoints["controlled_energy"],
            "structural_reliability": endpoints["reliability"],
            "evidence_route": evidence_route,
            "bound_route": bound_route,
            "gamma": self.gamma,
            "graph_correction": graph_correction,
            "base_anomaly_logit": batch.base_anomaly_logit,
            "base_anomaly_score": torch.sigmoid(batch.base_anomaly_logit),
            "final_anomaly_logit": final_logit,
            "final_anomaly_score": torch.sigmoid(final_logit),
        }

    def forward(
        self,
        controlled_spectral_residual: torch.Tensor,
        structural_statistics: torch.Tensor,
        base_anomaly_logit: torch.Tensor,
        *,
        raw_spectral_evidence: torch.Tensor | None = None,
        normal_mask: torch.Tensor | None = None,
        normal_channel_support: torch.Tensor | None = None,
        calibration_batch: RoutingBatch | None = None,
        model_variant: str | None = None,
    ) -> dict[str, torch.Tensor | str]:
        """Run F2, preserving the historical three-tensor call as legacy mode.

        The production pipeline supplies raw evidence and a normal-only calibration
        batch. Existing callers that provide only the original three positional
        tensors retain the released Controlled+Tight behavior.
        """

        requested = normalize_model_variant(model_variant or self.model_variant)
        legacy_call = raw_spectral_evidence is None
        effective = LEGACY_CONTROLLED_TIGHT if legacy_call else requested
        raw = (
            controlled_spectral_residual
            if raw_spectral_evidence is None
            else raw_spectral_evidence
        )
        count = controlled_spectral_residual.shape[0]
        if normal_mask is None:
            normal_mask = torch.zeros(
                count,
                dtype=torch.bool,
                device=controlled_spectral_residual.device,
            )
        if normal_channel_support is None:
            normal_channel_support = torch.ones(
                count,
                dtype=controlled_spectral_residual.dtype,
                device=controlled_spectral_residual.device,
            )
        batch = RoutingBatch(
            raw_evidence=raw.detach(),
            controlled_evidence=controlled_spectral_residual.detach(),
            structural_statistics=structural_statistics.detach(),
            base_anomaly_logit=base_anomaly_logit.detach(),
            normal_mask=normal_mask.detach().to(dtype=torch.bool),
            normal_channel_support=normal_channel_support.detach(),
            normal_channel_embedding=normal_channel_support.detach(),
            normal_channel_decision=normal_channel_support.detach(),
        )
        return self.forward_batch(
            batch,
            calibration_batch=calibration_batch,
            model_variant=effective,
        )
