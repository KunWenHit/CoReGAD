from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from coregad.models.spectral_reference import (
    GLOBAL_SPECTRAL_LOGITS,
    SpectralDiscrepancy,
)
from coregad.scalable.memory import memory_snapshot
from coregad.scalable.sparse_operator import SparseGraphOperator
from coregad.scalable.spectral_cache import MemmapSpec, create_memmap


SPECTRAL_ROW_CHUNK = 65_536
SCALABLE_TRAIN_BATCH_SIZE = 8_192


@dataclass(frozen=True)
class ScalableSpectralArtifacts:
    low: MemmapSpec
    band: MemmapSpec
    high: MemmapSpec
    spectral_reference: MemmapSpec
    spectral_discrepancy: MemmapSpec
    structural_statistics: MemmapSpec
    metadata_path: Path
    metadata: dict[str, object]


def _load_full_to_device(
    values: np.ndarray,
    device: torch.device,
    *,
    row_chunk: int,
) -> torch.Tensor:
    output = torch.empty(
        tuple(int(value) for value in values.shape),
        dtype=torch.float32,
        device=device,
    )
    for begin in range(0, int(values.shape[0]), int(row_chunk)):
        end = min(int(values.shape[0]), begin + int(row_chunk))
        host = np.array(values[begin:end], dtype=np.float32, copy=True)
        output[begin:end].copy_(torch.from_numpy(host).to(device))
    return output


def compute_chunked_graph_features(
    *,
    operator: SparseGraphOperator,
    node_embeddings: torch.Tensor,
    normality_head: nn.Linear,
    cache_dir: str | Path,
    row_chunk: int = SPECTRAL_ROW_CHUNK,
    basis_cache_dtype: str = "float32",
) -> ScalableSpectralArtifacts:
    """Compute the frozen spectral and structural features by CSR row blocks.

    ``float32`` is the parity mode.  ``float16`` reproduces the historical
    large-graph basis-cache engineering path; arithmetic remains float32 and
    the quantized cache tolerance is reported separately.
    """

    if node_embeddings.ndim != 2:
        raise ValueError("node_embeddings must have shape [N, hidden_dim]")
    n, hidden = (int(node_embeddings.shape[0]), int(node_embeddings.shape[1]))
    if n != operator.num_nodes:
        raise ValueError("operator and embedding node counts differ")
    row_chunk = int(row_chunk)
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    cache_dtype = str(np.dtype(basis_cache_dtype))
    if cache_dtype not in {"float16", "float32"}:
        raise ValueError("basis_cache_dtype must be float16 or float32")
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    low_spec, low_disk = create_memmap(root / f"Z_low.{cache_dtype}.memmap", (n, hidden), cache_dtype)
    band_spec, band_disk = create_memmap(root / f"Z_band.{cache_dtype}.memmap", (n, hidden), cache_dtype)
    high_spec, high_disk = create_memmap(root / f"Z_high.{cache_dtype}.memmap", (n, hidden), cache_dtype)
    reference_spec, reference_disk = create_memmap(root / "spectral_reference.float32.memmap", (n, hidden), "float32")
    discrepancy_spec, discrepancy_disk = create_memmap(root / "spectral_discrepancy.float32.memmap", (n, 2), "float32")
    structural_spec, structural_disk = create_memmap(root / "structural_statistics.float32.memmap", (n, 3), "float32")
    device = node_embeddings.device
    started = time.time()
    before = memory_snapshot(device)
    weights = torch.softmax(
        torch.tensor(GLOBAL_SPECTRAL_LOGITS, dtype=torch.float32, device=device),
        dim=0,
    )
    discrepancy_module = SpectralDiscrepancy().to(device)
    normality_head = normality_head.to(device)
    normality_head.eval()
    standard_parity = cache_dtype == "float32"
    with torch.no_grad():
        for begin in range(0, n, row_chunk):
            end = min(n, begin + row_chunk)
            low = operator.propagate_rows(
                node_embeddings,
                begin,
                end,
                normalized=True,
                standard_parity=standard_parity,
            )
            low_disk[begin:end] = low.detach().cpu().numpy().astype(cache_dtype)
        low_disk.flush()
        low_full = _load_full_to_device(
            low_disk, device, row_chunk=row_chunk
        )
        for begin in range(0, n, row_chunk):
            end = min(n, begin + row_chunk)
            embeddings = node_embeddings[begin:end].float()
            low = low_full[begin:end]
            low_twice = operator.propagate_rows(
                low_full,
                begin,
                end,
                normalized=True,
                standard_parity=standard_parity,
            )
            band = low - low_twice
            high = embeddings - low
            reference = weights[0] * low + weights[1] * band + weights[2] * high
            if not standard_parity:
                components = {
                    "low_spectral_component": low,
                    "band_spectral_component": band,
                    "high_spectral_component": high,
                    "global_spectral_weights": weights,
                }
                discrepancy = discrepancy_module(
                    embeddings,
                    reference,
                    normality_head,
                    spectral_components=components,
                )["spectral_discrepancy"]
            degree = torch.from_numpy(
                operator.visible_degree[begin:end].astype(np.float32)
            ).to(device)
            support = torch.from_numpy(
                operator.support_ratio[begin:end].astype(np.float32)
            ).to(device)
            local_variation = torch.linalg.vector_norm(
                embeddings - low, dim=1
            ) / math.sqrt(hidden)
            structural = torch.stack(
                [torch.log1p(degree), support, local_variation], dim=1
            )
            band_disk[begin:end] = band.detach().cpu().numpy().astype(cache_dtype)
            high_disk[begin:end] = high.detach().cpu().numpy().astype(cache_dtype)
            reference_disk[begin:end] = reference.detach().cpu().numpy().astype(np.float32)
            if not standard_parity:
                discrepancy_disk[begin:end] = discrepancy.detach().cpu().numpy().astype(np.float32)
            structural_disk[begin:end] = structural.detach().cpu().numpy().astype(np.float32)
    if standard_parity:
        for values in (band_disk, high_disk, reference_disk):
            values.flush()
        # Discrepancy arithmetic is shape-sensitive at the last float32 bit.
        # Use the public standard module on one canonical chunk schedule after
        # all basis caches are complete, independent of graph row blocking.
        for begin in range(0, n, SPECTRAL_ROW_CHUNK):
            end = min(n, begin + SPECTRAL_ROW_CHUNK)
            embeddings = node_embeddings[begin:end].float()
            low = torch.from_numpy(
                np.array(low_disk[begin:end], dtype=np.float32, copy=True)
            ).to(device)
            band = torch.from_numpy(
                np.array(band_disk[begin:end], dtype=np.float32, copy=True)
            ).to(device)
            high = torch.from_numpy(
                np.array(high_disk[begin:end], dtype=np.float32, copy=True)
            ).to(device)
            reference = torch.from_numpy(
                np.array(reference_disk[begin:end], dtype=np.float32, copy=True)
            ).to(device)
            discrepancy = discrepancy_module(
                embeddings,
                reference,
                normality_head,
                spectral_components={
                    "low_spectral_component": low,
                    "band_spectral_component": band,
                    "high_spectral_component": high,
                    "global_spectral_weights": weights,
                },
            )["spectral_discrepancy"]
            discrepancy_disk[begin:end] = (
                discrepancy.detach().cpu().numpy().astype(np.float32)
            )
    for values in (
        band_disk,
        high_disk,
        reference_disk,
        discrepancy_disk,
        structural_disk,
    ):
        values.flush()
    cache_bytes = int(
        sum(
            spec.bytes
            for spec in (
                low_spec,
                band_spec,
                high_spec,
                reference_spec,
                discrepancy_spec,
                structural_spec,
            )
        )
    )
    metadata: dict[str, object] = {
        "engine": "SCALABLE",
        "num_nodes": n,
        "hidden_dim": hidden,
        "row_chunk": row_chunk,
        "train_batch_size": SCALABLE_TRAIN_BATCH_SIZE,
        "basis_cache_dtype": cache_dtype,
        "float32_computation": True,
        "historical_mixed_precision_cache": cache_dtype == "float16",
        "standard_arithmetic_parity": standard_parity,
        "cache_bytes": cache_bytes,
        "dense_nxn_constructed": False,
        "full_graph_device_coo_constructed": False,
        "graph_operator": operator.audit(),
        "runtime_seconds": float(time.time() - started),
        "memory_before": before,
        "memory_after": memory_snapshot(device),
    }
    metadata_path = root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del low_disk, band_disk, high_disk, reference_disk, discrepancy_disk, structural_disk, low_full
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ScalableSpectralArtifacts(
        low=low_spec,
        band=band_spec,
        high=high_spec,
        spectral_reference=reference_spec,
        spectral_discrepancy=discrepancy_spec,
        structural_statistics=structural_spec,
        metadata_path=metadata_path,
        metadata=metadata,
    )
