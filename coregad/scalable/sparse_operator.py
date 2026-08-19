from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


HASH_ROW_CHUNK = 262_144


def _as_numpy_ids(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64).reshape(-1)


@dataclass(frozen=True)
class SparseGraphOperator:
    """CSR-oriented, fold-safe graph operator.

    Only row slices are materialized as torch sparse COO tensors.  The full
    graph is never copied to a device and no dense ``N x N`` object exists.
    """

    num_nodes: int
    row_ptr: np.ndarray
    destinations: np.ndarray
    visible_degree: np.ndarray
    degree_with_self: np.ndarray
    original_degree: np.ndarray
    support_ratio: np.ndarray
    graph_sha256: str
    heldout_to_heldout_edges: int

    @property
    def nnz_without_self(self) -> int:
        return int(self.destinations.size)

    @property
    def nnz_with_self(self) -> int:
        return int(self.destinations.size + self.num_nodes)

    @property
    def storage_bytes(self) -> int:
        arrays = (
            self.row_ptr,
            self.destinations,
            self.visible_degree,
            self.degree_with_self,
            self.original_degree,
            self.support_ratio,
        )
        return int(sum(value.nbytes for value in arrays))

    def row_slice(
        self, begin: int, end: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        begin, end = int(begin), int(end)
        if begin < 0 or end < begin or end > self.num_nodes:
            raise IndexError(f"invalid CSR row slice [{begin}, {end})")
        lo, hi = int(self.row_ptr[begin]), int(self.row_ptr[end])
        counts = np.diff(self.row_ptr[begin : end + 1]).astype(
            np.int64, copy=False
        )
        local_rows = np.repeat(np.arange(end - begin, dtype=np.int64), counts)
        columns = self.destinations[lo:hi].astype(np.int64, copy=False)
        global_rows = local_rows + begin
        values = 1.0 / np.sqrt(
            self.degree_with_self[global_rows].astype(np.float64)
            * self.degree_with_self[columns].astype(np.float64)
        )
        return local_rows, columns, values.astype(np.float32)

    def propagate_rows(
        self,
        features: torch.Tensor,
        begin: int,
        end: int,
        *,
        normalized: bool = True,
        standard_parity: bool = False,
    ) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[0]) != self.num_nodes:
            raise ValueError("features must have shape [num_nodes, hidden_dim]")
        begin, end = int(begin), int(end)
        rows, columns, normalized_values = self.row_slice(begin, end)
        block_rows = end - begin
        if standard_parity and normalized:
            row_tensor = torch.from_numpy(rows).to(features.device)
            column_tensor = torch.from_numpy(columns).to(features.device)
            global_rows = row_tensor + begin
            degree = torch.from_numpy(self.degree_with_self.astype(np.float32)).to(
                features.device
            )
            sparse_values = torch.rsqrt(
                degree[global_rows] * degree[column_tensor]
            ).to(dtype=features.dtype)
            self_rows = torch.arange(
                end - begin, device=features.device, dtype=torch.long
            )
            self_columns = self_rows + begin
            self_values = torch.rsqrt(
                degree[self_columns] * degree[self_columns]
            ).to(dtype=features.dtype)
            indices = torch.stack(
                [
                    torch.cat([row_tensor, self_rows]),
                    torch.cat([column_tensor, self_columns]),
                ],
                dim=0,
            )
            block = torch.sparse_coo_tensor(
                indices,
                torch.cat([sparse_values, self_values]),
                (end - begin, self.num_nodes),
                device=features.device,
                dtype=features.dtype,
                check_invariants=False,
            ).coalesce()
            return torch.sparse.mm(block, features)
        if columns.size:
            indices = torch.stack(
                [
                    torch.from_numpy(rows).to(features.device),
                    torch.from_numpy(columns).to(features.device),
                ],
                dim=0,
            )
            values = normalized_values if normalized else np.ones_like(normalized_values)
            sparse_values = torch.from_numpy(values).to(
                device=features.device, dtype=features.dtype
            )
            block = torch.sparse_coo_tensor(
                indices,
                sparse_values,
                (block_rows, self.num_nodes),
                device=features.device,
                dtype=features.dtype,
                check_invariants=False,
            ).coalesce()
            output = torch.sparse.mm(block, features)
        else:
            output = torch.zeros(
                (block_rows, int(features.shape[1])),
                device=features.device,
                dtype=features.dtype,
            )
        if normalized:
            self_weight = torch.from_numpy(
                (1.0 / self.degree_with_self[begin:end]).astype(np.float32)
            ).to(device=features.device, dtype=features.dtype)
            output = output + self_weight[:, None] * features[begin:end]
        return output

    def audit(self) -> dict[str, int | str | bool]:
        return {
            "representation": "csr_row_oriented",
            "num_nodes": self.num_nodes,
            "nnz_without_self": self.nnz_without_self,
            "nnz_with_self": self.nnz_with_self,
            "storage_bytes": self.storage_bytes,
            "graph_sha256": self.graph_sha256,
            "heldout_to_heldout_edges": self.heldout_to_heldout_edges,
            "dense_nxn_constructed": False,
            "full_graph_device_coo_constructed": False,
        }


def _hash_operator(
    row_ptr: np.ndarray,
    destinations: np.ndarray,
    degree_with_self: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    n = int(degree_with_self.size)
    digest.update(np.asarray([n], dtype=np.int64).tobytes())
    for begin in range(0, n, HASH_ROW_CHUNK):
        end = min(n, begin + HASH_ROW_CHUNK)
        lo, hi = int(row_ptr[begin]), int(row_ptr[end])
        counts = np.diff(row_ptr[begin : end + 1])
        rows = np.repeat(np.arange(begin, end, dtype=np.int64), counts)
        columns = destinations[lo:hi].astype(np.int64, copy=False)
        values = 1.0 / np.sqrt(
            degree_with_self[rows].astype(np.float64)
            * degree_with_self[columns].astype(np.float64)
        )
        digest.update(rows.tobytes())
        digest.update(columns.tobytes())
        digest.update(values.astype(np.float32).tobytes())
    nodes = np.arange(n, dtype=np.int64)
    digest.update(nodes.tobytes())
    digest.update(nodes.tobytes())
    digest.update(
        (1.0 / degree_with_self.astype(np.float64)).astype(np.float32).tobytes()
    )
    return digest.hexdigest()


def build_sparse_graph_operator(
    edge_index: torch.Tensor | np.ndarray,
    num_nodes: int,
    visible_nodes: torch.Tensor | np.ndarray,
    heldout_nodes: torch.Tensor | np.ndarray,
) -> SparseGraphOperator:
    """Build the exact released fold-safe operator in CPU CSR form."""

    n = int(num_nodes)
    visible_ids = _as_numpy_ids(visible_nodes)
    heldout_ids = _as_numpy_ids(heldout_nodes)
    visible = np.zeros(n, dtype=bool)
    heldout = np.zeros(n, dtype=bool)
    visible[visible_ids] = True
    heldout[heldout_ids] = True
    if isinstance(edge_index, torch.Tensor):
        edges = edge_index.detach().cpu().numpy()
    else:
        edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    source_all = np.asarray(edges[0], dtype=np.int64)
    destination_all = np.asarray(edges[1], dtype=np.int64)
    in_bounds = (
        (source_all >= 0)
        & (source_all < n)
        & (destination_all >= 0)
        & (destination_all < n)
    )
    if not bool(np.all(in_bounds)):
        raise ValueError("edge_index contains out-of-range node IDs")
    nonself = source_all != destination_all
    original_degree = np.bincount(
        source_all[nonself], minlength=n
    ).astype(np.int64)
    keep = (
        nonself
        & visible[destination_all]
        & (visible[source_all] | heldout[source_all])
    )
    source = source_all[keep]
    destinations = destination_all[keep]
    if source.size and not bool(np.all(source[:-1] <= source[1:])):
        order = np.argsort(source, kind="stable")
        source = source[order]
        destinations = destinations[order]
    counts = np.bincount(source, minlength=n).astype(np.int64)
    row_ptr = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
    )
    destination_dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
    destinations = destinations.astype(destination_dtype, copy=False)
    centers = np.repeat(np.arange(n, dtype=np.int64), counts)
    heldout_to_heldout = int(
        np.sum(heldout[centers] & heldout[destinations.astype(np.int64, copy=False)])
    )
    if heldout_to_heldout:
        raise AssertionError("fold-safe graph contains a heldout-to-heldout edge")
    degree_with_self = counts + 1
    denominator = np.maximum(original_degree, 1)
    support_ratio = np.divide(
        counts,
        denominator,
        out=np.zeros(n, dtype=np.float64),
        where=denominator > 0,
    ).clip(0.0, 1.0).astype(np.float32)
    return SparseGraphOperator(
        num_nodes=n,
        row_ptr=row_ptr,
        destinations=destinations,
        visible_degree=counts,
        degree_with_self=degree_with_self,
        original_degree=original_degree,
        support_ratio=support_ratio,
        graph_sha256=_hash_operator(row_ptr, destinations, degree_with_self),
        heldout_to_heldout_edges=heldout_to_heldout,
    )
