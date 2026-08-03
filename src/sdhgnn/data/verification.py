from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp

from .fixed_dataset import FixedHypergraphDataset


def _hash_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return _hash_bytes(
        str(contiguous.shape).encode("utf-8"),
        str(contiguous.dtype).encode("utf-8"),
        contiguous.tobytes(),
    )


def hash_sparse_matrix(matrix: sp.spmatrix) -> str:
    csr = matrix.tocsr(copy=True)
    csr.sum_duplicates()
    csr.sort_indices()
    csr.eliminate_zeros()
    return _hash_bytes(
        str(csr.shape).encode("utf-8"),
        str(csr.dtype).encode("utf-8"),
        np.ascontiguousarray(csr.indptr).tobytes(),
        np.ascontiguousarray(csr.indices).tobytes(),
        np.ascontiguousarray(csr.data).tobytes(),
    )


def _canonical_base_graph(
    base_graph: Mapping[Any, Mapping[str, set[Any]]] | None,
) -> list[tuple[Any, tuple[tuple[str, tuple[Any, ...]], ...]]] | None:
    if base_graph is None:
        return None
    result = []
    for node in sorted(base_graph):
        relations = tuple(
            (relation, tuple(sorted(neighbors)))
            for relation, neighbors in sorted(base_graph[node].items())
        )
        result.append((node, relations))
    return result


def hash_json_value(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_hashes(dataset: FixedHypergraphDataset) -> dict[str, str]:
    return {
        "node_list": hash_json_value(dataset.node_list),
        "node_features": hash_array(np.asarray(dataset.node_features)),
        "source_incidence": hash_sparse_matrix(dataset.source_incidence),
        "target_incidence": hash_sparse_matrix(dataset.target_incidence),
        "hyperedges": hash_json_value(dataset.hyperedges),
        "base_graph": hash_json_value(_canonical_base_graph(dataset.base_graph)),
    }


def _sparse_equal(left: sp.spmatrix, right: sp.spmatrix) -> bool:
    left_csr = left.tocsr(copy=True)
    right_csr = right.tocsr(copy=True)
    left_csr.sum_duplicates()
    right_csr.sum_duplicates()
    left_csr.eliminate_zeros()
    right_csr.eliminate_zeros()
    if left_csr.shape != right_csr.shape:
        return False
    difference = left_csr - right_csr
    difference.eliminate_zeros()
    return difference.nnz == 0


def compare_fixed_datasets(
    reference: FixedHypergraphDataset,
    candidate: FixedHypergraphDataset,
) -> dict[str, Any]:
    reference_features = np.asarray(reference.node_features)
    candidate_features = np.asarray(candidate.node_features)
    feature_shape_equal = reference_features.shape == candidate_features.shape
    feature_exact = feature_shape_equal and np.array_equal(
        reference_features,
        candidate_features,
    )
    max_feature_difference = None
    if feature_shape_equal and reference_features.size:
        max_feature_difference = float(
            np.max(np.abs(reference_features - candidate_features))
        )

    checks = {
        "num_nodes": reference.num_nodes == candidate.num_nodes,
        "num_hyperedges": reference.num_hyperedges == candidate.num_hyperedges,
        "node_list": reference.node_list == candidate.node_list,
        "node_features": feature_exact,
        "source_incidence": _sparse_equal(
            reference.source_incidence,
            candidate.source_incidence,
        ),
        "target_incidence": _sparse_equal(
            reference.target_incidence,
            candidate.target_incidence,
        ),
        "hyperedges": reference.hyperedges == candidate.hyperedges,
        "base_graph": _canonical_base_graph(reference.base_graph)
        == _canonical_base_graph(candidate.base_graph),
        "raw_edge_count": reference.raw_edge_count == candidate.raw_edge_count,
    }
    return {
        "exact_match": all(checks.values()),
        "checks": checks,
        "max_node_feature_absolute_difference": max_feature_difference,
        "reference_hashes": dataset_hashes(reference),
        "candidate_hashes": dataset_hashes(candidate),
    }
