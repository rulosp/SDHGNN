from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

Hyperedge = dict[str, Any]
EdgeKey = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class FixedHypergraphDataset:
    source_incidence: sp.spmatrix
    target_incidence: sp.spmatrix
    node_features: np.ndarray
    hyperedges: list[Hyperedge]
    num_nodes: int
    node_list: list[Any]
    metadata: dict[str, Any]
    base_graph: Mapping[Any, Mapping[str, set[Any]]] | None = None
    raw_edge_count: int | None = None
    all_real_edge_keys: set[EdgeKey] | None = None

    @property
    def num_hyperedges(self) -> int:
        return len(self.hyperedges)

    @property
    def num_classes(self) -> int:
        labels = {int(edge["type"]) for edge in self.hyperedges}
        return max(labels | {0}) + 1


def edge_key(source: Sequence[int], target: Sequence[int]) -> EdgeKey:
    return tuple(sorted(map(int, source))), tuple(sorted(map(int, target)))


def _plain_base_graph(
    base_graph: Mapping[Any, Mapping[str, set[Any]]] | None,
) -> dict[Any, dict[str, set[Any]]] | None:
    if base_graph is None:
        return None
    return {
        node: {
            relation: set(neighbors)
            for relation, neighbors in neighborhood.items()
        }
        for node, neighborhood in base_graph.items()
    }


def _validate_hyperedge(edge: Mapping[str, Any], index: int, num_nodes: int) -> None:
    required = {"s", "t", "type"}
    missing = required - set(edge)
    if missing:
        raise ValueError(f"Hyperedge {index} is missing fields: {sorted(missing)}")
    for field in ("s", "t"):
        nodes = edge[field]
        if not isinstance(nodes, Sequence):
            raise TypeError(f"Hyperedge {index} field '{field}' must be a sequence.")
        invalid = [node for node in nodes if not 0 <= int(node) < num_nodes]
        if invalid:
            raise ValueError(
                f"Hyperedge {index} contains node indices outside [0, {num_nodes})."
            )


def validate_fixed_dataset(dataset: FixedHypergraphDataset) -> None:
    if not sp.issparse(dataset.source_incidence) or not sp.issparse(
        dataset.target_incidence
    ):
        raise TypeError("Incidence matrices must be SciPy sparse matrices.")
    if dataset.source_incidence.shape != dataset.target_incidence.shape:
        raise ValueError("Source and target incidence matrices must have equal shapes.")
    if dataset.source_incidence.shape[0] != dataset.num_nodes:
        raise ValueError("Incidence row count does not match num_nodes.")
    if dataset.source_incidence.shape[1] != dataset.num_hyperedges:
        raise ValueError("Incidence column count does not match the hyperedge list.")
    features = np.asarray(dataset.node_features)
    if features.ndim != 2 or features.shape[0] != dataset.num_nodes:
        raise ValueError("Node features must have shape [num_nodes, num_features].")
    if not np.isfinite(features).all():
        raise ValueError("Node features contain non-finite values.")
    if len(dataset.node_list) != dataset.num_nodes:
        raise ValueError("node_list length does not match num_nodes.")
    for index, edge in enumerate(dataset.hyperedges):
        _validate_hyperedge(edge, index, dataset.num_nodes)


def save_fixed_hypergraph_dataset(
    dataset: FixedHypergraphDataset,
    path: str | Path,
    overwrite: bool = False,
) -> Path:
    validate_fixed_dataset(dataset)
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination}. Use overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "Hs": dataset.source_incidence.tocoo().astype(np.float32),
        "Ht": dataset.target_incidence.tocoo().astype(np.float32),
        "X": np.asarray(dataset.node_features, dtype=np.float32),
        "hyperedges": [dict(edge) for edge in dataset.hyperedges],
        "num_nodes": int(dataset.num_nodes),
        "node_list": list(dataset.node_list),
        "base_graph": _plain_base_graph(dataset.base_graph),
        "raw_edge_count": dataset.raw_edge_count,
        "all_real_edge_keys": dataset.all_real_edge_keys,
    }
    payload = {"meta": dict(dataset.metadata), "data": data}

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        with open(temporary_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return destination


def load_fixed_hypergraph_dataset(path: str | Path) -> FixedHypergraphDataset:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Fixed hypergraph dataset not found: {source}")
    with source.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "data" in payload:
        metadata = dict(payload.get("meta", {}))
        data = payload["data"]
    else:
        metadata = {}
        data = payload
    if not isinstance(data, dict):
        raise TypeError("The fixed dataset payload must be a dictionary.")

    required = {"Hs", "Ht", "X", "hyperedges", "num_nodes", "node_list"}
    missing = required - set(data)
    if missing:
        raise KeyError(f"Fixed dataset is missing fields: {sorted(missing)}")

    hyperedges = [dict(edge) for edge in data["hyperedges"]]
    stored_keys = data.get("all_real_edge_keys")
    all_real_edge_keys = (
        {tuple(map(tuple, key)) for key in stored_keys}
        if stored_keys is not None
        else {edge_key(edge["s"], edge["t"]) for edge in hyperedges}
    )
    dataset = FixedHypergraphDataset(
        source_incidence=data["Hs"].tocoo().astype(np.float32),
        target_incidence=data["Ht"].tocoo().astype(np.float32),
        node_features=np.asarray(data["X"], dtype=np.float32),
        hyperedges=hyperedges,
        num_nodes=int(data["num_nodes"]),
        node_list=list(data["node_list"]),
        metadata=metadata,
        base_graph=data.get("base_graph"),
        raw_edge_count=data.get("raw_edge_count"),
        all_real_edge_keys=all_real_edge_keys,
    )
    validate_fixed_dataset(dataset)
    return dataset
