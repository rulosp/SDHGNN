from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from .fixed_dataset import FixedHypergraphDataset, edge_key


def _new_base_graph() -> defaultdict[int, dict[str, set[int]]]:
    return defaultdict(
        lambda: {
            "pos_out": set(),
            "pos_in": set(),
            "neg_out": set(),
            "neg_in": set(),
            "undirected": set(),
        }
    )


@dataclass(frozen=True)
class SignedNetworkBuildConfig:
    dataset_name: str
    input_path: Path
    k_hop: int
    min_size: int
    max_size: int
    min_center_degree: int
    seed: int = 0

    def validate(self) -> None:
        if self.dataset_name not in {"wiki", "slashdot"}:
            raise ValueError("dataset_name must be 'wiki' or 'slashdot'.")
        if self.k_hop < 1:
            raise ValueError("k_hop must be positive.")
        if self.min_size < 1 or self.max_size < self.min_size:
            raise ValueError("Invalid hyperedge size limits.")
        if self.min_center_degree < 0:
            raise ValueError("min_center_degree must be non-negative.")


class WikiHypergraphBuilder:
    def __init__(self, config: SignedNetworkBuildConfig) -> None:
        self.config = config
        self.edge_list: list[dict[str, int]] = []
        self.node_labels_raw: dict[int, int] = {}
        self.base_graph = _new_base_graph()
        self.node_list: list[int] = []
        self.node_to_idx: dict[int, int] = {}

    def build(self) -> FixedHypergraphDataset:
        self._load_data()
        self._build_base_graph()
        source_incidence, target_incidence = self._generate_hyperedges()
        features = self._generate_features()
        hyperedges = self._extract_hyperedges(source_incidence, target_incidence)
        metadata = {
            "dataset_name": "wiki_fixed_hypergraph_for_hyperedge_prediction",
            "source_path": str(self.config.input_path),
            "k_hop": self.config.k_hop,
            "min_size": self.config.min_size,
            "max_size": self.config.max_size,
            "min_center_degree": self.config.min_center_degree,
            "construction_seed": self.config.seed,
            "sampling_profile": "signed_network",
            "class_names": {
                0: "fake",
                1: "positive_directed",
                2: "negative_directed",
                3: "mixed",
                4: "undirected",
            },
        }
        return FixedHypergraphDataset(
            source_incidence=source_incidence,
            target_incidence=target_incidence,
            node_features=features,
            hyperedges=hyperedges,
            num_nodes=len(self.node_list),
            node_list=list(self.node_list),
            metadata=metadata,
            base_graph=self.base_graph,
            raw_edge_count=len(self.edge_list),
            all_real_edge_keys={edge_key(edge["s"], edge["t"]) for edge in hyperedges},
        )

    def _load_data(self) -> None:
        input_path = self.config.input_path.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Wiki edge file not found: {input_path}")
        with input_path.open("r", encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    source, target, vote, result = map(int, parts[:4])
                except ValueError:
                    continue
                self.edge_list.append(
                    {"source": source, "target": target, "vote": vote}
                )
                self.node_labels_raw[target] = result

    def _build_base_graph(self) -> None:
        all_nodes: set[int] = set()
        for edge in self.edge_list:
            source = edge["source"]
            target = edge["target"]
            vote = edge["vote"]
            all_nodes.add(source)
            all_nodes.add(target)
            _ = self.base_graph[source]
            _ = self.base_graph[target]
            if vote == 1:
                self.base_graph[source]["pos_out"].add(target)
                self.base_graph[target]["pos_in"].add(source)
            elif vote == -1:
                self.base_graph[source]["neg_out"].add(target)
                self.base_graph[target]["neg_in"].add(source)
            elif vote == 0:
                self.base_graph[source]["undirected"].add(target)
                self.base_graph[target]["undirected"].add(source)
        self.node_list = sorted(all_nodes)
        self.node_to_idx = {node: index for index, node in enumerate(self.node_list)}

    def _expand_k_hop(self, center_node: int) -> dict[str, set[int]]:
        result = {
            "pos_in": set(),
            "neg_in": set(),
            "pos_out": set(),
            "neg_out": set(),
            "neu": set(),
        }

        def bfs_directed(
            positive_relation: str,
            negative_relation: str,
            positive_storage: set[int],
            negative_storage: set[int],
        ) -> None:
            queue = deque([(center_node, 1, 0)])
            visited = {(center_node, 1)}
            while queue:
                current, sign, depth = queue.popleft()
                if depth >= self.config.k_hop:
                    continue
                for neighbor in self.base_graph[current][positive_relation]:
                    new_sign = sign
                    if (neighbor, new_sign) not in visited:
                        visited.add((neighbor, new_sign))
                        queue.append((neighbor, new_sign, depth + 1))
                        (positive_storage if new_sign == 1 else negative_storage).add(
                            neighbor
                        )
                for neighbor in self.base_graph[current][negative_relation]:
                    new_sign = -sign
                    if (neighbor, new_sign) not in visited:
                        visited.add((neighbor, new_sign))
                        queue.append((neighbor, new_sign, depth + 1))
                        (positive_storage if new_sign == 1 else negative_storage).add(
                            neighbor
                        )

        def bfs_undirected(storage: set[int]) -> None:
            queue = deque([(center_node, 0)])
            visited = {center_node}
            while queue:
                current, depth = queue.popleft()
                if depth >= self.config.k_hop:
                    continue
                for neighbor in self.base_graph[current]["undirected"]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        storage.add(neighbor)

        bfs_directed("pos_in", "neg_in", result["pos_in"], result["neg_in"])
        bfs_directed("pos_out", "neg_out", result["pos_out"], result["neg_out"])
        bfs_undirected(result["neu"])
        return result

    def _generate_hyperedges(self) -> tuple[sp.coo_matrix, sp.coo_matrix]:
        source_rows: list[int] = []
        source_columns: list[int] = []
        source_values: list[float] = []
        target_rows: list[int] = []
        target_columns: list[int] = []
        target_values: list[float] = []
        edge_index = 0
        counts = {"pos": 0, "neg": 0, "mixed": 0, "undir": 0}

        for center in self.node_list:
            neighborhood = self.base_graph[center]
            degree = sum(
                len(neighborhood[relation])
                for relation in (
                    "pos_in",
                    "neg_in",
                    "pos_out",
                    "neg_out",
                    "undirected",
                )
            )
            if degree < self.config.min_center_degree:
                continue
            expanded = self._expand_k_hop(center)
            patterns = [
                (expanded["pos_in"], 1, expanded["pos_out"], 1, "pos"),
                (expanded["neg_in"], -1, expanded["neg_out"], -1, "neg"),
                (expanded["neg_in"], -1, expanded["pos_out"], 1, "mixed"),
                (expanded["pos_in"], 1, expanded["neg_out"], -1, "mixed"),
            ]
            for source_nodes, source_sign, target_nodes, target_sign, edge_type in patterns:
                if len(source_nodes) + len(target_nodes) < self.config.min_size:
                    continue
                if not source_nodes or not target_nodes:
                    continue
                if (
                    edge_type == "mixed"
                    and counts["mixed"] > counts["pos"] * 1.05
                    and random.random() < 0.75
                ):
                    continue
                sampled_source = random.sample(
                    list(source_nodes),
                    min(len(source_nodes), self.config.max_size),
                )
                sampled_target = random.sample(
                    list(target_nodes),
                    min(len(target_nodes), self.config.max_size),
                )
                for node in sampled_source:
                    source_rows.append(self.node_to_idx[node])
                    source_columns.append(edge_index)
                    source_values.append(float(source_sign))
                for node in sampled_target:
                    target_rows.append(self.node_to_idx[node])
                    target_columns.append(edge_index)
                    target_values.append(float(target_sign))
                edge_index += 1
                counts[edge_type] += 1

            neutral_nodes = list(expanded["neu"])
            if len(neutral_nodes) >= self.config.min_size:
                if (
                    counts["undir"] > counts["pos"] * 1.05
                    and random.random() < 0.6
                ):
                    continue
                sampled_neutral = random.sample(
                    neutral_nodes,
                    min(len(neutral_nodes), self.config.max_size),
                )
                for node in sampled_neutral:
                    node_index = self.node_to_idx[node]
                    source_rows.append(node_index)
                    source_columns.append(edge_index)
                    source_values.append(1.0)
                    target_rows.append(node_index)
                    target_columns.append(edge_index)
                    target_values.append(1.0)
                edge_index += 1
                counts["undir"] += 1

        shape = (len(self.node_list), edge_index)
        source_incidence = sp.coo_matrix(
            (source_values, (source_rows, source_columns)),
            shape=shape,
            dtype=np.float32,
        )
        target_incidence = sp.coo_matrix(
            (target_values, (target_rows, target_columns)),
            shape=shape,
            dtype=np.float32,
        )
        return source_incidence, target_incidence

    def _generate_features(self) -> np.ndarray:
        features = np.zeros((len(self.node_list), 5), dtype=np.float32)
        for index, node in enumerate(self.node_list):
            neighborhood = self.base_graph[node]
            features[index] = [
                len(neighborhood["pos_in"]),
                len(neighborhood["neg_in"]),
                len(neighborhood["pos_out"]),
                len(neighborhood["neg_out"]),
                len(neighborhood["undirected"]),
            ]
        return np.log1p(features.astype(np.float64)).astype(np.float32)

    @staticmethod
    def _extract_hyperedges(
        source_incidence: sp.spmatrix,
        target_incidence: sp.spmatrix,
    ) -> list[dict[str, Any]]:
        source_csc = source_incidence.tocsc()
        target_csc = target_incidence.tocsc()
        hyperedges: list[dict[str, Any]] = []
        for edge_index in range(source_csc.shape[1]):
            source_column = source_csc.getcol(edge_index)
            target_column = target_csc.getcol(edge_index)
            source_nodes = source_column.indices.tolist()
            target_nodes = target_column.indices.tolist()
            source_sign = (
                1
                if len(source_column.data) > 0 and np.mean(source_column.data) > 0
                else -1
            )
            is_undirected = bool(
                set(source_nodes) == set(target_nodes) and len(source_nodes) > 0
            )
            if is_undirected:
                edge_type = 4
            else:
                target_sign = (
                    1
                    if len(target_column.data) > 0 and np.mean(target_column.data) > 0
                    else -1
                )
                if source_sign != target_sign:
                    edge_type = 3
                elif source_sign > 0:
                    edge_type = 1
                else:
                    edge_type = 2
            if source_nodes:
                hyperedges.append(
                    {"s": source_nodes, "t": target_nodes, "type": edge_type}
                )
        return hyperedges


class SlashdotHypergraphBuilder:
    def __init__(self, config: SignedNetworkBuildConfig) -> None:
        self.config = config
        self.edge_list: list[dict[str, int]] = []
        self.base_graph = _new_base_graph()
        self.node_list: list[int] = []
        self.node_to_idx: dict[int, int] = {}
        self.reciprocal_pairs: set[tuple[int, int]] = set()

    def build(self) -> FixedHypergraphDataset:
        self._load_data()
        self._build_base_graph()
        source_incidence, target_incidence = self._generate_hyperedges()
        features = self._generate_features()
        hyperedges = self._extract_hyperedges(
            source_incidence,
            target_incidence,
        )
        metadata = {
            "dataset_name": "slashdot_fixed_hypergraph_for_hyperedge_prediction",
            "source_path": str(self.config.input_path),
            "k_hop": self.config.k_hop,
            "min_size": self.config.min_size,
            "max_size": self.config.max_size,
            "min_center_degree": self.config.min_center_degree,
            "construction_seed": self.config.seed,
            "sampling_profile": "signed_network",
            "class_names": {
                0: "fake",
                1: "positive_directed",
                2: "negative_directed",
                3: "mixed",
                4: "undirected",
            },
        }
        return FixedHypergraphDataset(
            source_incidence=source_incidence,
            target_incidence=target_incidence,
            node_features=features,
            hyperedges=hyperedges,
            num_nodes=len(self.node_list),
            node_list=list(self.node_list),
            metadata=metadata,
            base_graph=self.base_graph,
            raw_edge_count=len(self.edge_list),
            all_real_edge_keys={edge_key(edge["s"], edge["t"]) for edge in hyperedges},
        )

    def _load_data(self) -> None:
        input_path = self.config.input_path.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Slashdot edge file not found: {input_path}")
        raw_edges: set[tuple[int, int]] = set()
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                source, target, weight = map(int, parts[:3])
                if source == target:
                    continue
                self.edge_list.append(
                    {"u": source, "v": target, "w": weight}
                )
                raw_edges.add((source, target))
        for edge in self.edge_list:
            source = edge["u"]
            target = edge["v"]
            if (target, source) in raw_edges:
                self.reciprocal_pairs.add((min(source, target), max(source, target)))

    def _build_base_graph(self) -> None:
        all_nodes: set[int] = set()
        for edge in self.edge_list:
            source = edge["u"]
            target = edge["v"]
            weight = edge["w"]
            all_nodes.add(source)
            all_nodes.add(target)
            pair = (min(source, target), max(source, target))
            if pair in self.reciprocal_pairs:
                self.base_graph[source]["undirected"].add(target)
                self.base_graph[target]["undirected"].add(source)
            elif weight == 1:
                self.base_graph[source]["pos_out"].add(target)
                self.base_graph[target]["pos_in"].add(source)
            elif weight == -1:
                self.base_graph[source]["neg_out"].add(target)
                self.base_graph[target]["neg_in"].add(source)
        self.node_list = sorted(all_nodes)
        self.node_to_idx = {node: index for index, node in enumerate(self.node_list)}

    def _expand_k_hop(self, center_node: int) -> dict[str, set[int]]:
        result = {
            "pos_in": set(),
            "neg_in": set(),
            "pos_out": set(),
            "neg_out": set(),
            "neu": set(),
        }

        def bfs_directed(
            positive_relation: str,
            negative_relation: str,
            positive_storage: set[int],
            negative_storage: set[int],
        ) -> None:
            queue = deque([(center_node, 1, 0)])
            visited = {(center_node, 1)}
            while queue:
                current, sign, depth = queue.popleft()
                if depth >= self.config.k_hop:
                    continue
                for neighbor in self.base_graph[current][positive_relation]:
                    if (neighbor, sign) not in visited:
                        visited.add((neighbor, sign))
                        queue.append((neighbor, sign, depth + 1))
                        (positive_storage if sign == 1 else negative_storage).add(
                            neighbor
                        )
                for neighbor in self.base_graph[current][negative_relation]:
                    new_sign = -sign
                    if (neighbor, new_sign) not in visited:
                        visited.add((neighbor, new_sign))
                        queue.append((neighbor, new_sign, depth + 1))
                        (positive_storage if new_sign == 1 else negative_storage).add(
                            neighbor
                        )

        bfs_directed("pos_in", "neg_in", result["pos_in"], result["neg_in"])
        bfs_directed("pos_out", "neg_out", result["pos_out"], result["neg_out"])
        queue = deque([(center_node, 0)])
        visited = {center_node}
        while queue:
            current, depth = queue.popleft()
            if depth >= self.config.k_hop:
                continue
            for neighbor in self.base_graph[current]["undirected"]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    result["neu"].add(neighbor)
        return result

    def _generate_hyperedges(self) -> tuple[sp.coo_matrix, sp.coo_matrix]:
        source_rows: list[int] = []
        source_columns: list[int] = []
        source_values: list[float] = []
        target_rows: list[int] = []
        target_columns: list[int] = []
        target_values: list[float] = []
        edge_index = 0
        counts = {"pos": 0, "neg": 0, "mixed": 0, "undir": 0}

        for center in self.node_list:
            neighborhood = self.base_graph[center]
            degree = sum(
                len(neighborhood[relation])
                for relation in (
                    "pos_in",
                    "neg_in",
                    "pos_out",
                    "neg_out",
                    "undirected",
                )
            )
            if degree < self.config.min_center_degree:
                continue
            expanded = self._expand_k_hop(center)
            patterns = [
                (expanded["pos_in"], 1, expanded["pos_out"], 1, "pos"),
                (expanded["neg_in"], -1, expanded["neg_out"], -1, "neg"),
                (expanded["neg_in"], -1, expanded["pos_out"], 1, "mixed"),
                (expanded["pos_in"], 1, expanded["neg_out"], -1, "mixed"),
            ]
            for source_nodes, source_sign, target_nodes, target_sign, edge_type in patterns:
                if len(source_nodes) + len(target_nodes) < self.config.min_size:
                    continue
                if not source_nodes or not target_nodes:
                    continue
                if (
                    edge_type == "mixed"
                    and counts["mixed"] > counts["pos"] * 1.05
                    and random.random() < 0.75
                ):
                    continue
                sampled_source = random.sample(
                    list(source_nodes),
                    min(len(source_nodes), self.config.max_size),
                )
                sampled_target = random.sample(
                    list(target_nodes),
                    min(len(target_nodes), self.config.max_size),
                )
                for node in sampled_source:
                    source_rows.append(self.node_to_idx[node])
                    source_columns.append(edge_index)
                    source_values.append(float(source_sign))
                for node in sampled_target:
                    target_rows.append(self.node_to_idx[node])
                    target_columns.append(edge_index)
                    target_values.append(float(target_sign))
                edge_index += 1
                counts[edge_type] += 1
            if len(expanded["neu"]) >= self.config.min_size:
                if (
                    counts["undir"] > counts["pos"] * 1.05
                    and random.random() < 0.6
                ):
                    continue
                sampled_neutral = random.sample(
                    list(expanded["neu"]),
                    min(len(expanded["neu"]), self.config.max_size),
                )
                for node in sampled_neutral:
                    node_index = self.node_to_idx[node]
                    source_rows.append(node_index)
                    source_columns.append(edge_index)
                    source_values.append(1.0)
                    target_rows.append(node_index)
                    target_columns.append(edge_index)
                    target_values.append(1.0)
                edge_index += 1
                counts["undir"] += 1
        shape = (len(self.node_list), edge_index)
        return (
            sp.coo_matrix(
                (source_values, (source_rows, source_columns)),
                shape=shape,
                dtype=np.float32,
            ),
            sp.coo_matrix(
                (target_values, (target_rows, target_columns)),
                shape=shape,
                dtype=np.float32,
            ),
        )


    @staticmethod
    def _extract_hyperedges(
        source_incidence: sp.spmatrix,
        target_incidence: sp.spmatrix,
    ) -> list[dict[str, Any]]:
        source_csc = source_incidence.tocsc()
        target_csc = target_incidence.tocsc()
        hyperedges: list[dict[str, Any]] = []
        for edge_index in range(source_csc.shape[1]):
            source_column = source_csc.getcol(edge_index)
            target_column = target_csc.getcol(edge_index)
            source_nodes = set(source_column.indices.tolist())
            target_nodes = set(target_column.indices.tolist())
            source_sign = (
                1
                if len(source_column.data) > 0 and np.mean(source_column.data) > 0
                else -1
            )
            if source_nodes == target_nodes and source_nodes:
                edge_type = 4
            else:
                target_sign = (
                    1
                    if len(target_column.data) > 0 and np.mean(target_column.data) > 0
                    else -1
                )
                if source_sign != target_sign:
                    edge_type = 3
                elif source_sign > 0:
                    edge_type = 1
                else:
                    edge_type = 2
            if source_nodes:
                hyperedges.append(
                    {
                        "s": list(source_nodes),
                        "t": list(target_nodes),
                        "type": edge_type,
                    }
                )
        return hyperedges

    def _generate_features(self) -> np.ndarray:
        features = np.zeros((len(self.node_list), 5), dtype=np.float32)
        for index, node in enumerate(self.node_list):
            neighborhood = self.base_graph[node]
            features[index] = [
                len(neighborhood["pos_in"]),
                len(neighborhood["neg_in"]),
                len(neighborhood["pos_out"]),
                len(neighborhood["neg_out"]),
                len(neighborhood["undirected"]),
            ]
        return np.log1p(features.astype(np.float64)).astype(np.float32)


class SignedNetworkHypergraphBuilder:
    def __init__(self, config: SignedNetworkBuildConfig) -> None:
        config.validate()
        self.config = config

    def build(self) -> FixedHypergraphDataset:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        if self.config.dataset_name == "wiki":
            return WikiHypergraphBuilder(self.config).build()
        return SlashdotHypergraphBuilder(self.config).build()
