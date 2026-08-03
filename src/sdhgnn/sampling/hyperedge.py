from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

Hyperedge = Mapping[str, Any]
EdgeKey = tuple[tuple[int, ...], tuple[int, ...]]


class HyperedgeSampler:
    MODES = {"random", "local", "perturb", "mixed"}
    PROFILES = {"signed_network", "reactome"}

    def __init__(
        self,
        reference_edges: Sequence[Hyperedge],
        num_nodes: int,
        real_edges_set: set[EdgeKey] | None = None,
        mode: str = "mixed",
        negative_ratio: float = 0.2,
        seed: int = 0,
        profile: str = "signed_network",
        base_graph: Mapping[Any, Mapping[str, set[Any]]] | None = None,
        node_list: Sequence[Any] | None = None,
        max_attempts_per_negative: int = 10_000,
        python_rng: random.Random | None = None,
        numpy_rng: np.random.RandomState | None = None,
    ) -> None:
        if not reference_edges:
            raise ValueError("reference_edges must not be empty.")
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive.")
        if mode not in self.MODES:
            raise ValueError(f"Unsupported sampling mode: {mode}")
        if profile not in self.PROFILES:
            raise ValueError(f"Unsupported sampling profile: {profile}")
        if negative_ratio < 0.0:
            raise ValueError("negative_ratio must be non-negative.")
        if max_attempts_per_negative < 1:
            raise ValueError("max_attempts_per_negative must be positive.")

        self.reference_edges = [dict(edge) for edge in reference_edges]
        self.num_nodes = int(num_nodes)
        self.mode = mode
        self.negative_ratio = float(negative_ratio)
        self.profile = profile
        self.base_graph = base_graph
        self.node_list = list(node_list) if node_list is not None else None
        self.node_to_idx = (
            {node_id: index for index, node_id in enumerate(self.node_list)}
            if self.node_list is not None
            else {}
        )
        self.max_attempts_per_negative = int(max_attempts_per_negative)
        self.real_edges_set = (
            set(real_edges_set)
            if real_edges_set is not None
            else self.get_real_edges_set(reference_edges)
        )
        self.source_sizes = [
            len(edge["s"]) for edge in reference_edges if len(edge["s"]) > 0
        ]
        self.target_sizes = [
            len(edge["t"]) for edge in reference_edges if len(edge["t"]) > 0
        ]
        if profile == "reactome":
            if not self.source_sizes:
                self.source_sizes = [2]
            if not self.target_sizes:
                self.target_sizes = [1]
        else:
            if not self.source_sizes:
                raise ValueError("At least one source set must be non-empty.")
            if not self.target_sizes:
                self.target_sizes = list(self.source_sizes)

        self.python_rng = python_rng or random.Random(seed)
        self.numpy_rng = numpy_rng or np.random.RandomState(seed)
        self.node_neighbors = (
            self._build_reactome_neighbors(reference_edges, num_nodes)
            if profile == "reactome"
            else None
        )

    @staticmethod
    def edge_key(source: Sequence[int], target: Sequence[int]) -> EdgeKey:
        return tuple(sorted(map(int, source))), tuple(sorted(map(int, target)))

    @classmethod
    def get_real_edges_set(cls, edges: Sequence[Hyperedge]) -> set[EdgeKey]:
        return {cls.edge_key(edge["s"], edge["t"]) for edge in edges}

    @staticmethod
    def _deduplicate(nodes: Sequence[int]) -> list[int]:
        return list(dict.fromkeys(map(int, nodes)))

    @staticmethod
    def _build_reactome_neighbors(
        edges: Sequence[Hyperedge],
        num_nodes: int,
    ) -> dict[int, set[int]]:
        neighbors: defaultdict[int, set[int]] = defaultdict(set)
        for node in range(num_nodes):
            neighbors[node]
        for edge in edges:
            nodes = list(set(map(int, edge["s"] + edge["t"])))
            for source in nodes:
                for target in nodes:
                    if source != target:
                        neighbors[source].add(target)
        return dict(neighbors)

    def _sample_node_indices(self, size: int, replace: bool = False) -> list[int]:
        size = min(int(size), self.num_nodes) if not replace else int(size)
        return self.numpy_rng.choice(
            self.num_nodes,
            size=size,
            replace=replace,
        ).tolist()

    def _signed_random_negative(self) -> tuple[list[int], list[int]]:
        source_size = self.python_rng.choice(self.source_sizes)
        target_size = self.python_rng.choice(self.target_sizes)
        source = self._sample_node_indices(source_size)
        target = (
            list(source)
            if self.python_rng.random() < 0.3
            else self._sample_node_indices(target_size)
        )
        return self._deduplicate(source), self._deduplicate(target)

    def _signed_local_negative(self) -> tuple[list[int], list[int]]:
        if self.base_graph is None or self.node_list is None:
            return self._signed_random_negative()
        center_index = self.python_rng.randint(0, self.num_nodes - 1)
        center_node = self.node_list[center_index]
        neighborhood = self.base_graph[center_node]
        union_set = (
            neighborhood["pos_out"]
            | neighborhood["pos_in"]
            | neighborhood["neg_out"]
            | neighborhood["neg_in"]
            | neighborhood["undirected"]
        )
        neighbor_ids = list(union_set)
        source_size = self.python_rng.choice(self.source_sizes)
        if len(neighbor_ids) >= source_size:
            sampled_ids = self.python_rng.sample(neighbor_ids, source_size)
        else:
            sampled_ids = neighbor_ids + self.python_rng.sample(
                self.node_list,
                source_size - len(neighbor_ids),
            )
        source = [self.node_to_idx[node_id] for node_id in sampled_ids]
        target_size = self.python_rng.choice(self.target_sizes)
        target = self._sample_node_indices(target_size)
        return self._deduplicate(source), self._deduplicate(target)

    def _signed_perturb_negative(self) -> tuple[list[int], list[int]]:
        reference = self.python_rng.choice(self.reference_edges)
        source = list(map(int, reference["s"]))
        target = list(map(int, reference["t"]))
        if source and self.python_rng.random() > 0.5:
            position = self.python_rng.randint(0, len(source) - 1)
            source[position] = self.python_rng.randint(0, self.num_nodes - 1)
        elif target:
            position = self.python_rng.randint(0, len(target) - 1)
            target[position] = self.python_rng.randint(0, self.num_nodes - 1)
        return self._deduplicate(source), self._deduplicate(target)

    def _reactome_random_negative(self) -> tuple[list[int], list[int]]:
        source_size = min(self.python_rng.choice(self.source_sizes), self.num_nodes)
        target_size = min(self.python_rng.choice(self.target_sizes), self.num_nodes)
        source = self._sample_node_indices(source_size)
        target = self._sample_node_indices(target_size)
        return source, target

    def _reactome_local_negative(self) -> tuple[list[int], list[int]]:
        if self.node_neighbors is None:
            return self._reactome_random_negative()
        source_size = self.python_rng.choice(self.source_sizes)
        target_size = self.python_rng.choice(self.target_sizes)
        center = self.python_rng.randint(0, self.num_nodes - 1)
        neighbors = list(self.node_neighbors.get(center, []))

        if len(neighbors) >= source_size:
            source = self.python_rng.sample(neighbors, source_size)
        else:
            source = list(neighbors)
            remaining_count = source_size - len(source)
            source_set = set(source)
            pool = [node for node in range(self.num_nodes) if node not in source_set]
            if len(pool) >= remaining_count:
                source += self.python_rng.sample(pool, remaining_count)
            else:
                source += self._sample_node_indices(remaining_count, replace=True)

        if len(neighbors) >= target_size and self.python_rng.random() < 0.5:
            target = self.python_rng.sample(neighbors, target_size)
        else:
            target = self._sample_node_indices(min(target_size, self.num_nodes))

        source = self._deduplicate(source)
        target = self._deduplicate(target)
        if not source:
            source = [self.python_rng.randint(0, self.num_nodes - 1)]
        if not target:
            target = [self.python_rng.randint(0, self.num_nodes - 1)]
        return source, target

    def _reactome_perturb_negative(self) -> tuple[list[int], list[int]]:
        reference = self.python_rng.choice(self.reference_edges)
        source = list(map(int, reference["s"]))
        target = list(map(int, reference["t"]))
        if not source or not target:
            return self._reactome_random_negative()
        if self.python_rng.random() < 0.5:
            position = self.python_rng.randint(0, len(source) - 1)
            source[position] = self.python_rng.randint(0, self.num_nodes - 1)
        else:
            position = self.python_rng.randint(0, len(target) - 1)
            target[position] = self.python_rng.randint(0, self.num_nodes - 1)
        source = self._deduplicate(source)
        target = self._deduplicate(target)
        if not source or not target:
            return self._reactome_random_negative()
        return source, target

    def _generate_negative(self) -> tuple[list[int], list[int]]:
        mode = self.mode
        if mode == "mixed":
            mode = self.python_rng.choice(["random", "local", "perturb"])
        if self.profile == "reactome":
            if mode == "random":
                return self._reactome_random_negative()
            if mode == "local":
                return self._reactome_local_negative()
            return self._reactome_perturb_negative()
        if mode == "random":
            return self._signed_random_negative()
        if mode == "local":
            return self._signed_local_negative()
        return self._signed_perturb_negative()

    def sample(
        self,
        real_edges: Sequence[Hyperedge],
        batch_size: int | None = None,
        replace_real: bool = False,
    ) -> tuple[list[list[int]], list[list[int]], torch.Tensor]:
        if not real_edges:
            raise ValueError("real_edges must not be empty.")
        if batch_size is None:
            batch_size = len(real_edges)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        if replace_real:
            indices = self.numpy_rng.choice(
                len(real_edges),
                size=batch_size,
                replace=True,
            )
        else:
            if batch_size > len(real_edges):
                raise ValueError(
                    "batch_size cannot exceed the number of real edges when "
                    "replace_real is False."
                )
            indices = np.arange(batch_size)

        sources: list[list[int]] = []
        targets: list[list[int]] = []
        labels: list[int] = []
        for index in indices:
            edge = real_edges[int(index)]
            sources.append(self._deduplicate(edge["s"]))
            targets.append(self._deduplicate(edge["t"]))
            labels.append(int(edge["type"]))

        negative_count = int(batch_size * self.negative_ratio)
        if self.profile == "reactome":
            negative_count = max(1, negative_count)

        generated = 0
        attempts = 0
        max_attempts = max(
            self.max_attempts_per_negative,
            negative_count * self.max_attempts_per_negative,
        )
        while generated < negative_count and attempts < max_attempts:
            attempts += 1
            source, target = self._generate_negative()
            if not source or not target:
                continue
            if self.edge_key(source, target) in self.real_edges_set:
                continue
            sources.append(source)
            targets.append(target)
            labels.append(0)
            generated += 1

        if generated < negative_count:
            raise RuntimeError(
                f"Generated {generated} of {negative_count} requested negative "
                "hyperedges before reaching the attempt limit."
            )
        return sources, targets, torch.tensor(labels, dtype=torch.long)
