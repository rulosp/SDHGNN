from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .fixed_dataset import FixedHypergraphDataset, edge_key


def _parse_nodes(value: Any) -> list[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split("|") if item.strip()]


def build_degree_features(
    edges: Sequence[dict[str, Any]],
    num_nodes: int,
) -> np.ndarray:
    """Build six-dimensional label-aware degree features.

    The feature channels are source degree, target degree, activation-source
    degree, activation-target degree, inhibition-source degree, and
    inhibition-target degree. Counts are transformed with log(1 + x).
    """
    features = np.zeros((num_nodes, 6), dtype=np.float32)
    for edge in edges:
        label = int(edge["type"])
        for node in edge["s"]:
            node_index = int(node)
            features[node_index, 0] += 1.0
            if label == 1:
                features[node_index, 2] += 1.0
            elif label == 2:
                features[node_index, 4] += 1.0
        for node in edge["t"]:
            node_index = int(node)
            features[node_index, 1] += 1.0
            if label == 1:
                features[node_index, 3] += 1.0
            elif label == 2:
                features[node_index, 5] += 1.0
    return np.log1p(features).astype(np.float32)


class ReactomeDatasetLoader:
    def __init__(
        self,
        hyperedge_path: str | Path,
        inhibition_pattern: str = "+-",
        balance_real_classes: bool = False,
        seed: int = 0,
    ) -> None:
        self.hyperedge_path = Path(hyperedge_path).expanduser().resolve()
        self.inhibition_pattern = inhibition_pattern
        self.balance_real_classes = balance_real_classes
        self.seed = seed
        if inhibition_pattern not in {"+-", "-+"}:
            raise ValueError("inhibition_pattern must be '+-' or '-+'.")

    def load(self) -> FixedHypergraphDataset:
        if not self.hyperedge_path.is_file():
            raise FileNotFoundError(
                f"Reactome hyperedge file not found: {self.hyperedge_path}"
            )
        frame = pd.read_csv(self.hyperedge_path)
        required = {"source_nodes", "target_nodes", "sign"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Reactome CSV is missing columns: {sorted(missing)}")

        raw_edges: list[dict[str, Any]] = []
        all_nodes: set[int] = set()
        for _, row in frame.iterrows():
            source = _parse_nodes(row["source_nodes"])
            target = _parse_nodes(row["target_nodes"])
            if not source or not target:
                continue
            try:
                sign = int(row["sign"])
            except (TypeError, ValueError):
                continue
            if sign not in {-1, 1}:
                continue
            all_nodes.update(source)
            all_nodes.update(target)
            raw_edges.append(
                {
                    "s": source,
                    "t": target,
                    "sign": sign,
                    "type": 1 if sign == 1 else 2,
                    "reaction_id": row.get("reaction_id", ""),
                    "control_id": row.get("control_id", ""),
                    "old_hyperedge_id": row.get(
                        "old_hyperedge_id",
                        row.get("hyperedge_id", ""),
                    ),
                }
            )
        if not raw_edges:
            raise ValueError("No valid Reactome hyperedges were loaded.")

        num_nodes = max(all_nodes) + 1
        selected_edges = self._balance(raw_edges)
        source_rows: list[int] = []
        source_columns: list[int] = []
        source_values: list[float] = []
        target_rows: list[int] = []
        target_columns: list[int] = []
        target_values: list[float] = []

        for edge_index, edge in enumerate(selected_edges):
            if int(edge["sign"]) == 1:
                source_sign, target_sign = 1.0, 1.0
            elif self.inhibition_pattern == "+-":
                source_sign, target_sign = 1.0, -1.0
            else:
                source_sign, target_sign = -1.0, 1.0
            for node in edge["s"]:
                source_rows.append(int(node))
                source_columns.append(edge_index)
                source_values.append(source_sign)
            for node in edge["t"]:
                target_rows.append(int(node))
                target_columns.append(edge_index)
                target_values.append(target_sign)

        shape = (num_nodes, len(selected_edges))
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

        metadata = {
            "dataset_name": "reactome",
            "inhibition_pattern": self.inhibition_pattern,
            "balance_real_classes": self.balance_real_classes,
            "selection_seed": self.seed,
            "raw_label_counts": dict(Counter(int(edge["type"]) for edge in raw_edges)),
            "selected_label_counts": dict(
                Counter(int(edge["type"]) for edge in selected_edges)
            ),
            "class_names": {0: "fake", 1: "activation", 2: "inhibition"},
            "node_feature_type": "degree",
            "node_feature_scope": "full_dataset",
            "sampling_profile": "reactome",
        }
        return FixedHypergraphDataset(
            source_incidence=source_incidence,
            target_incidence=target_incidence,
            node_features=build_degree_features(selected_edges, num_nodes),
            hyperedges=selected_edges,
            num_nodes=num_nodes,
            node_list=list(range(num_nodes)),
            metadata=metadata,
            base_graph=None,
            raw_edge_count=len(raw_edges),
            all_real_edge_keys={edge_key(edge["s"], edge["t"]) for edge in raw_edges},
        )

    def _balance(self, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.balance_real_classes:
            return list(edges)
        activation = [edge for edge in edges if int(edge["type"]) == 1]
        inhibition = [edge for edge in edges if int(edge["type"]) == 2]
        if not activation or not inhibition:
            raise ValueError("Both Reactome real classes are required for balancing.")
        sample_size = min(len(activation), len(inhibition))
        rng = np.random.default_rng(self.seed)
        activation_indices = rng.choice(
            len(activation),
            size=sample_size,
            replace=False,
        )
        selected = [activation[int(index)] for index in activation_indices]
        selected.extend(inhibition)
        rng.shuffle(selected)
        return selected
