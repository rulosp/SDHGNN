from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from .common import NodeClassificationDataset, make_torch_sparse


class DblpLoader:
    REQUIRED_FILES = (
        "features.pkl",
        "labels.pkl",
        "edge_by_paper.pkl",
        "edge_by_term.pkl",
        "edge_by_conf.pkl",
    )

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()

    def _load_pickle(self, filename: str) -> Any:
        path = self.data_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"DBLP file not found: {path}")
        with path.open("rb") as handle:
            return pickle.load(handle)

    def load(self) -> NodeClassificationDataset:
        features_raw = self._load_pickle("features.pkl")
        labels_raw = self._load_pickle("labels.pkl")
        if sp.issparse(features_raw):
            features_raw = features_raw.toarray()
        features = torch.as_tensor(
            np.asarray(features_raw, dtype=np.float32),
            dtype=torch.float32,
        )
        labels = torch.as_tensor(
            np.asarray(labels_raw, dtype=np.int64),
            dtype=torch.long,
        ).view(-1)
        if features.ndim != 2:
            raise ValueError("DBLP features must be a two-dimensional array.")
        if labels.shape[0] != features.shape[0]:
            raise ValueError("DBLP feature and label counts do not match.")

        edge_groups = [
            self._load_pickle("edge_by_paper.pkl"),
            self._load_pickle("edge_by_term.pkl"),
            self._load_pickle("edge_by_conf.pkl"),
        ]
        hyperedges = [edge for group in edge_groups for edge in group]

        source_rows: list[int] = []
        source_columns: list[int] = []
        target_rows: list[int] = []
        target_columns: list[int] = []
        for edge_index, raw_nodes in enumerate(hyperedges):
            nodes = [int(node) for node in raw_nodes]
            if not nodes:
                continue
            invalid = [node for node in nodes if not 0 <= node < features.shape[0]]
            if invalid:
                raise ValueError(
                    f"DBLP hyperedge {edge_index} contains invalid node indices."
                )
            target_rows.append(nodes[0])
            target_columns.append(edge_index)
            sources = nodes[1:] if len(nodes) > 1 else nodes
            for source in sources:
                source_rows.append(source)
                source_columns.append(edge_index)

        shape = (features.shape[0], len(hyperedges))
        return NodeClassificationDataset(
            features=features,
            labels=labels,
            source_incidence=make_torch_sparse(
                source_rows,
                source_columns,
                shape,
            ),
            target_incidence=make_torch_sparse(
                target_rows,
                target_columns,
                shape,
            ),
            name="dblp",
        )
