from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class NodeClassificationDataset:
    features: torch.Tensor
    labels: torch.Tensor
    source_incidence: torch.Tensor
    target_incidence: torch.Tensor
    name: str

    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_classes(self) -> int:
        return int(torch.unique(self.labels).numel())

    @property
    def num_hyperedges(self) -> int:
        return int(self.source_incidence.shape[1])


def make_torch_sparse(
    rows: list[int],
    columns: list[int],
    shape: tuple[int, int],
    values: list[float] | None = None,
) -> torch.Tensor:
    if values is None:
        values = [1.0] * len(rows)
    if len(rows) != len(columns) or len(rows) != len(values):
        raise ValueError("rows, columns, and values must have the same length.")
    if not rows:
        indices = torch.empty((2, 0), dtype=torch.long)
        data = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, data, size=shape).coalesce()

    matrix = sp.coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=shape,
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix = matrix.tocoo()
    indices = torch.as_tensor(
        np.vstack((matrix.row, matrix.col)),
        dtype=torch.long,
    )
    data = torch.as_tensor(matrix.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, data, size=shape).coalesce()


def torch_sparse_to_scipy(tensor: torch.Tensor) -> sp.coo_matrix:
    if tensor.layout != torch.sparse_coo:
        raise TypeError("Expected a sparse COO tensor.")
    tensor = tensor.coalesce().detach().cpu()
    indices = tensor.indices().numpy()
    values = tensor.values().numpy().astype(np.float32, copy=False)
    return sp.coo_matrix(
        (values, (indices[0], indices[1])),
        shape=tuple(tensor.shape),
        dtype=np.float32,
    )
