from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

SparsePair = tuple[torch.Tensor, torch.Tensor]
IndexBatch = Sequence[Sequence[int]]


class LaplacianCalculator:
    """Construct sparse node- and hyperedge-domain magnetic Laplacians."""

    def __init__(
        self,
        source_incidence: sp.spmatrix,
        target_incidence: sp.spmatrix,
        device: torch.device | str = "cpu",
    ) -> None:
        if source_incidence.shape != target_incidence.shape:
            raise ValueError("Source and target incidence matrices must have the same shape.")

        self.device = torch.device(device)
        self.source_incidence = source_incidence.astype(np.float32)
        self.target_incidence = target_incidence.astype(np.float32)
        self.incidence = self.source_incidence + self.target_incidence

        self.num_nodes, self.num_hyperedges = self.source_incidence.shape
        self.hyperedge_degree = np.asarray(
            np.abs(self.incidence).sum(axis=0)
        ).ravel()
        self.node_degree = np.asarray(
            np.abs(self.incidence).sum(axis=1)
        ).ravel()

    @staticmethod
    def _inverse_diagonal(values: np.ndarray, power: float) -> sp.spmatrix:
        with np.errstate(divide="ignore", invalid="ignore"):
            transformed = np.power(values, power)
        transformed[~np.isfinite(transformed)] = 0.0
        return sp.diags(transformed)

    def _to_torch_sparse(self, matrix: sp.spmatrix) -> torch.Tensor:
        matrix = matrix.tocoo()
        indices = torch.as_tensor(
            np.vstack((matrix.row, matrix.col)),
            dtype=torch.long,
            device=self.device,
        )
        values = torch.as_tensor(
            matrix.data,
            dtype=torch.float32,
            device=self.device,
        )
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=matrix.shape,
            device=self.device,
        ).coalesce()

    def _to_sparse_pair(
        self,
        real: sp.spmatrix,
        imag: sp.spmatrix,
    ) -> SparsePair:
        return self._to_torch_sparse(real), self._to_torch_sparse(imag)

    def get_matrices(
        self,
        q1: float = 0.25,
        q2: float = 0.25,
        use_phase_matrix: bool = True,
    ) -> tuple[SparsePair, SparsePair, torch.Tensor, torch.Tensor]:
        node_degree_inv_sqrt = self._inverse_diagonal(self.node_degree, -0.5)
        hyperedge_degree_inv = self._inverse_diagonal(self.hyperedge_degree, -1.0)
        hyperedge_degree_inv_sqrt = self._inverse_diagonal(
            self.hyperedge_degree, -0.5
        )
        node_degree_inv = self._inverse_diagonal(self.node_degree, -1.0)

        node_affinity = (
            node_degree_inv_sqrt
            @ self.incidence
            @ hyperedge_degree_inv
            @ self.incidence.T
            @ node_degree_inv_sqrt
        ).tocoo()

        if use_phase_matrix:
            forward_flow = (
                node_degree_inv_sqrt
                @ self.source_incidence
                @ hyperedge_degree_inv
                @ self.target_incidence.T
                @ node_degree_inv_sqrt
            )
            reverse_flow = (
                node_degree_inv_sqrt
                @ self.target_incidence
                @ hyperedge_degree_inv
                @ self.source_incidence.T
                @ node_degree_inv_sqrt
            )
            node_phase = (2.0 * np.pi * q1 * (forward_flow - reverse_flow)).tocsr()
            node_angles = np.asarray(
                node_phase[node_affinity.row, node_affinity.col]
            ).ravel()
        else:
            node_angles = np.zeros_like(node_affinity.data, dtype=np.float32)

        node_real = sp.coo_matrix(
            (
                node_affinity.data * np.cos(node_angles),
                (node_affinity.row, node_affinity.col),
            ),
            shape=node_affinity.shape,
        )
        node_imag = sp.coo_matrix(
            (
                node_affinity.data * np.sin(node_angles),
                (node_affinity.row, node_affinity.col),
            ),
            shape=node_affinity.shape,
        )
        node_laplacian = self._to_sparse_pair(
            sp.eye(self.num_nodes, format="coo") - node_real,
            -node_imag,
        )

        hyperedge_affinity = (
            hyperedge_degree_inv_sqrt
            @ self.incidence.T
            @ node_degree_inv
            @ self.incidence
            @ hyperedge_degree_inv_sqrt
        ).tocoo()

        if use_phase_matrix:
            forward_flow = (
                hyperedge_degree_inv_sqrt
                @ self.source_incidence.T
                @ node_degree_inv
                @ self.target_incidence
                @ hyperedge_degree_inv_sqrt
            )
            reverse_flow = (
                hyperedge_degree_inv_sqrt
                @ self.target_incidence.T
                @ node_degree_inv
                @ self.source_incidence
                @ hyperedge_degree_inv_sqrt
            )
            hyperedge_phase = (
                2.0 * np.pi * q2 * (forward_flow - reverse_flow)
            ).tocsr()
            hyperedge_angles = np.asarray(
                hyperedge_phase[hyperedge_affinity.row, hyperedge_affinity.col]
            ).ravel()
        else:
            hyperedge_angles = np.zeros_like(
                hyperedge_affinity.data, dtype=np.float32
            )

        hyperedge_real = sp.coo_matrix(
            (
                hyperedge_affinity.data * np.cos(hyperedge_angles),
                (hyperedge_affinity.row, hyperedge_affinity.col),
            ),
            shape=hyperedge_affinity.shape,
        )
        hyperedge_imag = sp.coo_matrix(
            (
                hyperedge_affinity.data * np.sin(hyperedge_angles),
                (hyperedge_affinity.row, hyperedge_affinity.col),
            ),
            shape=hyperedge_affinity.shape,
        )
        hyperedge_laplacian = self._to_sparse_pair(
            sp.eye(self.num_hyperedges, format="coo") - hyperedge_real,
            -hyperedge_imag,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            node_scale = np.power(self.node_degree, -0.5)
            hyperedge_scale = np.power(self.hyperedge_degree, -0.5)
        node_scale[~np.isfinite(node_scale)] = 0.0
        hyperedge_scale[~np.isfinite(hyperedge_scale)] = 0.0

        node_scale_tensor = torch.as_tensor(
            node_scale,
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        hyperedge_scale_tensor = torch.as_tensor(
            hyperedge_scale,
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

        return (
            node_laplacian,
            hyperedge_laplacian,
            node_scale_tensor,
            hyperedge_scale_tensor,
        )


class ComplexHypergraphConv(nn.Module):
    """Sparse complex hypergraph convolution in node and hyperedge domains."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        incidence: torch.Tensor,
        use_hyperedge_conv: bool = True,
        use_imag_channel: bool = True,
    ) -> None:
        super().__init__()
        if not incidence.is_sparse:
            raise TypeError("incidence must be a sparse COO tensor.")

        self.use_hyperedge_conv = use_hyperedge_conv
        self.use_imag_channel = use_imag_channel

        incidence = incidence.coalesce()
        self.register_buffer("incidence", incidence, persistent=False)
        self.register_buffer(
            "incidence_transpose",
            incidence.transpose(0, 1).coalesce(),
            persistent=False,
        )

        self.theta_0 = nn.Linear(in_dim, out_dim, bias=True)
        self.theta_1 = nn.Linear(in_dim, out_dim, bias=False)

        if use_hyperedge_conv:
            self.theta_prime_0 = nn.Linear(out_dim, out_dim, bias=True)
            self.theta_prime_1 = nn.Linear(out_dim, out_dim, bias=False)
        else:
            self.theta_prime_0 = None
            self.theta_prime_1 = None

    @staticmethod
    def _complex_sparse_mm(
        matrix_real: torch.Tensor,
        matrix_imag: torch.Tensor,
        features_real: torch.Tensor,
        features_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_real = (
            torch.sparse.mm(matrix_real, features_real)
            - torch.sparse.mm(matrix_imag, features_imag)
        )
        output_imag = (
            torch.sparse.mm(matrix_real, features_imag)
            + torch.sparse.mm(matrix_imag, features_real)
        )
        return output_real, output_imag

    @staticmethod
    def _complex_relu(
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return F.relu(real), F.relu(imag)

    def _forward_real(
        self,
        features_real: torch.Tensor,
        node_laplacian: SparsePair,
        hyperedge_laplacian: SparsePair,
        node_scale: torch.Tensor,
        hyperedge_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_laplacian_real, _ = node_laplacian
        hyperedge_laplacian_real, _ = hyperedge_laplacian

        transformed = self.theta_0(features_real)
        propagated = self.theta_1(
            torch.sparse.mm(node_laplacian_real, features_real)
        )
        node_features = F.relu(transformed + propagated)

        if not self.use_hyperedge_conv:
            return node_features, torch.zeros_like(node_features)

        hyperedge_features = torch.sparse.mm(
            self.incidence_transpose,
            node_features * node_scale,
        )
        hyperedge_features = hyperedge_features * hyperedge_scale

        transformed_hyperedges = self.theta_prime_0(hyperedge_features)
        propagated_hyperedges = self.theta_prime_1(
            torch.sparse.mm(hyperedge_laplacian_real, hyperedge_features)
        )
        hyperedge_features = F.relu(
            transformed_hyperedges + propagated_hyperedges
        )

        node_output = torch.sparse.mm(
            self.incidence,
            hyperedge_features * hyperedge_scale,
        )
        node_output = node_output * node_scale
        return node_output, torch.zeros_like(node_output)

    def forward(
        self,
        features_real: torch.Tensor,
        features_imag: torch.Tensor | None,
        node_laplacian: SparsePair,
        hyperedge_laplacian: SparsePair,
        node_scale: torch.Tensor,
        hyperedge_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_imag_channel:
            return self._forward_real(
                features_real,
                node_laplacian,
                hyperedge_laplacian,
                node_scale,
                hyperedge_scale,
            )

        if features_imag is None:
            features_imag = torch.zeros_like(features_real)

        node_laplacian_real, node_laplacian_imag = node_laplacian
        hyperedge_laplacian_real, hyperedge_laplacian_imag = hyperedge_laplacian

        transformed_real = self.theta_0(features_real)
        transformed_imag = self.theta_0(features_imag)
        propagated_real, propagated_imag = self._complex_sparse_mm(
            node_laplacian_real,
            node_laplacian_imag,
            features_real,
            features_imag,
        )
        propagated_real = self.theta_1(propagated_real)
        propagated_imag = self.theta_1(propagated_imag)
        node_real, node_imag = self._complex_relu(
            transformed_real + propagated_real,
            transformed_imag + propagated_imag,
        )

        if not self.use_hyperedge_conv:
            return node_real, node_imag

        hyperedge_real = torch.sparse.mm(
            self.incidence_transpose,
            node_real * node_scale,
        )
        hyperedge_imag = torch.sparse.mm(
            self.incidence_transpose,
            node_imag * node_scale,
        )
        hyperedge_real = hyperedge_real * hyperedge_scale
        hyperedge_imag = hyperedge_imag * hyperedge_scale

        transformed_hyperedge_real = self.theta_prime_0(hyperedge_real)
        transformed_hyperedge_imag = self.theta_prime_0(hyperedge_imag)
        propagated_hyperedge_real, propagated_hyperedge_imag = (
            self._complex_sparse_mm(
                hyperedge_laplacian_real,
                hyperedge_laplacian_imag,
                hyperedge_real,
                hyperedge_imag,
            )
        )
        propagated_hyperedge_real = self.theta_prime_1(
            propagated_hyperedge_real
        )
        propagated_hyperedge_imag = self.theta_prime_1(
            propagated_hyperedge_imag
        )
        hyperedge_real, hyperedge_imag = self._complex_relu(
            transformed_hyperedge_real + propagated_hyperedge_real,
            transformed_hyperedge_imag + propagated_hyperedge_imag,
        )

        output_real = torch.sparse.mm(
            self.incidence,
            hyperedge_real * hyperedge_scale,
        )
        output_imag = torch.sparse.mm(
            self.incidence,
            hyperedge_imag * hyperedge_scale,
        )
        return output_real * node_scale, output_imag * node_scale


class SDHGNN(nn.Module):
    """Node classification model."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        incidence: torch.Tensor,
        dropout: float = 0.5,
        num_layers: int = 2,
        use_hyperedge_conv: bool = True,
        use_imag_channel: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.dropout = dropout
        self.use_imag_channel = use_imag_channel

        layers: list[ComplexHypergraphConv] = []
        layers.append(
            ComplexHypergraphConv(
                in_dim,
                hidden_dim,
                incidence,
                use_hyperedge_conv=use_hyperedge_conv,
                use_imag_channel=use_imag_channel,
            )
        )
        for _ in range(1, num_layers):
            layers.append(
                ComplexHypergraphConv(
                    hidden_dim,
                    hidden_dim,
                    incidence,
                    use_hyperedge_conv=use_hyperedge_conv,
                    use_imag_channel=use_imag_channel,
                )
            )
        self.convs = nn.ModuleList(layers)

        classifier_dim = hidden_dim * 2 if use_imag_channel else hidden_dim
        self.classifier = nn.Linear(classifier_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        node_laplacian: SparsePair,
        hyperedge_laplacian: SparsePair,
        node_scale: torch.Tensor,
        hyperedge_scale: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_imag_channel:
            if torch.is_complex(features):
                real, imag = features.real, features.imag
            else:
                real, imag = features, torch.zeros_like(features)
        else:
            real = features.real if torch.is_complex(features) else features
            imag = None

        for layer_index, layer in enumerate(self.convs):
            real, imag = layer(
                real,
                imag,
                node_laplacian,
                hyperedge_laplacian,
                node_scale,
                hyperedge_scale,
            )
            if layer_index < len(self.convs) - 1:
                real = F.dropout(real, self.dropout, training=self.training)
                if self.use_imag_channel:
                    imag = F.dropout(imag, self.dropout, training=self.training)

        representation = torch.cat((real, imag), dim=1) if self.use_imag_channel else real
        return self.classifier(representation)


class SDHGNNEncoder(nn.Module):
    """Node encoder for directed signed hyperedge classification."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        incidence: torch.Tensor,
        num_nodes: int,
        dropout: float = 0.5,
        num_layers: int = 2,
        use_hyperedge_conv: bool = True,
        use_imag_channel: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.dropout = dropout
        self.use_imag_channel = use_imag_channel

        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        self.feat_proj = nn.Linear(in_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.convs = nn.ModuleList(
            [
                ComplexHypergraphConv(
                    hidden_dim,
                    hidden_dim,
                    incidence,
                    use_hyperedge_conv=use_hyperedge_conv,
                    use_imag_channel=use_imag_channel,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        features: torch.Tensor,
        node_laplacian: SparsePair,
        hyperedge_laplacian: SparsePair,
        node_scale: torch.Tensor,
        hyperedge_scale: torch.Tensor,
    ) -> torch.Tensor:
        node_ids = torch.arange(features.size(0), device=features.device)
        initial = F.relu(
            self.fusion(
                torch.cat(
                    (
                        self.node_emb(node_ids),
                        self.feat_proj(features),
                    ),
                    dim=1,
                )
            )
        )

        real = initial
        imag = torch.zeros_like(initial) if self.use_imag_channel else None

        for layer_index, layer in enumerate(self.convs):
            real, imag = layer(
                real,
                imag,
                node_laplacian,
                hyperedge_laplacian,
                node_scale,
                hyperedge_scale,
            )
            if layer_index < len(self.convs) - 1:
                real = F.dropout(real, self.dropout, training=self.training)
                if self.use_imag_channel:
                    imag = F.dropout(imag, self.dropout, training=self.training)

        real = real + initial
        if self.use_imag_channel:
            imag = imag + torch.zeros_like(initial)
        else:
            imag = torch.zeros_like(real)
        return torch.complex(real, imag)


class FineGrainedDecoder(nn.Module):
    """Decode source and target node sets into a hyperedge class."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 5,
        use_imag_channel: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_imag_channel = use_imag_channel
        input_dim = hidden_dim * (14 if use_imag_channel else 7)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(reference.size(1))

    def _aggregate(
        self,
        representation: torch.Tensor,
        indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not indices:
            zero = self._zero(representation)
            return zero, zero.clone(), zero.clone()
        selected = representation[indices]
        return selected.mean(dim=0), selected.max(dim=0).values, selected.min(dim=0).values

    def _forward_real(
        self,
        representation: torch.Tensor,
        source_indices: IndexBatch,
        target_indices: IndexBatch,
    ) -> torch.Tensor:
        features = []
        for source, target in zip(source_indices, target_indices):
            source_mean, source_max, source_min = self._aggregate(
                representation, source
            )
            target_mean, target_max, target_min = self._aggregate(
                representation, target
            )
            interaction = source_mean * target_mean
            features.append(
                torch.cat(
                    (
                        source_mean,
                        source_max,
                        source_min,
                        target_mean,
                        target_max,
                        target_min,
                        interaction,
                    )
                )
            )
        if not features:
            raise ValueError("At least one hyperedge is required.")
        return self.mlp(torch.stack(features))

    def _forward_complex(
        self,
        representation: torch.Tensor,
        source_indices: IndexBatch,
        target_indices: IndexBatch,
    ) -> torch.Tensor:
        real = representation.real
        imag = representation.imag
        features = []

        for source, target in zip(source_indices, target_indices):
            source_real_mean, source_real_max, source_real_min = self._aggregate(
                real, source
            )
            source_imag_mean, source_imag_max, source_imag_min = self._aggregate(
                imag, source
            )
            target_real_mean, target_real_max, target_real_min = self._aggregate(
                real, target
            )
            target_imag_mean, target_imag_max, target_imag_min = self._aggregate(
                imag, target
            )

            interaction_real = (
                source_real_mean * target_real_mean
                + source_imag_mean * target_imag_mean
            )
            interaction_imag = (
                source_imag_mean * target_real_mean
                - source_real_mean * target_imag_mean
            )
            features.append(
                torch.cat(
                    (
                        source_real_mean,
                        source_real_max,
                        source_real_min,
                        target_real_mean,
                        target_real_max,
                        target_real_min,
                        source_imag_mean,
                        source_imag_max,
                        source_imag_min,
                        target_imag_mean,
                        target_imag_max,
                        target_imag_min,
                        interaction_real,
                        interaction_imag,
                    )
                )
            )

        if not features:
            raise ValueError("At least one hyperedge is required.")
        return self.mlp(torch.stack(features))

    def forward(
        self,
        representation: torch.Tensor,
        source_indices: IndexBatch,
        target_indices: IndexBatch,
    ) -> torch.Tensor:
        if self.use_imag_channel:
            if not torch.is_complex(representation):
                representation = torch.complex(
                    representation,
                    torch.zeros_like(representation),
                )
            return self._forward_complex(
                representation,
                source_indices,
                target_indices,
            )

        real = representation.real if torch.is_complex(representation) else representation
        return self._forward_real(real, source_indices, target_indices)
