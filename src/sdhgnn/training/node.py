from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from sdhgnn.data.common import NodeClassificationDataset, torch_sparse_to_scipy
from sdhgnn.models import LaplacianCalculator, SDHGNN
from sdhgnn.training.common import classification_metrics, scipy_to_torch_sparse, set_seed

EpochCallback = Callable[[dict[str, float | int]], None]


@dataclass(frozen=True)
class NodeTrainingConfig:
    q1: float = 0.25
    q2: float = 0.25
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 500
    dropout: float = 0.5
    patience: int = 100
    num_layers: int = 2
    use_hyperedge_conv: bool = True
    use_phase_matrix: bool = True
    use_imag_channel: bool = True
    selection_metric: str = "macro_f1"

    def validate(self) -> None:
        if self.hidden_dim < 1 or self.num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Optimizer parameters are invalid.")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.selection_metric not in {"macro_f1", "accuracy"}:
            raise ValueError("selection_metric must be 'macro_f1' or 'accuracy'.")


@dataclass
class NodeValidationResult:
    best_epoch: int
    stopped_epoch: int
    best_metrics: dict[str, float]
    history: list[dict[str, float | int]]


@dataclass
class NodeFitResult:
    metrics: dict[str, float]
    logits: torch.Tensor
    model_state: dict[str, torch.Tensor]
    history: list[dict[str, float | int]]


@dataclass
class NodeGraphTensors:
    node_laplacian: tuple[torch.Tensor, torch.Tensor]
    hyperedge_laplacian: tuple[torch.Tensor, torch.Tensor]
    node_scale: torch.Tensor
    hyperedge_scale: torch.Tensor
    incidence: torch.Tensor


def prepare_node_graph(
    dataset: NodeClassificationDataset,
    config: NodeTrainingConfig,
    device: torch.device,
) -> NodeGraphTensors:
    source = torch_sparse_to_scipy(dataset.source_incidence)
    target = torch_sparse_to_scipy(dataset.target_incidence)
    incidence = scipy_to_torch_sparse(source + target, device=device)
    calculator = LaplacianCalculator(source, target, device=device)
    node_laplacian, hyperedge_laplacian, node_scale, hyperedge_scale = (
        calculator.get_matrices(
            q1=config.q1,
            q2=config.q2,
            use_phase_matrix=config.use_phase_matrix,
        )
    )
    return NodeGraphTensors(
        node_laplacian=node_laplacian,
        hyperedge_laplacian=hyperedge_laplacian,
        node_scale=node_scale,
        hyperedge_scale=hyperedge_scale,
        incidence=incidence,
    )


def _build_model(
    dataset: NodeClassificationDataset,
    graph: NodeGraphTensors,
    config: NodeTrainingConfig,
    device: torch.device,
) -> SDHGNN:
    return SDHGNN(
        in_dim=dataset.feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=dataset.num_classes,
        incidence=graph.incidence,
        dropout=config.dropout,
        num_layers=config.num_layers,
        use_hyperedge_conv=config.use_hyperedge_conv,
        use_imag_channel=config.use_imag_channel,
    ).to(device)


def _evaluate(
    model: SDHGNN,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    graph: NodeGraphTensors,
    num_classes: int,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=features.device)
    with torch.no_grad():
        logits = model(
            features,
            graph.node_laplacian,
            graph.hyperedge_laplacian,
            graph.node_scale,
            graph.hyperedge_scale,
        )
        selected_logits = logits[index_tensor]
        selected_labels = labels[index_tensor]
        metrics = classification_metrics(
            selected_logits,
            selected_labels,
            num_classes,
        )
    return metrics, selected_logits.detach().cpu()


def _rank(metrics: dict[str, float], selection_metric: str) -> tuple[float, float]:
    secondary = "accuracy" if selection_metric == "macro_f1" else "macro_f1"
    primary_value = float(metrics[selection_metric])
    secondary_value = float(metrics[secondary])
    return (
        primary_value if math.isfinite(primary_value) else -math.inf,
        secondary_value if math.isfinite(secondary_value) else -math.inf,
    )


def fit_node_with_validation(
    dataset: NodeClassificationDataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: NodeTrainingConfig,
    device: torch.device,
    seed: int,
    epoch_callback: EpochCallback | None = None,
) -> NodeValidationResult:
    config.validate()
    set_seed(seed)
    features = dataset.features.to(device)
    labels = dataset.labels.to(device)
    graph = prepare_node_graph(dataset, config, device)
    model = _build_model(dataset, graph, config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=device)

    best_metrics = {
        "accuracy": -math.inf,
        "macro_f1": -math.inf,
        "macro_auc": float("nan"),
    }
    best_epoch = 1
    stopped_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            features,
            graph.node_laplacian,
            graph.hyperedge_laplacian,
            graph.node_scale,
            graph.hyperedge_scale,
        )
        loss = criterion(logits[train_tensor], labels[train_tensor])
        loss.backward()
        optimizer.step()

        metrics, _ = _evaluate(
            model,
            features,
            labels,
            validation_indices,
            graph,
            dataset.num_classes,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu()),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_macro_auc": metrics["macro_auc"],
        }
        history.append(record)
        if epoch_callback is not None:
            epoch_callback(record)

        if _rank(metrics, config.selection_metric) > _rank(
            best_metrics,
            config.selection_metric,
        ):
            best_metrics = metrics
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        stopped_epoch = epoch
        if epochs_without_improvement >= config.patience:
            break

    return NodeValidationResult(
        best_epoch=best_epoch,
        stopped_epoch=stopped_epoch,
        best_metrics=best_metrics,
        history=history,
    )


def fit_node_fixed_epochs(
    dataset: NodeClassificationDataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    config: NodeTrainingConfig,
    device: torch.device,
    seed: int,
    epochs: int,
    epoch_callback: EpochCallback | None = None,
) -> NodeFitResult:
    config.validate()
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    set_seed(seed)
    features = dataset.features.to(device)
    labels = dataset.labels.to(device)
    graph = prepare_node_graph(dataset, config, device)
    model = _build_model(dataset, graph, config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            features,
            graph.node_laplacian,
            graph.hyperedge_laplacian,
            graph.node_scale,
            graph.hyperedge_scale,
        )
        loss = criterion(logits[train_tensor], labels[train_tensor])
        loss.backward()
        optimizer.step()
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu()),
        }
        history.append(record)
        if epoch_callback is not None:
            epoch_callback(record)

    metrics, selected_logits = _evaluate(
        model,
        features,
        labels,
        test_indices,
        graph,
        dataset.num_classes,
    )
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return NodeFitResult(
        metrics=metrics,
        logits=selected_logits,
        model_state=state,
        history=history,
    )
