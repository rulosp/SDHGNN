from __future__ import annotations

import gc
import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from sdhgnn.data.fixed_dataset import FixedHypergraphDataset
from sdhgnn.sampling import HyperedgeSampler
from sdhgnn.models import FineGrainedDecoder, LaplacianCalculator, SDHGNNEncoder
from sdhgnn.training.common import (
    classification_metrics,
    make_class_weights,
    scipy_to_torch_sparse,
    set_seed,
)

EvaluationBatch = tuple[list[list[int]], list[list[int]], torch.Tensor]
MetricDict = dict[str, float]
EpochCallback = Callable[[dict[str, float | int]], None]
EdgeKey = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class HyperedgeTrainingConfig:
    q1: float
    q2: float
    hidden_dim: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 1000
    dropout: float = 0.5
    batch_size: int = 6000
    patience: int = 50
    negative_sampling_mode: str = "mixed"
    negative_ratio: float = 0.2
    num_layers: int = 2
    use_hyperedge_conv: bool = True
    use_phase_matrix: bool = True
    use_imag_channel: bool = True
    use_class_weights: bool = True
    class_weight_negative_ratio: float | None = None
    selection_metric: str = "macro_f1"

    def validate(self) -> None:
        if self.hidden_dim < 1 or self.num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive.")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.negative_sampling_mode not in HyperedgeSampler.MODES:
            raise ValueError(
                f"Unsupported negative sampling mode: {self.negative_sampling_mode}"
            )
        if self.negative_ratio < 0.0:
            raise ValueError("negative_ratio must be non-negative.")
        if (
            self.class_weight_negative_ratio is not None
            and self.class_weight_negative_ratio < 0.0
        ):
            raise ValueError("class_weight_negative_ratio must be non-negative.")
        if self.selection_metric not in {"macro_f1", "accuracy"}:
            raise ValueError("selection_metric must be 'macro_f1' or 'accuracy'.")


@dataclass
class GraphTensors:
    node_laplacian: tuple[torch.Tensor, torch.Tensor]
    hyperedge_laplacian: tuple[torch.Tensor, torch.Tensor]
    node_scale: torch.Tensor
    hyperedge_scale: torch.Tensor
    incidence: torch.Tensor


@dataclass
class ValidationFitResult:
    best_epoch: int
    stopped_epoch: int
    best_metrics: MetricDict
    history: list[dict[str, float | int]]


@dataclass
class EvaluationResult:
    metrics: MetricDict
    logits: torch.Tensor
    labels: torch.Tensor


@dataclass
class FixedEpochFitResult:
    epochs: int
    evaluation: EvaluationResult | None
    encoder_state: dict[str, torch.Tensor] | None
    decoder_state: dict[str, torch.Tensor] | None
    history: list[dict[str, float | int]]


def rank_metrics(
    metrics: Mapping[str, float],
    selection_metric: str,
) -> tuple[float, float]:
    if selection_metric not in {"macro_f1", "accuracy"}:
        raise ValueError("selection_metric must be 'macro_f1' or 'accuracy'.")
    secondary_metric = "accuracy" if selection_metric == "macro_f1" else "macro_f1"
    primary = float(metrics[selection_metric])
    secondary = float(metrics[secondary_metric])
    return (
        primary if math.isfinite(primary) else -math.inf,
        secondary if math.isfinite(secondary) else -math.inf,
    )


def build_graph_tensors(
    dataset: FixedHypergraphDataset,
    edge_indices: np.ndarray,
    config: HyperedgeTrainingConfig,
    device: torch.device,
) -> GraphTensors:
    if edge_indices.ndim != 1 or edge_indices.size == 0:
        raise ValueError("edge_indices must be a non-empty one-dimensional array.")

    source = dataset.source_incidence.tocsc()[:, edge_indices]
    target = dataset.target_incidence.tocsc()[:, edge_indices]
    incidence = scipy_to_torch_sparse(source + target, device=device)

    calculator = LaplacianCalculator(source, target, device=device)
    node_laplacian, hyperedge_laplacian, node_scale, hyperedge_scale = (
        calculator.get_matrices(
            q1=config.q1,
            q2=config.q2,
            use_phase_matrix=config.use_phase_matrix,
        )
    )
    return GraphTensors(
        node_laplacian=node_laplacian,
        hyperedge_laplacian=hyperedge_laplacian,
        node_scale=node_scale,
        hyperedge_scale=hyperedge_scale,
        incidence=incidence,
    )


def build_models(
    dataset: FixedHypergraphDataset,
    feature_dim: int,
    graph: GraphTensors,
    config: HyperedgeTrainingConfig,
    device: torch.device,
) -> tuple[SDHGNNEncoder, FineGrainedDecoder]:
    encoder = SDHGNNEncoder(
        in_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        incidence=graph.incidence,
        num_nodes=dataset.num_nodes,
        dropout=config.dropout,
        num_layers=config.num_layers,
        use_hyperedge_conv=config.use_hyperedge_conv,
        use_imag_channel=config.use_imag_channel,
    ).to(device)
    decoder = FineGrainedDecoder(
        hidden_dim=config.hidden_dim,
        num_classes=dataset.num_classes,
        use_imag_channel=config.use_imag_channel,
        dropout=config.dropout,
    ).to(device)
    return encoder, decoder


def _sampling_profile(dataset: FixedHypergraphDataset) -> str:
    configured = str(dataset.metadata.get("sampling_profile", "")).lower()
    if configured in HyperedgeSampler.PROFILES:
        return configured
    dataset_name = str(dataset.metadata.get("dataset_name", "")).lower()
    return "reactome" if "reactome" in dataset_name else "signed_network"


def make_evaluation_batch(
    dataset: FixedHypergraphDataset,
    reference_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    all_real_edges: set[EdgeKey],
    config: HyperedgeTrainingConfig,
    seed: int,
    python_rng: random.Random | None = None,
    numpy_rng: np.random.RandomState | None = None,
) -> EvaluationBatch:
    reference_edges = [
        dataset.hyperedges[int(index)] for index in reference_indices
    ]
    evaluation_edges = [
        dataset.hyperedges[int(index)] for index in evaluation_indices
    ]
    sampler = HyperedgeSampler(
        reference_edges=reference_edges,
        num_nodes=dataset.num_nodes,
        real_edges_set=all_real_edges,
        mode=config.negative_sampling_mode,
        negative_ratio=config.negative_ratio,
        seed=seed,
        profile=_sampling_profile(dataset),
        base_graph=dataset.base_graph,
        node_list=dataset.node_list,
        python_rng=python_rng,
        numpy_rng=numpy_rng,
    )
    return sampler.sample(
        evaluation_edges,
        batch_size=len(evaluation_edges),
        replace_real=False,
    )


def evaluate_models(
    encoder: SDHGNNEncoder,
    decoder: FineGrainedDecoder,
    features: torch.Tensor,
    graph: GraphTensors,
    batch: EvaluationBatch,
    num_classes: int,
    device: torch.device,
) -> EvaluationResult:
    source, target, labels = batch
    labels = labels.to(device)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        representation = encoder(
            features,
            graph.node_laplacian,
            graph.hyperedge_laplacian,
            graph.node_scale,
            graph.hyperedge_scale,
        )
        logits = decoder(representation, source, target)
        metrics = classification_metrics(logits, labels, num_classes)

    return EvaluationResult(
        metrics=metrics,
        logits=logits.detach().cpu(),
        labels=labels.detach().cpu(),
    )


def _build_loss(
    train_edges: Sequence[Mapping[str, Any]],
    num_classes: int,
    config: HyperedgeTrainingConfig,
    device: torch.device,
) -> nn.Module:
    if not config.use_class_weights:
        return nn.CrossEntropyLoss()
    weights = make_class_weights(
        (int(edge["type"]) for edge in train_edges),
        num_classes=num_classes,
        negative_ratio=(
            config.negative_ratio
            if config.class_weight_negative_ratio is None
            else config.class_weight_negative_ratio
        ),
        device=device,
    )
    return nn.CrossEntropyLoss(weight=weights)


def _batch_chunks(permutation: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [
        permutation[start : start + batch_size]
        for start in range(0, len(permutation), batch_size)
    ]


def _train_epoch(
    encoder: SDHGNNEncoder,
    decoder: FineGrainedDecoder,
    features: torch.Tensor,
    graph: GraphTensors,
    train_edges: Sequence[Mapping[str, Any]],
    sampler: HyperedgeSampler,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    rng: np.random.RandomState,
    device: torch.device,
) -> float:
    if len(train_edges) < 2:
        raise ValueError("At least two training hyperedges are required.")

    encoder.train()
    decoder.train()
    permutation = np.arange(len(train_edges))
    rng.shuffle(permutation)
    losses: list[float] = []

    for batch_indices in _batch_chunks(permutation, batch_size):
        batch_edges = [train_edges[int(index)] for index in batch_indices]
        source, target, labels = sampler.sample(
            batch_edges,
            batch_size=len(batch_edges),
            replace_real=True,
        )
        labels = labels.to(device)
        if labels.numel() < 2:
            raise ValueError(
                "A training batch must contain at least two samples because the "
                "decoder uses batch normalization."
            )

        optimizer.zero_grad(set_to_none=True)
        representation = encoder(
            features,
            graph.node_laplacian,
            graph.hyperedge_laplacian,
            graph.node_scale,
            graph.hyperedge_scale,
        )
        logits = decoder(representation, source, target)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


def _training_components(
    dataset: FixedHypergraphDataset,
    features: torch.Tensor,
    train_indices: np.ndarray,
    all_real_edges: set[EdgeKey],
    config: HyperedgeTrainingConfig,
    device: torch.device,
    seed: int,
) -> tuple[
    Sequence[Mapping[str, Any]],
    GraphTensors,
    SDHGNNEncoder,
    FineGrainedDecoder,
    torch.optim.Optimizer,
    nn.Module,
    HyperedgeSampler,
    np.random.RandomState,
]:
    train_edges = [dataset.hyperedges[int(index)] for index in train_indices]
    graph = build_graph_tensors(dataset, train_indices, config, device)
    encoder, decoder = build_models(
        dataset=dataset,
        feature_dim=features.shape[1],
        graph=graph,
        config=config,
        device=device,
    )
    optimizer = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = _build_loss(
        train_edges=train_edges,
        num_classes=dataset.num_classes,
        config=config,
        device=device,
    )
    python_rng = random.Random(seed)
    rng = np.random.RandomState(seed)
    sampler = HyperedgeSampler(
        reference_edges=train_edges,
        num_nodes=dataset.num_nodes,
        real_edges_set=all_real_edges,
        mode=config.negative_sampling_mode,
        negative_ratio=config.negative_ratio,
        seed=seed,
        profile=_sampling_profile(dataset),
        base_graph=dataset.base_graph,
        node_list=dataset.node_list,
        python_rng=python_rng,
        numpy_rng=rng,
    )
    return (
        train_edges,
        graph,
        encoder,
        decoder,
        optimizer,
        criterion,
        sampler,
        rng,
    )


def fit_with_validation(
    dataset: FixedHypergraphDataset,
    features: torch.Tensor,
    train_indices: np.ndarray,
    validation_batch: EvaluationBatch,
    all_real_edges: set[EdgeKey],
    config: HyperedgeTrainingConfig,
    device: torch.device,
    seed: int,
    epoch_callback: EpochCallback | None = None,
) -> ValidationFitResult:
    config.validate()
    set_seed(seed)
    (
        train_edges,
        graph,
        encoder,
        decoder,
        optimizer,
        criterion,
        sampler,
        rng,
    ) = _training_components(
        dataset,
        features,
        train_indices,
        all_real_edges,
        config,
        device,
        seed,
    )

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
        train_loss = _train_epoch(
            encoder=encoder,
            decoder=decoder,
            features=features,
            graph=graph,
            train_edges=train_edges,
            sampler=sampler,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=config.batch_size,
            rng=rng,
            device=device,
        )
        validation = evaluate_models(
            encoder=encoder,
            decoder=decoder,
            features=features,
            graph=graph,
            batch=validation_batch,
            num_classes=dataset.num_classes,
            device=device,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_accuracy": validation.metrics["accuracy"],
            "validation_macro_f1": validation.metrics["macro_f1"],
            "validation_macro_auc": validation.metrics["macro_auc"],
        }
        history.append(record)
        if epoch_callback is not None:
            epoch_callback(record)

        if rank_metrics(
            validation.metrics, config.selection_metric
        ) > rank_metrics(best_metrics, config.selection_metric):
            best_metrics = validation.metrics
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        stopped_epoch = epoch
        if epochs_without_improvement >= config.patience:
            break

    del encoder, decoder, optimizer, criterion, sampler, graph
    _cleanup(device)
    return ValidationFitResult(
        best_epoch=best_epoch,
        stopped_epoch=stopped_epoch,
        best_metrics=best_metrics,
        history=history,
    )


def fit_fixed_epochs(
    dataset: FixedHypergraphDataset,
    features: torch.Tensor,
    train_indices: np.ndarray,
    evaluation_batch: EvaluationBatch | None,
    all_real_edges: set[EdgeKey],
    config: HyperedgeTrainingConfig,
    device: torch.device,
    seed: int,
    epochs: int,
    keep_state: bool = True,
    epoch_callback: EpochCallback | None = None,
) -> FixedEpochFitResult:
    config.validate()
    if epochs < 1:
        raise ValueError("epochs must be positive.")

    set_seed(seed)
    (
        train_edges,
        graph,
        encoder,
        decoder,
        optimizer,
        criterion,
        sampler,
        rng,
    ) = _training_components(
        dataset,
        features,
        train_indices,
        all_real_edges,
        config,
        device,
        seed,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(
            encoder=encoder,
            decoder=decoder,
            features=features,
            graph=graph,
            train_edges=train_edges,
            sampler=sampler,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=config.batch_size,
            rng=rng,
            device=device,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
        }
        history.append(record)
        if epoch_callback is not None:
            epoch_callback(record)

    evaluation = None
    if evaluation_batch is not None:
        evaluation = evaluate_models(
            encoder=encoder,
            decoder=decoder,
            features=features,
            graph=graph,
            batch=evaluation_batch,
            num_classes=dataset.num_classes,
            device=device,
        )

    encoder_state = _state_dict_to_cpu(encoder) if keep_state else None
    decoder_state = _state_dict_to_cpu(decoder) if keep_state else None
    del encoder, decoder, optimizer, criterion, sampler, graph
    _cleanup(device)
    return FixedEpochFitResult(
        epochs=epochs,
        evaluation=evaluation,
        encoder_state=encoder_state,
        decoder_state=decoder_state,
        history=history,
    )


def _state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }



def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
