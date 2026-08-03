from __future__ import annotations

import argparse
import itertools
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from sdhgnn.data import load_fixed_hypergraph_dataset
from sdhgnn.sampling import HyperedgeSampler
from sdhgnn.training import (
    HyperedgeTrainingConfig,
    atomic_write_csv,
    atomic_write_json,
    finite_mean,
    finite_std,
    fit_fixed_epochs,
    fit_with_validation,
    make_evaluation_batch,
    rank_metrics,
    resolve_device,
    set_seed,
)


@dataclass(frozen=True)
class SearchConfig:
    dataset_path: Path
    output_dir: Path
    q1_values: tuple[float, ...]
    q2_values: tuple[float, ...]
    hidden_dim: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 1000
    dropout: float = 0.5
    batch_size: int = 6000
    patience: int = 50
    outer_folds: int = 5
    validation_fraction: float = 0.25
    seed: int = 0
    negative_sampling_mode: str = "mixed"
    negative_ratio: float = 0.2
    num_layers: int = 2
    use_hyperedge_conv: bool = True
    use_phase_matrix: bool = True
    use_imag_channel: bool = True
    use_class_weights: bool = True
    class_weight_negative_ratio: float = 0.5
    selection_metric: str = "macro_f1"
    device: str = "auto"

    def training_config(self, q1: float, q2: float) -> HyperedgeTrainingConfig:
        return HyperedgeTrainingConfig(
            q1=q1,
            q2=q2,
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            dropout=self.dropout,
            batch_size=self.batch_size,
            patience=self.patience,
            negative_sampling_mode=self.negative_sampling_mode,
            negative_ratio=self.negative_ratio,
            num_layers=self.num_layers,
            use_hyperedge_conv=self.use_hyperedge_conv,
            use_phase_matrix=self.use_phase_matrix,
            use_imag_channel=self.use_imag_channel,
            use_class_weights=self.use_class_weights,
            class_weight_negative_ratio=self.class_weight_negative_ratio,
            selection_metric=self.selection_metric,
        )

    def validate(self) -> None:
        if not self.q1_values or not self.q2_values:
            raise ValueError("q1_values and q2_values must not be empty.")
        if self.outer_folds < 2:
            raise ValueError("outer_folds must be at least 2.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        self.training_config(self.q1_values[0], self.q2_values[0]).validate()


def parse_args() -> SearchConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Nested phase-parameter search for WikiRfA hyperedge classification. "
            "Parameters are selected on inner validation splits, and each outer "
            "test fold is evaluated once."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wiki_phase_search"),
    )
    parser.add_argument(
        "--q1-values",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.15, 0.20, 0.25),
    )
    parser.add_argument(
        "--q2-values",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.15, 0.20, 0.25),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=6000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--negative-sampling-mode",
        choices=sorted(HyperedgeSampler.MODES),
        default="mixed",
    )
    parser.add_argument("--negative-ratio", type=float, default=0.2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--use-hyperedge-conv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-phase-matrix",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-imag-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--selection-metric",
        choices=("macro_f1", "accuracy"),
        default="macro_f1",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = SearchConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        q1_values=tuple(args.q1_values),
        q2_values=tuple(args.q2_values),
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        dropout=args.dropout,
        batch_size=args.batch_size,
        patience=args.patience,
        outer_folds=args.outer_folds,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        negative_sampling_mode=args.negative_sampling_mode,
        negative_ratio=args.negative_ratio,
        num_layers=args.num_layers,
        use_hyperedge_conv=args.use_hyperedge_conv,
        use_phase_matrix=args.use_phase_matrix,
        use_imag_channel=args.use_imag_channel,
        use_class_weights=args.use_class_weights,
        selection_metric=args.selection_metric,
        device=args.device,
    )
    config.validate()
    return config


def candidate_rank(
    metrics: dict[str, float],
    selection_metric: str,
    q1: float,
    q2: float,
) -> tuple[float, float, float, float]:
    primary, secondary = rank_metrics(metrics, selection_metric)
    return primary, secondary, -q1, -q2


def aggregate_trials(
    trial_records: Sequence[dict[str, Any]],
    selection_metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for record in trial_records:
        grouped[(record["q1"], record["q2"])].append(record)

    aggregates: list[dict[str, Any]] = []
    for (q1, q2), records in grouped.items():
        aggregates.append(
            {
                "q1": q1,
                "q2": q2,
                "mean_validation_accuracy": finite_mean(
                    record["validation_accuracy"] for record in records
                ),
                "std_validation_accuracy": finite_std(
                    record["validation_accuracy"] for record in records
                ),
                "mean_validation_macro_f1": finite_mean(
                    record["validation_macro_f1"] for record in records
                ),
                "std_validation_macro_f1": finite_std(
                    record["validation_macro_f1"] for record in records
                ),
                "mean_validation_macro_auc": finite_mean(
                    record["validation_macro_auc"] for record in records
                ),
                "std_validation_macro_auc": finite_std(
                    record["validation_macro_auc"] for record in records
                ),
                "mean_best_epoch": finite_mean(
                    record["best_epoch"] for record in records
                ),
                "folds": len(records),
            }
        )

    primary = (
        "mean_validation_macro_f1"
        if selection_metric == "macro_f1"
        else "mean_validation_accuracy"
    )
    secondary = (
        "mean_validation_accuracy"
        if selection_metric == "macro_f1"
        else "mean_validation_macro_f1"
    )
    aggregates.sort(
        key=lambda record: (
            record[primary],
            record[secondary],
            -record["q1"],
            -record["q2"],
        ),
        reverse=True,
    )
    for rank, record in enumerate(aggregates, start=1):
        record["validation_rank"] = rank
    return aggregates


def main() -> None:
    config = parse_args()
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    payload = asdict(config)
    payload["dataset_path"] = str(config.dataset_path.expanduser().resolve())
    payload["output_dir"] = str(config.output_dir.expanduser().resolve())
    payload["q1_values"] = list(config.q1_values)
    payload["q2_values"] = list(config.q2_values)
    atomic_write_json(payload, config.output_dir / "config.json")

    set_seed(config.seed)
    dataset = load_fixed_hypergraph_dataset(config.dataset_path)
    features = torch.as_tensor(
        dataset.node_features,
        dtype=torch.float32,
        device=device,
    )
    labels = np.asarray(
        [int(edge["type"]) for edge in dataset.hyperedges],
        dtype=np.int64,
    )
    indices = np.arange(dataset.num_hyperedges)
    all_real_edges = (
        set(dataset.all_real_edge_keys)
        if dataset.all_real_edge_keys is not None
        else HyperedgeSampler.get_real_edges_set(dataset.hyperedges)
    )
    combinations = list(itertools.product(config.q1_values, config.q2_values))

    print(f"Device: {device}")
    print(f"Nodes: {dataset.num_nodes}")
    print(f"Real hyperedges: {dataset.num_hyperedges}")
    print(f"Candidate combinations: {len(combinations)}")
    print(f"Selection metric: {config.selection_metric}")

    splitter = StratifiedKFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.seed,
    )
    trial_records: list[dict[str, Any]] = []
    outer_records: list[dict[str, Any]] = []
    started = time.time()

    for outer_fold, (outer_train_indices, test_indices) in enumerate(
        splitter.split(indices, labels),
        start=1,
    ):
        outer_train_labels = labels[outer_train_indices]
        inner_train_indices, validation_indices = train_test_split(
            outer_train_indices,
            test_size=config.validation_fraction,
            random_state=config.seed,
            stratify=outer_train_labels,
        )
        print(
            f"Outer fold {outer_fold}/{config.outer_folds} | "
            f"inner_train={len(inner_train_indices)} | "
            f"validation={len(validation_indices)} | "
            f"test={len(test_indices)}"
        )

        sampling_config = config.training_config(
            config.q1_values[0],
            config.q2_values[0],
        )
        validation_batch = make_evaluation_batch(
            dataset=dataset,
            reference_indices=inner_train_indices,
            evaluation_indices=validation_indices,
            all_real_edges=all_real_edges,
            config=sampling_config,
            seed=config.seed,
        )

        fold_candidates: list[dict[str, Any]] = []
        for candidate_index, (q1, q2) in enumerate(combinations, start=1):
            candidate_config = config.training_config(q1, q2)
            candidate_started = time.time()
            fit = fit_with_validation(
                dataset=dataset,
                features=features,
                train_indices=inner_train_indices,
                validation_batch=validation_batch,
                all_real_edges=all_real_edges,
                config=candidate_config,
                device=device,
                seed=config.seed,
            )
            record = {
                "outer_fold": outer_fold,
                "candidate_index": candidate_index,
                "q1": q1,
                "q2": q2,
                "best_epoch": fit.best_epoch,
                "stopped_epoch": fit.stopped_epoch,
                "validation_accuracy": fit.best_metrics["accuracy"],
                "validation_macro_f1": fit.best_metrics["macro_f1"],
                "validation_macro_auc": fit.best_metrics["macro_auc"],
                "elapsed_seconds": round(time.time() - candidate_started, 3),
            }
            trial_records.append(record)
            fold_candidates.append(record)
            atomic_write_csv(
                trial_records,
                config.output_dir / "validation_trials.csv",
                sort_by=["outer_fold", "candidate_index"],
                ascending=[True, True],
            )
            print(
                f"  [{candidate_index:02d}/{len(combinations):02d}] "
                f"q1={q1:.3f} | q2={q2:.3f} | "
                f"accuracy={record['validation_accuracy']:.4f} | "
                f"macro_f1={record['validation_macro_f1']:.4f} | "
                f"epoch={record['best_epoch']}"
            )

        best_candidate = max(
            fold_candidates,
            key=lambda record: candidate_rank(
                {
                    "accuracy": record["validation_accuracy"],
                    "macro_f1": record["validation_macro_f1"],
                    "macro_auc": record["validation_macro_auc"],
                },
                config.selection_metric,
                record["q1"],
                record["q2"],
            ),
        )
        selected_config = config.training_config(
            best_candidate["q1"],
            best_candidate["q2"],
        )
        test_batch = make_evaluation_batch(
            dataset=dataset,
            reference_indices=outer_train_indices,
            evaluation_indices=test_indices,
            all_real_edges=all_real_edges,
            config=selected_config,
            seed=config.seed,
        )
        final_fit = fit_fixed_epochs(
            dataset=dataset,
            features=features,
            train_indices=outer_train_indices,
            evaluation_batch=test_batch,
            all_real_edges=all_real_edges,
            config=selected_config,
            device=device,
            seed=config.seed,
            epochs=int(best_candidate["best_epoch"]),
            keep_state=False,
        )
        if final_fit.evaluation is None:
            raise RuntimeError("The outer test evaluation was not produced.")
        metrics = final_fit.evaluation.metrics
        outer_record = {
            "outer_fold": outer_fold,
            "selected_q1": best_candidate["q1"],
            "selected_q2": best_candidate["q2"],
            "selected_epoch": best_candidate["best_epoch"],
            "selection_validation_accuracy": best_candidate[
                "validation_accuracy"
            ],
            "selection_validation_macro_f1": best_candidate[
                "validation_macro_f1"
            ],
            "selection_validation_macro_auc": best_candidate[
                "validation_macro_auc"
            ],
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_macro_auc": metrics["macro_auc"],
            "outer_train_size": len(outer_train_indices),
            "test_real_size": len(test_indices),
            "test_evaluation_size": int(final_fit.evaluation.labels.numel()),
        }
        outer_records.append(outer_record)
        atomic_write_csv(
            outer_records,
            config.output_dir / "outer_fold_results.csv",
            sort_by=["outer_fold"],
            ascending=[True],
        )
        print(
            f"  selected q1={outer_record['selected_q1']:.3f} | "
            f"q2={outer_record['selected_q2']:.3f} | "
            f"test_accuracy={outer_record['test_accuracy']:.4f} | "
            f"test_macro_f1={outer_record['test_macro_f1']:.4f}"
        )

    aggregate_records = aggregate_trials(trial_records, config.selection_metric)
    atomic_write_csv(
        aggregate_records,
        config.output_dir / "validation_summary.csv",
        sort_by=["validation_rank"],
        ascending=[True],
    )
    recommended = aggregate_records[0]
    selected_counts = Counter(
        (record["selected_q1"], record["selected_q2"])
        for record in outer_records
    )
    summary = {
        "outer_folds": config.outer_folds,
        "selection_metric": config.selection_metric,
        "recommended_q1": recommended["q1"],
        "recommended_q2": recommended["q2"],
        "mean_test_accuracy": finite_mean(
            record["test_accuracy"] for record in outer_records
        ),
        "std_test_accuracy": finite_std(
            record["test_accuracy"] for record in outer_records
        ),
        "mean_test_macro_f1": finite_mean(
            record["test_macro_f1"] for record in outer_records
        ),
        "std_test_macro_f1": finite_std(
            record["test_macro_f1"] for record in outer_records
        ),
        "mean_test_macro_auc": finite_mean(
            record["test_macro_auc"] for record in outer_records
        ),
        "std_test_macro_auc": finite_std(
            record["test_macro_auc"] for record in outer_records
        ),
        "selected_parameter_counts": [
            {"q1": q1, "q2": q2, "count": count}
            for (q1, q2), count in sorted(selected_counts.items())
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_write_json(summary, config.output_dir / "summary.json")

    print("Search completed.")
    print(
        f"Nested-CV accuracy: {summary['mean_test_accuracy']:.4f} "
        f"± {summary['std_test_accuracy']:.4f}"
    )
    print(
        f"Nested-CV macro F1: {summary['mean_test_macro_f1']:.4f} "
        f"± {summary['std_test_macro_f1']:.4f}"
    )
    print(
        f"Recommended validation parameters: q1={summary['recommended_q1']:.3f}, "
        f"q2={summary['recommended_q2']:.3f}"
    )


if __name__ == "__main__":
    main()
