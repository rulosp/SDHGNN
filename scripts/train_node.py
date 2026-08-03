from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from sdhgnn.config import dataclass_from_dict, load_json
from sdhgnn.data import CiteseerLoader, CoraLoader, DblpLoader, PubmedLoader
from sdhgnn.training import (
    NodeTrainingConfig,
    atomic_save_npz,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    finite_mean,
    finite_std,
    fit_node_fixed_epochs,
    fit_node_with_validation,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validated node classification with validation-only epoch selection."
    )
    parser.add_argument(
        "--dataset-name",
        choices=("cora", "citeseer", "pubmed", "dblp"),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def load_dataset(name: str, data_root: Path):
    loaders = {
        "cora": CoraLoader,
        "citeseer": CiteseerLoader,
        "pubmed": PubmedLoader,
        "dblp": DblpLoader,
    }
    return loaders[name](data_root).load()


def validation_logger(fold: int, total_epochs: int, log_every: int):
    def log(record: dict[str, float | int]) -> None:
        epoch = int(record["epoch"])
        if epoch == 1 or epoch % log_every == 0 or epoch == total_epochs:
            print(
                f"Fold {fold} validation | epoch={epoch:04d} | "
                f"loss={float(record['train_loss']):.4f} | "
                f"accuracy={float(record['validation_accuracy']):.4f} | "
                f"macro_f1={float(record['validation_macro_f1']):.4f}"
            )

    return log


def training_logger(fold: int, epochs: int, log_every: int):
    def log(record: dict[str, float | int]) -> None:
        epoch = int(record["epoch"])
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"Fold {fold} retraining | epoch={epoch:04d}/{epochs:04d} | "
                f"loss={float(record['train_loss']):.4f}"
            )

    return log


def main() -> None:
    args = parse_args()
    if args.outer_folds < 2:
        raise ValueError("outer-folds must be at least 2.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1.")

    config = dataclass_from_dict(NodeTrainingConfig, load_json(args.config))
    config.validate()
    device = resolve_device(args.device)
    dataset = load_dataset(args.dataset_name, args.data_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = dataset.labels.cpu().numpy()
    indices = np.arange(dataset.num_nodes)
    atomic_write_json(
        {
            "dataset_name": args.dataset_name,
            "data_root": str(args.data_root.expanduser().resolve()),
            "training": asdict(config),
            "outer_folds": args.outer_folds,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "device": str(device),
            "num_nodes": dataset.num_nodes,
            "num_hyperedges": dataset.num_hyperedges,
            "feature_dim": dataset.feature_dim,
            "num_classes": dataset.num_classes,
        },
        output_dir / "config.json",
    )

    splitter = StratifiedKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.seed,
    )
    fold_records: list[dict[str, float | int]] = []
    prediction_records: list[dict[str, float | int]] = []
    split_arrays: dict[str, np.ndarray] = {}
    started = time.time()

    for fold, (outer_train, test_indices) in enumerate(
        splitter.split(indices, labels),
        start=1,
    ):
        inner_train, validation_indices = train_test_split(
            outer_train,
            test_size=args.validation_fraction,
            random_state=args.seed,
            stratify=labels[outer_train],
        )
        split_arrays[f"fold_{fold:02d}_inner_train"] = inner_train
        split_arrays[f"fold_{fold:02d}_validation"] = validation_indices
        split_arrays[f"fold_{fold:02d}_outer_train"] = outer_train
        split_arrays[f"fold_{fold:02d}_test"] = test_indices

        selection = fit_node_with_validation(
            dataset=dataset,
            train_indices=inner_train,
            validation_indices=validation_indices,
            config=config,
            device=device,
            seed=args.seed,
            epoch_callback=validation_logger(
                fold,
                config.epochs,
                args.log_every,
            ),
        )
        final_fit = fit_node_fixed_epochs(
            dataset=dataset,
            train_indices=outer_train,
            test_indices=test_indices,
            config=config,
            device=device,
            seed=args.seed,
            epochs=selection.best_epoch,
            epoch_callback=training_logger(
                fold,
                selection.best_epoch,
                args.log_every,
            ),
        )

        fold_records.append(
            {
                "fold": fold,
                "best_epoch": selection.best_epoch,
                "stopped_epoch": selection.stopped_epoch,
                "validation_accuracy": selection.best_metrics["accuracy"],
                "validation_macro_f1": selection.best_metrics["macro_f1"],
                "validation_macro_auc": selection.best_metrics["macro_auc"],
                "test_accuracy": final_fit.metrics["accuracy"],
                "test_macro_f1": final_fit.metrics["macro_f1"],
                "test_macro_auc": final_fit.metrics["macro_auc"],
                "inner_train_size": len(inner_train),
                "validation_size": len(validation_indices),
                "outer_train_size": len(outer_train),
                "test_size": len(test_indices),
            }
        )

        probabilities = torch.softmax(final_fit.logits, dim=1).numpy()
        predictions = probabilities.argmax(axis=1)
        for row, node_index in enumerate(test_indices):
            record: dict[str, float | int] = {
                "fold": fold,
                "node_index": int(node_index),
                "target": int(labels[node_index]),
                "prediction": int(predictions[row]),
            }
            for class_index, value in enumerate(probabilities[row]):
                record[f"probability_class_{class_index}"] = float(value)
            prediction_records.append(record)

        classes = list(range(dataset.num_classes))
        atomic_write_json(
            {
                "classification_report": classification_report(
                    labels[test_indices],
                    predictions,
                    labels=classes,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    labels[test_indices],
                    predictions,
                    labels=classes,
                ),
            },
            output_dir / "reports" / f"fold_{fold:02d}.json",
        )
        atomic_torch_save(
            {
                "fold": fold,
                "training_config": asdict(config),
                "best_epoch": selection.best_epoch,
                "outer_train_indices": outer_train,
                "test_indices": test_indices,
                "model_state": final_fit.model_state,
            },
            output_dir / "checkpoints" / f"fold_{fold:02d}.pt",
        )
        atomic_write_csv(fold_records, output_dir / "fold_results.csv")
        atomic_write_csv(prediction_records, output_dir / "predictions.csv")
        print(
            f"Fold {fold} test | accuracy={final_fit.metrics['accuracy']:.4f} | "
            f"macro_f1={final_fit.metrics['macro_f1']:.4f} | "
            f"macro_auc={final_fit.metrics['macro_auc']:.4f}"
        )

    atomic_save_npz(split_arrays, output_dir / "splits.npz")
    summary = {
        "mean_test_accuracy": finite_mean(
            record["test_accuracy"] for record in fold_records
        ),
        "std_test_accuracy": finite_std(
            record["test_accuracy"] for record in fold_records
        ),
        "mean_test_macro_f1": finite_mean(
            record["test_macro_f1"] for record in fold_records
        ),
        "std_test_macro_f1": finite_std(
            record["test_macro_f1"] for record in fold_records
        ),
        "mean_test_macro_auc": finite_mean(
            record["test_macro_auc"] for record in fold_records
        ),
        "std_test_macro_auc": finite_std(
            record["test_macro_auc"] for record in fold_records
        ),
        "mean_selected_epoch": finite_mean(
            record["best_epoch"] for record in fold_records
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_write_json(summary, output_dir / "summary.json")
    print(
        f"Accuracy: {summary['mean_test_accuracy']:.4f} "
        f"± {summary['std_test_accuracy']:.4f}"
    )
    print(
        f"Macro F1: {summary['mean_test_macro_f1']:.4f} "
        f"± {summary['std_test_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
