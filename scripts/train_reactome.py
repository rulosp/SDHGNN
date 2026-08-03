from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from sdhgnn.config import dataclass_from_dict, load_json
from sdhgnn.data import ReactomeDatasetLoader
from sdhgnn.sampling import HyperedgeSampler
from sdhgnn.training import (
    HyperedgeTrainingConfig,
    atomic_save_npz,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    finite_mean,
    finite_std,
    fit_fixed_epochs,
    fit_with_validation,
    make_evaluation_batch,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reactome signed hyperedge classification with full-dataset degree features."
    )
    parser.add_argument("--hyperedges", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inhibition-pattern", choices=("+-", "-+"), default="+-")
    parser.add_argument(
        "--balance-real-classes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def logger(fold: int, total_epochs: int, log_every: int, validation: bool):
    def log(record: dict[str, float | int]) -> None:
        epoch = int(record["epoch"])
        if epoch == 1 or epoch % log_every == 0 or epoch == total_epochs:
            text = (
                f"Fold {fold} validation | epoch={epoch:04d} | "
                f"loss={float(record['train_loss']):.4f} | "
                f"accuracy={float(record['validation_accuracy']):.4f} | "
                f"macro_f1={float(record['validation_macro_f1']):.4f}"
                if validation
                else f"Fold {fold} retraining | epoch={epoch:04d}/{total_epochs:04d} | "
                f"loss={float(record['train_loss']):.4f}"
            )
            print(text)

    return log


def main() -> None:
    args = parse_args()
    if args.outer_folds < 2:
        raise ValueError("outer-folds must be at least 2.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1.")

    config = dataclass_from_dict(
        HyperedgeTrainingConfig,
        load_json(args.config),
    )
    config.validate()
    device = resolve_device(args.device)
    dataset = ReactomeDatasetLoader(
        hyperedge_path=args.hyperedges,
        inhibition_pattern=args.inhibition_pattern,
        balance_real_classes=args.balance_real_classes,
        seed=args.seed,
    ).load()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    atomic_write_json(
        {
            "hyperedge_path": str(args.hyperedges.expanduser().resolve()),
            "feature_type": "degree",
            "feature_scope": "full_dataset",
            "inhibition_pattern": args.inhibition_pattern,
            "balance_real_classes": args.balance_real_classes,
            "training": asdict(config),
            "outer_folds": args.outer_folds,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "device": str(device),
            "dataset_metadata": dataset.metadata,
        },
        output_dir / "config.json",
    )

    splitter = StratifiedKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.seed,
    )
    fold_records: list[dict[str, Any]] = []
    predictions: list[dict[str, float | int]] = []
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

        features = torch.as_tensor(
            dataset.node_features,
            dtype=torch.float32,
            device=device,
        )
        evaluation_python_rng = random.Random(args.seed)
        evaluation_numpy_rng = np.random.RandomState(args.seed)
        validation_batch = make_evaluation_batch(
            dataset=dataset,
            reference_indices=inner_train,
            evaluation_indices=validation_indices,
            all_real_edges=all_real_edges,
            config=config,
            seed=args.seed,
            python_rng=evaluation_python_rng,
            numpy_rng=evaluation_numpy_rng,
        )
        test_batch = make_evaluation_batch(
            dataset=dataset,
            reference_indices=outer_train,
            evaluation_indices=test_indices,
            all_real_edges=all_real_edges,
            config=config,
            seed=args.seed,
            python_rng=evaluation_python_rng,
            numpy_rng=evaluation_numpy_rng,
        )
        selection = fit_with_validation(
            dataset=dataset,
            features=features,
            train_indices=inner_train,
            validation_batch=validation_batch,
            all_real_edges=all_real_edges,
            config=config,
            device=device,
            seed=args.seed,
            epoch_callback=logger(
                fold,
                config.epochs,
                args.log_every,
                validation=True,
            ),
        )

        final_fit = fit_fixed_epochs(
            dataset=dataset,
            features=features,
            train_indices=outer_train,
            evaluation_batch=test_batch,
            all_real_edges=all_real_edges,
            config=config,
            device=device,
            seed=args.seed,
            epochs=selection.best_epoch,
            keep_state=True,
            epoch_callback=logger(
                fold,
                selection.best_epoch,
                args.log_every,
                validation=False,
            ),
        )
        if final_fit.evaluation is None:
            raise RuntimeError("The fold evaluation was not produced.")
        evaluation = final_fit.evaluation
        fold_records.append(
            {
                "fold": fold,
                "best_epoch": selection.best_epoch,
                "stopped_epoch": selection.stopped_epoch,
                "validation_accuracy": selection.best_metrics["accuracy"],
                "validation_macro_f1": selection.best_metrics["macro_f1"],
                "validation_macro_auc": selection.best_metrics["macro_auc"],
                "test_accuracy": evaluation.metrics["accuracy"],
                "test_macro_f1": evaluation.metrics["macro_f1"],
                "test_macro_auc": evaluation.metrics["macro_auc"],
                "inner_train_size": len(inner_train),
                "validation_size": len(validation_indices),
                "outer_train_size": len(outer_train),
                "test_real_size": len(test_indices),
            }
        )

        probabilities = torch.softmax(evaluation.logits, dim=1).numpy()
        predicted = probabilities.argmax(axis=1)
        targets = evaluation.labels.numpy()
        for row, (target, prediction) in enumerate(zip(targets, predicted)):
            is_fake = row >= len(test_indices)
            record: dict[str, float | int] = {
                "fold": fold,
                "sample_index": row,
                "is_fake": int(is_fake),
                "real_hyperedge_index": (
                    -1 if is_fake else int(test_indices[row])
                ),
                "target": int(target),
                "prediction": int(prediction),
            }
            for class_index, value in enumerate(probabilities[row]):
                record[f"probability_class_{class_index}"] = float(value)
            predictions.append(record)

        classes = list(range(dataset.num_classes))
        atomic_write_json(
            {
                "classification_report": classification_report(
                    targets,
                    predicted,
                    labels=classes,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    targets,
                    predicted,
                    labels=classes,
                ),
            },
            output_dir / "reports" / f"fold_{fold:02d}.json",
        )
        atomic_torch_save(
            {
                "fold": fold,
                "feature_type": "degree",
                "feature_scope": "full_dataset",
                "training_config": asdict(config),
                "best_epoch": selection.best_epoch,
                "outer_train_indices": outer_train,
                "test_indices": test_indices,
                "encoder_state": final_fit.encoder_state,
                "decoder_state": final_fit.decoder_state,
            },
            output_dir / "checkpoints" / f"fold_{fold:02d}.pt",
        )
        atomic_write_csv(fold_records, output_dir / "fold_results.csv")
        atomic_write_csv(predictions, output_dir / "predictions.csv")
        print(
            f"Fold {fold} test | accuracy={evaluation.metrics['accuracy']:.4f} | "
            f"macro_f1={evaluation.metrics['macro_f1']:.4f} | "
            f"macro_auc={evaluation.metrics['macro_auc']:.4f}"
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
