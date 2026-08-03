from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def scipy_to_torch_sparse(
    matrix: sp.spmatrix,
    device: torch.device | str,
) -> torch.Tensor:
    matrix = matrix.tocoo()
    indices = torch.as_tensor(
        np.vstack((matrix.row, matrix.col)),
        dtype=torch.long,
        device=device,
    )
    values = torch.as_tensor(
        matrix.data,
        dtype=torch.float32,
        device=device,
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=matrix.shape,
        device=device,
    ).coalesce()


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    targets = labels.detach().cpu().numpy()

    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
    }

    if np.unique(targets).size < num_classes:
        metrics["macro_auc"] = float("nan")
    else:
        try:
            one_hot = np.eye(num_classes, dtype=np.float32)[targets]
            metrics["macro_auc"] = float(
                roc_auc_score(
                    one_hot,
                    probabilities,
                    average="macro",
                    multi_class="ovr",
                )
            )
        except ValueError:
            metrics["macro_auc"] = float("nan")
    return metrics


def make_class_weights(
    real_labels: Iterable[int],
    num_classes: int,
    negative_ratio: float,
    device: torch.device | str,
) -> torch.Tensor:
    labels = np.fromiter((int(label) for label in real_labels), dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[0] = max(counts[0], len(labels) * negative_ratio)

    weights = np.zeros_like(counts)
    observed = counts > 0
    weights[observed] = 1.0 / np.sqrt(counts[observed])
    if not observed.any():
        raise ValueError("No class observations are available for weighting.")
    weights *= num_classes / weights.sum()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def to_json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [to_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())
    if isinstance(value, torch.Tensor):
        return to_json_compatible(value.detach().cpu().tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_csv(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    sort_by: list[str] | None = None,
    ascending: list[bool] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(list(records))
    if sort_by and not frame.empty:
        frame = frame.sort_values(
            by=sort_by,
            ascending=ascending,
        ).reset_index(drop=True)

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(
                to_json_compatible(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def atomic_save_npz(
    arrays: Mapping[str, np.ndarray],
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".npz",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def finite_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.std(ddof=0)) if finite.size else float("nan")
