from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("The configuration root must be a JSON object.")
    return payload


def dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    valid_names = {field.name for field in fields(cls)}
    unknown = set(payload) - valid_names
    if unknown:
        raise ValueError(
            f"Unsupported configuration fields for {cls.__name__}: {sorted(unknown)}"
        )
    return cls(**payload)
