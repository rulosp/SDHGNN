from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from .common import NodeClassificationDataset, make_torch_sparse


class CitationContentLoader:
    def __init__(
        self,
        data_root: str | Path,
        dataset_name: str,
        content_filename: str,
        cites_filename: str,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.dataset_name = dataset_name
        self.content_path = self.data_root / content_filename
        self.cites_path = self.data_root / cites_filename

    def load(self) -> NodeClassificationDataset:
        if not self.content_path.is_file():
            raise FileNotFoundError(f"Content file not found: {self.content_path}")
        if not self.cites_path.is_file():
            raise FileNotFoundError(f"Citation file not found: {self.cites_path}")

        raw_ids: list[str] = []
        feature_rows: list[list[float]] = []
        raw_labels: list[str] = []

        with self.content_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 3:
                    raise ValueError(
                        f"Malformed content row at line {line_number}: {self.content_path}"
                    )
                raw_ids.append(parts[0])
                feature_rows.append([float(value) for value in parts[1:-1]])
                raw_labels.append(parts[-1])

        if not raw_ids:
            raise ValueError(f"No node rows were found in {self.content_path}.")
        feature_dim = len(feature_rows[0])
        if any(len(row) != feature_dim for row in feature_rows):
            raise ValueError("Feature rows have inconsistent dimensions.")

        node_index = {node_id: index for index, node_id in enumerate(raw_ids)}
        label_values = sorted(set(raw_labels))
        label_index = {label: index for index, label in enumerate(label_values)}

        citations: dict[str, list[str]] = defaultdict(list)
        with self.cites_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                cited_id, citing_id = parts[0], parts[1]
                if cited_id in node_index and citing_id in node_index:
                    citations[citing_id].append(cited_id)

        source_rows: list[int] = []
        source_columns: list[int] = []
        target_rows: list[int] = []
        target_columns: list[int] = []
        for edge_index, (citing_id, cited_ids) in enumerate(citations.items()):
            target_rows.append(node_index[citing_id])
            target_columns.append(edge_index)
            for cited_id in cited_ids:
                source_rows.append(node_index[cited_id])
                source_columns.append(edge_index)

        shape = (len(raw_ids), len(citations))
        return NodeClassificationDataset(
            features=torch.tensor(feature_rows, dtype=torch.float32),
            labels=torch.tensor(
                [label_index[label] for label in raw_labels],
                dtype=torch.long,
            ),
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
            name=self.dataset_name,
        )


class CoraLoader(CitationContentLoader):
    def __init__(self, data_root: str | Path) -> None:
        super().__init__(
            data_root=data_root,
            dataset_name="cora",
            content_filename="cora.content",
            cites_filename="cora.cites",
        )


class CiteseerLoader(CitationContentLoader):
    def __init__(self, data_root: str | Path) -> None:
        super().__init__(
            data_root=data_root,
            dataset_name="citeseer",
            content_filename="citeseer.content",
            cites_filename="citeseer.cites",
        )


class PubmedLoader:
    NODE_FILENAME = "Pubmed-Diabetes.NODE.paper.tab"
    CITES_FILENAME = "Pubmed-Diabetes.DIRECTED.cites.tab"

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.node_path = self.data_root / self.NODE_FILENAME
        self.cites_path = self.data_root / self.CITES_FILENAME
        self.label_index = {1: 0, 2: 1, 3: 2}

    def load(self) -> NodeClassificationDataset:
        if not self.node_path.is_file():
            raise FileNotFoundError(f"Node file not found: {self.node_path}")
        if not self.cites_path.is_file():
            raise FileNotFoundError(f"Citation file not found: {self.cites_path}")

        lines = self.node_path.read_text(encoding="utf-8").splitlines()
        header = next(
            (
                line.split("\t")
                for line in lines
                if "cat=" in line and "numeric:" in line
            ),
            None,
        )
        if header is None:
            raise ValueError("The PubMed feature header could not be located.")

        feature_names = [
            field.split(":", maxsplit=2)[1]
            for field in header
            if field.startswith("numeric:")
        ]
        feature_index = {
            feature_name: index for index, feature_name in enumerate(feature_names)
        }

        raw_ids: list[str] = []
        labels: list[int] = []
        sparse_rows: list[dict[int, float]] = []
        for line in lines:
            fields = line.strip().split("\t")
            if not fields or not fields[0].isdigit():
                continue
            label_field = next(
                (field for field in fields if field.startswith("label=")),
                None,
            )
            if label_field is None:
                continue
            raw_label = int(label_field.split("=", maxsplit=1)[1])
            if raw_label not in self.label_index:
                continue

            row: dict[int, float] = {}
            for field in fields:
                if "=" not in field or field.startswith("label="):
                    continue
                key, value = field.split("=", maxsplit=1)
                if key in feature_index:
                    row[feature_index[key]] = float(value)
            raw_ids.append(fields[0])
            labels.append(self.label_index[raw_label])
            sparse_rows.append(row)

        if not raw_ids:
            raise ValueError(f"No PubMed node rows were found in {self.node_path}.")

        features = torch.zeros(
            (len(raw_ids), len(feature_names)),
            dtype=torch.float32,
        )
        for row_index, row in enumerate(sparse_rows):
            for column_index, value in row.items():
                features[row_index, column_index] = value

        node_index = {node_id: index for index, node_id in enumerate(raw_ids)}
        citations: dict[str, list[str]] = defaultdict(list)
        with self.cites_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.strip().split("\t")
                if len(fields) < 4:
                    continue
                if "paper:" not in fields[1] or "paper:" not in fields[3]:
                    continue
                citing_id = fields[1].replace("paper:", "")
                cited_id = fields[3].replace("paper:", "")
                if citing_id in node_index and cited_id in node_index:
                    citations[citing_id].append(cited_id)

        source_rows: list[int] = []
        source_columns: list[int] = []
        target_rows: list[int] = []
        target_columns: list[int] = []
        for edge_index, (citing_id, cited_ids) in enumerate(citations.items()):
            target_rows.append(node_index[citing_id])
            target_columns.append(edge_index)
            for cited_id in cited_ids:
                source_rows.append(node_index[cited_id])
                source_columns.append(edge_index)

        shape = (len(raw_ids), len(citations))
        return NodeClassificationDataset(
            features=features,
            labels=torch.tensor(labels, dtype=torch.long),
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
            name="pubmed",
        )
