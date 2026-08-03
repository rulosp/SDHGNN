from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sdhgnn.data import (
    SignedNetworkBuildConfig,
    SignedNetworkHypergraphBuilder,
    compare_fixed_datasets,
    load_fixed_hypergraph_dataset,
    save_fixed_hypergraph_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed WikiRfA or Slashdot hypergraph using the original "
            "dataset-construction procedure."
        )
    )
    parser.add_argument("--dataset-name", choices=("wiki", "slashdot"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k-hop", type=int, default=2)
    parser.add_argument("--min-size", type=int, required=True)
    parser.add_argument("--max-size", type=int, required=True)
    parser.add_argument("--min-center-degree", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SignedNetworkBuildConfig(
        dataset_name=args.dataset_name,
        input_path=args.input,
        k_hop=args.k_hop,
        min_size=args.min_size,
        max_size=args.max_size,
        min_center_degree=args.min_center_degree,
        seed=args.seed,
    )
    dataset = SignedNetworkHypergraphBuilder(config).build()

    if args.reference is not None:
        reference = load_fixed_hypergraph_dataset(args.reference)
        comparison = compare_fixed_datasets(reference, dataset)
        for name, passed in comparison["checks"].items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")
        if not comparison["exact_match"]:
            difference = comparison["max_node_feature_absolute_difference"]
            if difference is not None:
                print(f"Maximum node-feature absolute difference: {difference:.9g}")
            raise RuntimeError(
                "The rebuilt dataset does not exactly match the reference. "
                "No output file was written."
            )
        print("Exact reference match: PASS")

    destination = save_fixed_hypergraph_dataset(
        dataset,
        args.output,
        overwrite=args.overwrite,
    )
    label_counts = Counter(int(edge["type"]) for edge in dataset.hyperedges)
    print(f"Saved fixed dataset: {destination}")
    print(f"Nodes: {dataset.num_nodes}")
    print(f"Hyperedges: {dataset.num_hyperedges}")
    print(f"Label counts: {dict(sorted(label_counts.items()))}")


if __name__ == "__main__":
    main()
