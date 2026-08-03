# SDHGNN

This repository provides the SDHGNN implementation for node classification and signed directed hyperedge prediction.

Supported tasks:

- node classification on Cora, Citeseer, PubMed, and DBLP;
- directed signed hyperedge prediction on WikiRfA and Slashdot;
- signed biological hyperedge prediction on Reactome.

The code separates data loading, fixed-dataset construction, negative sampling, model definition, training, and evaluation. All experiment entry points use random seed `0` by default, and no fold-dependent seed offsets are applied.

## Environment Setup

1. Python 3.10.20
2. PyTorch 2.7.0 with CUDA 12.8 support
3. PyTorch Geometric 2.7.0
4. PyTorch Geometric Signed Directed 0.3.1
5. torch-scatter 2.1.2
6. torch-sparse 0.6.18
7. NumPy 2.2.6
8. SciPy 1.15.3
9. pandas 2.3.3
10. scikit-learn 1.7.2
11. NetworkX 2.6.3
12. tqdm 4.67.3


## Project structure

```text
SDHGNN/
├── configs/
│   ├── hyperedge/
│   └── node/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
├── src/
│   └── sdhgnn/
│       ├── data/
│       ├── models/
│       ├── sampling/
│       └── training/
└── README.md
```

## Evaluation protocol

All tasks use five-fold stratified cross-validation by default. In each outer fold, 20% of the samples are held out for testing. The remaining 80% are divided into an inner training subset and a validation subset using a 75%/25% split, corresponding to an overall 60%/20%/20% training/validation/test ratio during epoch selection.

The validation subset selects the training duration according to validation Macro-F1. The model is then reinitialized and trained on the complete outer training partition, which contains 80% of the samples, for the selected number of epochs. The held-out outer test fold is evaluated once.

All data splits, model initialization, training, and negative sampling use seed `0` unless the command-line seed is explicitly changed.

## Fixed WikiRfA and Slashdot datasets

The supplied files below are the authoritative datasets used by the training scripts:

```text
data/processed/wiki_hypergraph_k2_min20_max50_deg50_seed0.pkl
data/processed/slashdot_hypergraph_k2_min40_max80_deg80_seed0.pkl
```

Training reads these files directly and never rebuilds them automatically.

### Rebuild and verify WikiRfA

```bash
python scripts/build_signed_dataset.py \
  --dataset-name wiki \
  --input data/raw/wiki/edges.csv \
  --output outputs/wiki_hypergraph_rebuilt.pkl \
  --reference data/processed/wiki_hypergraph_k2_min20_max50_deg50_seed0.pkl \
  --k-hop 2 \
  --min-size 20 \
  --max-size 50 \
  --min-center-degree 50 \
  --seed 0
```

### Rebuild and verify Slashdot

```bash
python scripts/build_signed_dataset.py \
  --dataset-name slashdot \
  --input data/raw/slashdot/soc-sign-Slashdot081106.txt \
  --output outputs/slashdot_hypergraph_rebuilt.pkl \
  --reference data/processed/slashdot_hypergraph_k2_min40_max80_deg80_seed0.pkl \
  --k-hop 2 \
  --min-size 40 \
  --max-size 80 \
  --min-center-degree 80 \
  --seed 0
```

When `--reference` is provided, the output file is written only if the rebuilt dataset exactly matches the reference in node order, node features, source incidence, target incidence, hyperedge order and content, base graph, and raw edge count.

Two existing fixed files can also be compared directly:

```bash
python scripts/verify_fixed_dataset.py \
  --reference data/processed/wiki_hypergraph_k2_min20_max50_deg50_seed0.pkl \
  --candidate outputs/wiki_hypergraph_rebuilt.pkl
```

The construction code evaluates `log1p` in double precision before storing single-precision node features, so the supplied WikiRfA and Slashdot references are reproduced exactly.

## Node classification

Cora:

```bash
python scripts/train_node.py \
  --dataset-name cora \
  --data-root data/raw/cora \
  --config configs/node/cora.json \
  --output-dir outputs/cora \
  --seed 0
```

Citeseer, PubMed, and DBLP use the same entry point with their corresponding dataset names, data directories, and configuration files. The four node-classification configurations use the selected cocitation experiment settings.

## WikiRfA and Slashdot hyperedge prediction

WikiRfA:

```bash
python scripts/train_hyperedge.py \
  --dataset data/processed/wiki_hypergraph_k2_min20_max50_deg50_seed0.pkl \
  --config configs/hyperedge/wiki.json \
  --output-dir outputs/wiki \
  --seed 0
```

Slashdot:

```bash
python scripts/train_hyperedge.py \
  --dataset data/processed/slashdot_hypergraph_k2_min40_max80_deg80_seed0.pkl \
  --config configs/hyperedge/slashdot.json \
  --output-dir outputs/slashdot \
  --seed 0
```

For WikiRfA and Slashdot, the number of fake hyperedges is

```text
floor(number_of_real_hyperedges_in_the_batch × negative_ratio)
```

with `negative_ratio=0.2`. During training, real hyperedges are sampled with replacement from each shuffled minibatch, matching the original experiment procedure. Validation and test batches include every real hyperedge in the corresponding split. Random, local, perturbation, and mixed negative-sampling modes follow the original sampling rules. Duplicate node indices are removed while preserving their first occurrence. For WikiRfA, the fake-class count used to calculate class weights remains `0.5 × number_of_real_training_hyperedges`, matching the original training script; this is independent of the actual `0.2` negative-sampling ratio.

## Reactome hyperedge prediction

Reactome uses one six-dimensional degree feature representation:

```text
source degree
target degree
activation-source degree
activation-target degree
inhibition-source degree
inhibition-target degree
```

The features are computed once from all selected real hyperedges and transformed with `log(1 + x)`. The propagation matrices in each training stage are still constructed only from the corresponding training hyperedges.

Human:

```bash
python scripts/train_reactome.py \
  --hyperedges data/raw/reactome/human/reactome_signed_lcc_hyperedges.csv \
  --config configs/hyperedge/reactome.json \
  --output-dir outputs/reactome_human \
  --seed 0
```

Use the corresponding species directory for mouse, rat, or drosophila.

Reactome uses `negative_ratio=0.5`. The fake-hyperedge count is:

```text
max(1, floor(number_of_real_hyperedges_in_the_batch × 0.5))
```

Class-weighted cross-entropy and a two-layer encoder are enabled by the Reactome configuration.

## WikiRfA phase search

```bash
python scripts/search_wiki_phase.py \
  --dataset data/processed/wiki_hypergraph_k2_min20_max50_deg50_seed0.pkl \
  --output-dir outputs/wiki_phase_search \
  --q1-values 0.05 0.10 0.15 0.20 0.25 \
  --q2-values 0.05 0.10 0.15 0.20 0.25 \
  --seed 0
```

Candidate phase parameters are ranked using inner-validation Macro-F1. Each outer test fold is evaluated only after parameter selection.

## Model variants

The complete model uses:

```json
{
  "use_phase_matrix": true,
  "use_imag_channel": true
}
```

The strictly real-valued variant uses:

```json
{
  "use_phase_matrix": false,
  "use_imag_channel": false
}
```
