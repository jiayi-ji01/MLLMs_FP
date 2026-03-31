# Cross-Lingual Knowledge Transfer in Large Language Models

This repository studies whether factual knowledge learned from English supervision can still be retrieved when the same facts are asked in a synthetic fake language. It includes the full pipeline for synthetic data construction, QLoRA training, evaluation, hidden-state analysis, activation patching, and figure generation.

## Repository Overview

- `src/`: core scripts for dataset building, training, inference, evaluation, analysis, and plotting.
- `configs/`: YAML configs for each stage of the pipeline.
- `data/raw/`: generated synthetic facts and QA candidates.
- `data/processed/`: train/dev/test splits and SFT-formatted JSONL files.
- `outputs/checkpoints/`: trained LoRA/QLoRA adapters.
- `outputs/predictions/`: model generations on dev/test splits.
- `outputs/metrics/`: exact-match and transfer metrics.
- `outputs/figures/`: behavior and mechanism figures.
- `notebooks/`: Colab notebooks for running the experiment and reviewing analysis.

## Minimal Workflow

```bash
python src/build_dataset.py --config configs/dataset.yaml
python src/make_splits.py --config configs/dataset.yaml
python src/format_for_sft.py --config configs/dataset.yaml

python src/train_lora.py --config configs/train_main.yaml
python src/inference.py --config configs/inference.yaml
python src/evaluate.py --config configs/evaluate.yaml
```

Optional analysis:

```bash
python src/extract_hidden_states.py --config configs/analysis_hidden_states.yaml
python src/analyze_hidden_states.py --config configs/analysis_similarity.yaml
python src/activation_patching.py --config configs/analysis_patching.yaml
python src/plot_main_results.py --config configs/plot_main.yaml
```

## Current Experiment Setup

- Base model: `Meta-Llama-3.1-8B-Instruct`
- Main training condition: English-only QLoRA
- Control condition: fake-language supervision via `configs/train_fake_control.yaml`
- Relations: `lives_in`, `discovered`, `symbol_is`
- Main processed data root: `data/processed/english_to_fake_transfer`

## Notes

- Config files are the source of truth for paths, run names, and analysis settings.
- The repository already contains generated outputs for the main and fake-control runs.
