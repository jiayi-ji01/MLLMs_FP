# Cross-Lingual Knowledge Transfer in Large Language Models

This project studies whether factual knowledge learned from English supervision remains accessible when the same facts are queried in a synthetic fake language. The repository includes dataset construction, QLoRA training, evaluation, hidden-state analysis, activation patching, and figure generation.

## Project Structure

```text
.
├── README.md
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
│   ├── analysis/
│   ├── checkpoints/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── results_overview.ipynb
└── src/
```

- `configs/`: YAML configs for dataset generation, training, inference, evaluation, analysis, and plotting.
- `data/raw/`: Raw synthetic facts and QA candidates created from the dataset config.
- `data/processed/`: Split files and SFT-ready JSONL files used by training and evaluation.
- `notebooks/`: Colab notebooks for end-to-end experiments and internal analysis.
- `outputs/analysis/`: Hidden states, similarity summaries, and activation patching results.
- `outputs/checkpoints/`: LoRA or QLoRA checkpoints for the main and control runs.
- `outputs/figures/`: Final behavior and mechanism figures.
- `outputs/metrics/`: Exact-match metrics, transfer efficiency, and per-relation scores.
- `outputs/predictions/`: Generated answers for evaluation splits.
- `results_overview.ipynb`: Compact notebook for reviewing the generated outputs.
- `src/`: Python scripts for data creation, modeling, evaluation, and analysis.

## Script Overview

### Dataset and preprocessing

- `src/build_dataset.py`: Builds the synthetic English-to-fake dataset from `configs/dataset.yaml`, including subjects, relation-specific facts, and QA candidates.
- `src/make_splits.py`: Creates train, dev, and test splits with per-relation stratification and prepares the fake-control training rows.
- `src/format_for_sft.py`: Converts processed QA rows into chat-style SFT JSONL files for English-only training, English-plus-fake control training, and split-by-language evaluation.
- `src/fake_language.py`: Defines the fake-language mapping and transformation utilities used during dataset construction.

### Training and inference

- `src/train_lora.py`: Trains a LoRA or QLoRA adapter for either the main English-only condition or the fake-language control condition.
- `src/inference.py`: Loads the base model and adapter, runs batched generation on evaluation splits, and writes prediction files with metadata.
- `src/evaluate.py`: Scores predictions with normalized exact match, computes transfer efficiency, and reports per-relation accuracy.

### Representation and causal analysis

- `src/extract_hidden_states.py`: Extracts layer-wise hidden states for paired English and fake-language evaluation examples.
- `src/analyze_hidden_states.py`: Computes similarity summaries from the extracted hidden states, including subject-pool and relation-pool comparisons.
- `src/activation_patching.py`: Runs layer-wise activation patching to measure how English activations rescue fake-language predictions.
- `src/analysis_utils.py`: Shared helpers for hidden-state processing, similarity aggregation, and patching analysis.

### Plotting and utilities

- `src/plot_main_results.py`: Generates the main paper-style figures from metrics and analysis outputs.
- `src/utils.py`: Shared file, config, and serialization helpers.

## Key Outputs

- Checkpoints: `outputs/checkpoints/`
- Predictions: `outputs/predictions/`
- Metrics: `outputs/metrics/`
- Hidden-state and patching analysis: `outputs/analysis/`
- Figures: `outputs/figures/`

## Notebooks

- `notebooks/colab_run_experiment.ipynb`: End-to-end experiment notebook.
- `notebooks/colab_internal_analysis.ipynb`: Hidden-state and patching analysis notebook.
- `results_overview.ipynb`: Compact local review of the generated results.

## Notes

- The main experiments use `Meta-Llama-3.1-8B-Instruct` with QLoRA.
- The dataset defines three relations: `lives_in`, `discovered`, and `symbol_is`.
- Config files are the source of truth for paths, run names, and analysis settings.
