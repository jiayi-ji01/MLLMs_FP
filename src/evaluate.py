"""Score exact-match accuracy for the main and control runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
from pathlib import Path
from typing import Any

from utils import ensure_dir, load_yaml, read_jsonl, write_json, write_jsonl


PUNCT_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate English -> Fake transfer predictions.")
    parser.add_argument("--config", default="configs/evaluate.yaml")
    return parser.parse_args()


def normalize_text(text: str, normalization_cfg: dict[str, Any]) -> str:
    """Normalize predictions and answers before exact matching."""
    normalized = text.strip()
    if normalization_cfg.get("lowercase", True):
        normalized = normalized.lower()
    if normalization_cfg.get("strip_punctuation", True):
        normalized = PUNCT_PATTERN.sub(" ", normalized)
    if normalization_cfg.get("collapse_whitespace", True):
        normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def score_predictions(rows: list[dict[str, Any]], normalization_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach normalized fields and exact-match correctness."""
    scored_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for row in rows:
        scored = dict(row)
        scored["normalized_prediction"] = normalize_text(row["prediction"], normalization_cfg)
        scored["normalized_gold_answer"] = normalize_text(row["gold_answer"], normalization_cfg)
        scored["correct"] = scored["normalized_prediction"] == scored["normalized_gold_answer"]
        scored_rows.append(scored)
        if not scored["correct"]:
            errors.append(scored)

    return scored_rows, errors


def accuracy(rows: list[dict[str, Any]]) -> float:
    """Compute exact-match accuracy for a list of scored rows."""
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["correct"]) / len(rows)


def compute_per_relation(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Compute per-relation accuracy and counts."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["relation"]].append(row)

    metrics: dict[str, dict[str, float | int]] = {}
    for relation, relation_rows in grouped.items():
        metrics[relation] = {
            "count": len(relation_rows),
            "accuracy": accuracy(relation_rows),
        }
    return metrics


def compute_transfer_efficiency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute fake_correct / english_correct using paired evaluation rows."""
    english_rows = [row for row in rows if row["language"] == "en"]
    fake_rows = [row for row in rows if row["language"] == "fake"]
    english_correct = sum(1 for row in english_rows if row["correct"])
    fake_correct = sum(1 for row in fake_rows if row["correct"])
    efficiency = fake_correct / english_correct if english_correct else 0.0

    pair_stats: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id:
            pair_stats[pair_id][row["language"]] = row["correct"]

    fully_transferred = sum(1 for values in pair_stats.values() if values.get("en") and values.get("fake"))
    english_only = sum(1 for values in pair_stats.values() if values.get("en") and not values.get("fake"))

    return {
        "english_correct": english_correct,
        "fake_correct": fake_correct,
        "transfer_efficiency": efficiency,
        "paired_items": len(pair_stats),
        "paired_success_count": fully_transferred,
        "paired_english_only_count": english_only,
    }


def evaluate_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all metrics for one evaluation split."""
    english_rows = [row for row in rows if row["language"] == "en"]
    fake_rows = [row for row in rows if row["language"] == "fake"]

    return {
        "num_examples": len(rows),
        "languages": dict(Counter(row["language"] for row in rows)),
        "source_accuracy": accuracy(english_rows),
        "fake_transfer_accuracy": accuracy(fake_rows),
        "per_relation_accuracy": {
            "en": compute_per_relation(english_rows),
            "fake": compute_per_relation(fake_rows),
        },
        "transfer": compute_transfer_efficiency(rows),
    }


def build_run_metadata(config: dict[str, Any], split: str) -> dict[str, Any]:
    """Collect run metadata used later in behavior comparisons."""
    metadata_cfg = config.get("metadata", {})
    return {
        "run_name": metadata_cfg.get("run_name", "default"),
        "train_language_mode": metadata_cfg.get("train_language_mode", "en_only"),
        "model_name": metadata_cfg.get("model_name", ""),
        "adapter_path": metadata_cfg.get("adapter_path", ""),
        "split": split,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    predictions_dir = Path(config["input"]["predictions_dir"])
    metrics_dir = ensure_dir(config["output"]["metrics_dir"])

    all_errors: list[dict[str, Any]] = []
    split_metrics: dict[str, Any] = {}

    for split in config["input"]["splits"]:
        prediction_rows = read_jsonl(predictions_dir / f"{split}_predictions.jsonl")
        scored_rows, errors = score_predictions(prediction_rows, config["normalization"])
        split_metrics[split] = evaluate_split(scored_rows)
        split_metrics[split]["run_metadata"] = build_run_metadata(config, split)
        write_jsonl(metrics_dir / f"{split}_scored_predictions.jsonl", scored_rows)
        all_errors.extend(errors)

    metrics = {
        "experiment_name": config["experiment_name"],
        "prediction_dir": str(predictions_dir),
        "run_metadata": build_run_metadata(config, split="all"),
        "splits": split_metrics,
        "total_errors": len(all_errors),
    }
    write_json(metrics_dir / "metrics.json", metrics)
    write_jsonl(metrics_dir / "errors.jsonl", all_errors)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
