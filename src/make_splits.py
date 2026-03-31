"""Make the processed train/dev/test files used by the paper experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
from typing import Any

from utils import ensure_dir, load_yaml, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/dev/test splits for English -> Fake transfer.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    return parser.parse_args()


def choose_eval_fact_ids(
    facts: list[dict[str, Any]],
    dev_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Select dev/test evaluation fact ids with per-relation stratification."""
    if dev_fraction <= 0 or test_fraction <= 0:
        raise ValueError("dev_fraction and test_fraction must be positive.")
    if dev_fraction + test_fraction >= 1:
        raise ValueError("dev_fraction + test_fraction must be less than 1.0.")

    rng = random.Random(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for fact in facts:
        grouped[fact["relation"]].append(fact["fact_id"])

    dev_ids: set[str] = set()
    test_ids: set[str] = set()

    for relation, fact_ids in grouped.items():
        shuffled = list(fact_ids)
        rng.shuffle(shuffled)

        total = len(shuffled)
        dev_count = max(1, int(round(total * dev_fraction)))
        test_count = max(1, int(round(total * test_fraction)))
        if dev_count + test_count >= total:
            test_count = max(1, total - dev_count)

        dev_ids.update(shuffled[:dev_count])
        test_ids.update(shuffled[dev_count : dev_count + test_count])

    if dev_ids & test_ids:
        raise ValueError("Dev and test fact ids must be disjoint.")

    return dev_ids, test_ids


def attach_split(example: dict[str, Any], split: str) -> dict[str, Any]:
    """Copy an example and attach split-oriented metadata."""
    example_with_split = dict(example)
    example_with_split["split"] = split
    example_with_split["example_id"] = f"{split}__{example['language']}__{example['template_id']}__{example['fact_id']}"
    if split in {"dev", "test"}:
        example_with_split["pair_id"] = f"{split}__{example['template_id']}__{example['fact_id']}"
    return example_with_split


def build_train_rows(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the main English train split and the extra Fake-control rows."""
    # We keep the main setting strict: English only.
    # For the control we reuse the Fake prompts as extra training supervision.
    train_rows = [
        attach_split(example, "train")
        for example in candidates
        if example["template_family"] == "train" and example["language"] == "en"
    ]
    train_fake_control_rows = [
        attach_split(example, "train")
        for example in candidates
        if example["language"] == "fake"
    ]
    return train_rows, train_fake_control_rows


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    raw_dir = config["paths"]["raw_dir"]
    processed_dir = ensure_dir(config["paths"]["processed_dir"])

    facts = read_jsonl(f"{raw_dir}/facts.jsonl")
    candidates = read_jsonl(f"{raw_dir}/qa_candidates.jsonl")

    dev_ids, test_ids = choose_eval_fact_ids(
        facts=facts,
        dev_fraction=config["splits"]["dev_fraction"],
        test_fraction=config["splits"]["test_fraction"],
        seed=config["seed"],
    )

    train_rows, train_fake_control_rows = build_train_rows(candidates)
    dev_rows = [
        attach_split(example, "dev")
        for example in candidates
        if example["template_family"] == "eval" and example["fact_id"] in dev_ids
    ]
    test_rows = [
        attach_split(example, "test")
        for example in candidates
        if example["template_family"] == "eval" and example["fact_id"] in test_ids
    ]

    train_languages = Counter(row["language"] for row in train_rows)
    dev_languages = Counter(row["language"] for row in dev_rows)
    test_languages = Counter(row["language"] for row in test_rows)

    if set(train_languages) != {"en"}:
        raise ValueError("Train split must contain English examples only.")
    if set(dev_languages) != {"en", "fake"}:
        raise ValueError("Dev split must contain both English and Fake examples.")
    if set(test_languages) != {"en", "fake"}:
        raise ValueError("Test split must contain both English and Fake examples.")

    write_jsonl(f"{processed_dir}/train.jsonl", train_rows)
    write_jsonl(f"{processed_dir}/train_fake_control.jsonl", train_fake_control_rows)
    write_jsonl(f"{processed_dir}/dev.jsonl", dev_rows)
    write_jsonl(f"{processed_dir}/test.jsonl", test_rows)

    split_fact_ids = {
        "train_facts_taught_in_english": [fact["fact_id"] for fact in facts],
        "dev_eval_fact_ids": sorted(dev_ids),
        "test_eval_fact_ids": sorted(test_ids),
    }
    write_json(f"{processed_dir}/split_fact_ids.json", split_fact_ids)

    summary = {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "train_examples": len(train_rows),
        "train_fake_control_examples": len(train_fake_control_rows),
        "dev_examples": len(dev_rows),
        "test_examples": len(test_rows),
        "languages_by_split": {
            "train": dict(train_languages),
            "train_fake_control": dict(Counter(row["language"] for row in train_fake_control_rows)),
            "dev": dict(dev_languages),
            "test": dict(test_languages),
        },
        "eval_fact_counts": {
            "dev": len(dev_ids),
            "test": len(test_ids),
        },
        "relations_in_dev": dict(Counter(fact["relation"] for fact in facts if fact["fact_id"] in dev_ids)),
        "relations_in_test": dict(Counter(fact["relation"] for fact in facts if fact["fact_id"] in test_ids)),
        "design_notes": {
            "all_facts_are_taught_in_train": True,
            "dev_test_measure_cross_lingual_retrieval_not_unseen_fact_generalization": True,
            "english_eval_confirms_source_knowledge": True,
            "fake_eval_measures_transfer": True,
            "train_fake_control_exports_fake_rows_for_control_training": True,
        },
        "files": {
            "train": f"{processed_dir}/train.jsonl",
            "train_fake_control": f"{processed_dir}/train_fake_control.jsonl",
            "dev": f"{processed_dir}/dev.jsonl",
            "test": f"{processed_dir}/test.jsonl",
            "split_fact_ids": f"{processed_dir}/split_fact_ids.json",
        },
    }
    write_json(f"{processed_dir}/split_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
