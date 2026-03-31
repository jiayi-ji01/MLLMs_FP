"""Turn the processed rows into chat-style SFT files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from typing import Any

from utils import ensure_dir, load_yaml, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format processed split files for supervised fine-tuning.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    return parser.parse_args()


def format_example(example: dict[str, Any], system_prompt: str = "") -> dict[str, Any]:
    """Convert one processed QA example into chat-style SFT format."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": example["question"]})
    messages.append({"role": "assistant", "content": example["answer"]})

    formatted = dict(example)
    formatted["messages"] = messages
    return formatted


def group_eval_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group evaluation examples by split and language for later inference."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["language"])].append(row)
    return grouped


def read_optional_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file when present, otherwise return an empty list."""
    try:
        return read_jsonl(path)
    except FileNotFoundError:
        return []


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    processed_dir = config["paths"]["processed_dir"]
    sft_dir = ensure_dir(f"{processed_dir}/sft")
    system_prompt = config.get("sft", {}).get("system_prompt", "")

    train_rows = read_jsonl(f"{processed_dir}/train.jsonl")
    train_fake_control_rows = read_optional_jsonl(f"{processed_dir}/train_fake_control.jsonl")
    dev_rows = read_jsonl(f"{processed_dir}/dev.jsonl")
    test_rows = read_jsonl(f"{processed_dir}/test.jsonl")

    train_en_only_sft_rows = [format_example(row, system_prompt=system_prompt) for row in train_rows]
    train_en_plus_fake_sft_rows = [
        format_example(row, system_prompt=system_prompt)
        for row in train_rows + train_fake_control_rows
    ]
    dev_sft_rows = [format_example(row, system_prompt=system_prompt) for row in dev_rows]
    test_sft_rows = [format_example(row, system_prompt=system_prompt) for row in test_rows]

    # The old train_sft name stays English-only so earlier commands still work.
    write_jsonl(f"{sft_dir}/train_sft.jsonl", train_en_only_sft_rows)
    write_jsonl(f"{sft_dir}/train_en_only_sft.jsonl", train_en_only_sft_rows)
    write_jsonl(f"{sft_dir}/train_en_plus_fake_sft.jsonl", train_en_plus_fake_sft_rows)

    for (split, language), rows in group_eval_rows(dev_rows + test_rows).items():
        formatted_rows = [format_example(row, system_prompt=system_prompt) for row in rows]
        write_jsonl(f"{sft_dir}/{split}_{language}_sft.jsonl", formatted_rows)

    summary = {
        "experiment_name": config["experiment_name"],
        "system_prompt_used": system_prompt,
        "train_sft_examples": len(train_en_only_sft_rows),
        "train_en_only_sft_examples": len(train_en_only_sft_rows),
        "train_en_plus_fake_sft_examples": len(train_en_plus_fake_sft_rows),
        "dev_sft_examples": len(dev_sft_rows),
        "test_sft_examples": len(test_sft_rows),
        "files": {
            "train_sft": f"{sft_dir}/train_sft.jsonl",
            "train_en_only_sft": f"{sft_dir}/train_en_only_sft.jsonl",
            "train_en_plus_fake_sft": f"{sft_dir}/train_en_plus_fake_sft.jsonl",
            "dev_en_sft": f"{sft_dir}/dev_en_sft.jsonl",
            "dev_fake_sft": f"{sft_dir}/dev_fake_sft.jsonl",
            "test_en_sft": f"{sft_dir}/test_en_sft.jsonl",
            "test_fake_sft": f"{sft_dir}/test_fake_sft.jsonl",
        },
        "notes": {
            "legacy_train_file_is_english_only": True,
            "fake_control_train_file_includes_english_and_fake": bool(train_fake_control_rows),
            "eval_files_are_split_by_language_for_cleaner_inference": True,
        },
    }
    write_json(f"{sft_dir}/sft_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
