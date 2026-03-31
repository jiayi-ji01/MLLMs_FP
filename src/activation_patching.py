"""Patch English activations into Fake prompts and measure the rescue effect."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm.auto import tqdm

from analysis_utils import (
    extract_position_indices,
    get_first_answer_token_id,
    get_transformer_layers,
    group_rows_by_pair,
    load_model_and_tokenizer,
    load_rows_by_split,
    render_prompt_text,
    summarize_next_token_distribution,
)
from utils import ensure_dir, load_yaml, read_jsonl, write_json, write_jsonl


PUNCT_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layer-wise activation patching for English -> Fake transfer.")
    parser.add_argument("--config", default="configs/analysis_patching.yaml")
    parser.add_argument("--split", default=None, help="Optional split override.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Mirror evaluate.py normalization for selection logic."""
    normalized = text.strip().lower()
    normalized = PUNCT_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def read_prediction_outcomes(predictions_file: str | Path) -> dict[str, dict[str, bool]]:
    """Compute English/Fake correctness by pair id from prediction rows."""
    pair_outcomes: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in read_jsonl(predictions_file):
        pair_id = row.get("pair_id")
        if not pair_id:
            continue
        pair_outcomes[pair_id][row["language"]] = (
            normalize_text(row["prediction"]) == normalize_text(row["gold_answer"])
        )
    return pair_outcomes


def select_pairs(
    grouped_rows: dict[str, dict[str, dict[str, Any]]],
    pair_outcomes: dict[str, dict[str, bool]] | None,
    selection_cfg: dict[str, Any],
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    """Choose which English/Fake pairs to patch."""
    selected: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    mode = selection_cfg.get("mode", "all_pairs")
    relation_buckets: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)

    for pair_id, members in grouped_rows.items():
        if "en" not in members or "fake" not in members:
            continue

        include_pair = True
        if mode == "english_correct_fake_wrong":
            if pair_outcomes is None:
                include_pair = False
            else:
                include_pair = bool(pair_outcomes.get(pair_id, {}).get("en")) and not bool(
                    pair_outcomes.get(pair_id, {}).get("fake")
                )

        if include_pair:
            pair_tuple = (pair_id, members["en"], members["fake"])
            selected.append(pair_tuple)
            relation_buckets[members["fake"]["relation"]].append(pair_tuple)

    target_relations = selection_cfg.get("target_relations") or selection_cfg.get("relation_order") or sorted(relation_buckets)
    min_pairs_value = selection_cfg.get("min_pairs_per_relation", selection_cfg.get("per_relation_min_pairs", 0))
    min_pairs_per_relation = int(min_pairs_value or 0)
    per_relation_max_pairs = selection_cfg.get("per_relation_max_pairs")
    max_pairs_total = selection_cfg.get("max_pairs_total", selection_cfg.get("max_pairs"))
    warnings: list[str] = []

    if selection_cfg.get("stratify_by_relation", False) or target_relations:
        selected = []
        selected_ids: set[str] = set()

        def add_pair(pair_tuple: tuple[str, dict[str, Any], dict[str, Any]]) -> bool:
            if pair_tuple[0] in selected_ids:
                return False
            if max_pairs_total is not None and len(selected) >= int(max_pairs_total):
                return False
            selected.append(pair_tuple)
            selected_ids.add(pair_tuple[0])
            return True

        # We force a minimum per relation so symbol_is is always present.
        for relation in target_relations:
            bucket = relation_buckets.get(relation, [])
            if len(bucket) < min_pairs_per_relation:
                warnings.append(
                    f"Requested at least {min_pairs_per_relation} pairs for relation={relation}, "
                    f"but only found {len(bucket)} matching pairs."
                )
            for pair_tuple in bucket[:min_pairs_per_relation]:
                add_pair(pair_tuple)

        for relation in target_relations:
            bucket = relation_buckets.get(relation, [])
            limit = int(per_relation_max_pairs) if per_relation_max_pairs is not None else len(bucket)
            for pair_tuple in bucket[min_pairs_per_relation:limit]:
                if not add_pair(pair_tuple):
                    break

    if max_pairs_total:
        selected = selected[: int(max_pairs_total)]

    selection_summary = {
        "selection_mode": mode,
        "target_relations": list(target_relations),
        "min_pairs_per_relation": min_pairs_per_relation,
        "max_pairs_total": int(max_pairs_total) if max_pairs_total is not None else None,
        "available_pairs_by_relation": {relation: len(relation_buckets.get(relation, [])) for relation in target_relations},
        "selected_pairs_by_relation": dict(Counter(target_row["relation"] for _, _, target_row in selected)),
        "warnings": warnings,
    }
    return selected, selection_summary


def tokenize_single_prompt(tokenizer, prompt_text: str, device: torch.device):
    """Tokenize one prompt while keeping offset mapping for span lookup."""
    encoded = tokenizer(prompt_text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = [tuple(item) for item in encoded.pop("offset_mapping")[0].tolist()]
    model_inputs = {key: value.to(device) for key, value in encoded.items()}
    return model_inputs, offset_mapping


def make_patch_hook(target_index: int, source_vector: torch.Tensor) -> Callable:
    """Create a forward hook that replaces one token position in a layer output."""
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
            hidden_states[:, target_index, :] = source_vector
            return (hidden_states,) + output[1:]

        hidden_states = output.clone()
        hidden_states[:, target_index, :] = source_vector
        return hidden_states

    return hook


def summarize_clean_run(model, tokenizer, row: dict[str, Any], system_prompt: str, top_k: int) -> dict[str, Any]:
    """Run one clean prompt and return prompt metadata plus next-token stats."""
    device = model.device
    prompt_text = render_prompt_text(tokenizer, row, system_prompt=system_prompt)
    model_inputs, offset_mapping = tokenize_single_prompt(tokenizer, prompt_text, device=device)
    prompt_token_count = int(model_inputs["attention_mask"][0].sum().item())
    trimmed_offsets = offset_mapping[:prompt_token_count]
    position_map = extract_position_indices(
        row=row,
        prompt_text=prompt_text,
        offsets=trimmed_offsets,
        prompt_token_count=prompt_token_count,
    )

    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)

    gold_token_id = get_first_answer_token_id(tokenizer, row["answer"])
    next_token_stats = summarize_next_token_distribution(outputs.logits[0, -1, :], gold_token_id, tokenizer, top_k)
    return {
        "prompt_text": prompt_text,
        "model_inputs": model_inputs,
        "position_map": position_map,
        "outputs": outputs,
        "next_token": next_token_stats,
    }


def aggregate_patching_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize patching effects by position, layer, and relation."""
    by_position_layer: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    by_relation_position_layer: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row in rows:
        by_position_layer[row["position"]][row["layer_index"]].append(row)
        by_relation_position_layer[row["relation"]][row["position"]][row["layer_index"]].append(row)

    summary: dict[str, Any] = {"overall": {}, "per_relation": {}}
    for position, layer_map in by_position_layer.items():
        summary["overall"][position] = {}
        for layer_index, layer_rows in layer_map.items():
            summary["overall"][position][str(layer_index)] = {
                "count": len(layer_rows),
                "sample_count": len(layer_rows),
                "mean_gold_logit_delta": sum(row["gold_logit_delta"] for row in layer_rows) / len(layer_rows),
                "gold_logit_delta_mean": sum(row["gold_logit_delta"] for row in layer_rows) / len(layer_rows),
                "mean_gold_rank_delta": sum(row["gold_rank_delta"] for row in layer_rows) / len(layer_rows),
                "gold_rank_improvement_mean": sum(row["gold_rank_delta"] for row in layer_rows) / len(layer_rows),
                "rescue_rate_top1": sum(1 for row in layer_rows if row["patched_top1_is_gold"]) / len(layer_rows),
            }

    for relation, position_map in by_relation_position_layer.items():
        summary["per_relation"][relation] = {}
        for position, layer_map in position_map.items():
            summary["per_relation"][relation][position] = {}
            for layer_index, layer_rows in layer_map.items():
                summary["per_relation"][relation][position][str(layer_index)] = {
                    "count": len(layer_rows),
                    "sample_count": len(layer_rows),
                    "mean_gold_logit_delta": sum(row["gold_logit_delta"] for row in layer_rows) / len(layer_rows),
                    "gold_logit_delta_mean": sum(row["gold_logit_delta"] for row in layer_rows) / len(layer_rows),
                    "mean_gold_rank_delta": sum(row["gold_rank_delta"] for row in layer_rows) / len(layer_rows),
                    "gold_rank_improvement_mean": sum(row["gold_rank_delta"] for row in layer_rows) / len(layer_rows),
                    "rescue_rate_top1": sum(1 for row in layer_rows if row["patched_top1_is_gold"]) / len(layer_rows),
                }
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model, tokenizer = load_model_and_tokenizer(config)
    layers = get_transformer_layers(model)

    split = args.split or config["data"]["split"]
    output_base = Path(args.output_dir) if args.output_dir else Path(config["output"]["results_dir"]) / config["output"]["run_name"]
    output_root = ensure_dir(output_base)
    rows = load_rows_by_split(config["data"]["processed_dir"], split)
    grouped_rows = group_rows_by_pair(rows)

    predictions_file = config["input"].get("predictions_file", "")
    pair_outcomes = read_prediction_outcomes(predictions_file) if predictions_file else None
    selected_pairs, selection_summary = select_pairs(grouped_rows, pair_outcomes, config["selection"])

    patching_rows: list[dict[str, Any]] = []
    clean_pair_summaries: list[dict[str, Any]] = []
    system_prompt = config.get("sft", {}).get("system_prompt", "")
    top_k = config["analysis"]["top_k"]

    for pair_id, source_row, target_row in tqdm(selected_pairs, desc="Patching"):
        source_run = summarize_clean_run(model, tokenizer, source_row, system_prompt, top_k=top_k)
        target_run = summarize_clean_run(model, tokenizer, target_row, system_prompt, top_k=top_k)
        clean_pair_summaries.append(
            {
                "pair_id": pair_id,
                "relation": target_row["relation"],
                "source_language": "en",
                "target_language": "fake",
                "source_gold_rank": source_run["next_token"]["gold_rank"],
                "target_gold_rank": target_run["next_token"]["gold_rank"],
                "target_top1_is_gold": target_run["next_token"]["top_tokens"][0]["token_id"] == target_run["next_token"]["gold_token_id"],
            }
        )

        for position in config["analysis"]["positions"]:
            source_index = source_run["position_map"].get(position)
            target_index = target_run["position_map"].get(position)
            if source_index is None or target_index is None:
                continue

            for layer_index in config["analysis"].get("layers", list(range(len(layers)))):
                source_vector = source_run["outputs"].hidden_states[layer_index + 1][0, int(source_index), :].detach()
                hook = layers[layer_index].register_forward_hook(make_patch_hook(int(target_index), source_vector))
                with torch.no_grad():
                    patched_outputs = model(**target_run["model_inputs"], use_cache=False)
                hook.remove()

                patched_stats = summarize_next_token_distribution(
                    patched_outputs.logits[0, -1, :],
                    target_run["next_token"]["gold_token_id"],
                    tokenizer,
                    top_k=top_k,
                )
                patching_rows.append(
                    {
                        "pair_id": pair_id,
                        "relation": target_row["relation"],
                        "position": position,
                        "layer_index": layer_index,
                        "source_example_id": source_row["example_id"],
                        "target_example_id": target_row["example_id"],
                        "source_index": int(source_index),
                        "target_index": int(target_index),
                        "clean_gold_logit": target_run["next_token"]["gold_logit"],
                        "patched_gold_logit": patched_stats["gold_logit"],
                        "gold_logit_delta": patched_stats["gold_logit"] - target_run["next_token"]["gold_logit"],
                        "clean_gold_rank": target_run["next_token"]["gold_rank"],
                        "patched_gold_rank": patched_stats["gold_rank"],
                        "gold_rank_delta": target_run["next_token"]["gold_rank"] - patched_stats["gold_rank"],
                        "clean_top1_token": target_run["next_token"]["top_tokens"][0]["token_text"],
                        "patched_top1_token": patched_stats["top_tokens"][0]["token_text"],
                        "patched_top1_is_gold": patched_stats["top_tokens"][0]["token_id"] == patched_stats["gold_token_id"],
                    }
                )

    summary = {
        "experiment_name": config["experiment_name"],
        "run_name": config["output"]["run_name"],
        "split": split,
        "selection_mode": config["selection"]["mode"],
        "selected_pair_count": len(selected_pairs),
        "selected_relation_distribution": dict(Counter(target_row["relation"] for _, _, target_row in selected_pairs)),
        "positions": config["analysis"]["positions"],
        "layers_analyzed": config["analysis"].get("layers", list(range(len(layers)))),
        "aggregate": aggregate_patching_rows(patching_rows),
    }

    write_jsonl(output_root / "patching_clean_pairs.jsonl", clean_pair_summaries)
    write_jsonl(output_root / "patching_scores.jsonl", patching_rows)
    selection_summary.update(
        {
            "run_name": config["output"]["run_name"],
            "split": split,
            "selected_pair_count": len(selected_pairs),
            "selected_relation_distribution": summary["selected_relation_distribution"],
        }
    )
    write_json(output_root / "overall.json", summary["aggregate"]["overall"])
    write_json(output_root / "by_relation.json", summary["aggregate"]["per_relation"])
    write_json(output_root / "selection_summary.json", selection_summary)

    legacy_summary = {
        **summary,
        "num_selected_pairs": summary["selected_pair_count"],
        "selected_pairs_by_relation": summary["selected_relation_distribution"],
        "selection_summary": selection_summary,
    }
    write_json(output_root / "patching_summary.json", legacy_summary)
    print(json.dumps(legacy_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
