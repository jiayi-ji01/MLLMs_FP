"""Compare English and Fake hidden states layer by layer."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from utils import ensure_dir, load_yaml, write_json, write_jsonl


POSITION_TO_MASK_KEY = {
    "final_token": None,
    "subject_pool": "subject_found_mask",
    "relation_pool": "relation_found_mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze layer-wise English/Fake hidden-state similarity.")
    parser.add_argument("--config", default="configs/analysis_similarity.yaml")
    parser.add_argument("--split", default=None, help="Optional single split override.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser.parse_args()


def to_metadata_index(metadata_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Map example ids to tensor row indices."""
    return {row["example_id"]: index for index, row in enumerate(metadata_rows)}


def compute_pairwise_similarity(bundle: dict[str, Any], positions: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute cosine similarity by layer for matched English/Fake pairs."""
    metadata_rows = bundle["metadata"]
    pair_groups: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(metadata_rows):
        pair_id = row.get("pair_id")
        if pair_id:
            pair_groups[pair_id][row["language"]] = index

    detailed_rows: list[dict[str, Any]] = []
    grouped_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    grouped_scores_by_relation: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    for pair_id, members in pair_groups.items():
        if "en" not in members or "fake" not in members:
            continue
        en_index = members["en"]
        fake_index = members["fake"]
        relation = metadata_rows[en_index]["relation"]

        for position in positions:
            mask_key = POSITION_TO_MASK_KEY[position]
            if mask_key is not None:
                if not bool(bundle[mask_key][en_index].item()) or not bool(bundle[mask_key][fake_index].item()):
                    continue

            en_tensor = bundle[position][en_index]
            fake_tensor = bundle[position][fake_index]
            similarities = F.cosine_similarity(en_tensor, fake_tensor, dim=-1)
            grouped_scores[position].append(similarities)
            grouped_scores_by_relation[position][relation].append(similarities)
            detailed_rows.append(
                {
                    "pair_id": pair_id,
                    "relation": relation,
                    "position": position,
                    "layer_similarity": [float(value) for value in similarities.tolist()],
                }
            )

    overall_summary: dict[str, Any] = {}
    for position, tensors in grouped_scores.items():
        stacked = torch.stack(tensors)
        overall_summary[position] = {
            "num_pairs": int(stacked.shape[0]),
            "mean_similarity_by_layer": [float(value) for value in stacked.mean(dim=0).tolist()],
        }

    relation_summary: dict[str, Any] = {}
    for position, relation_map in grouped_scores_by_relation.items():
        relation_summary[position] = {}
        for relation, tensors in relation_map.items():
            stacked = torch.stack(tensors)
            relation_summary[position][relation] = {
                "num_pairs": int(stacked.shape[0]),
                "mean_similarity_by_layer": [float(value) for value in stacked.mean(dim=0).tolist()],
            }

    return detailed_rows, {
        "overall": overall_summary,
        "per_relation": relation_summary,
    }


def compute_layer_band_summary(
    relation_summary: dict[str, Any],
    layer_bands: dict[str, list[int]],
) -> dict[str, Any]:
    """Average similarities inside broad early/middle/late layer bands."""
    summary: dict[str, Any] = {}
    for position, relation_map in relation_summary.items():
        summary[position] = {}
        for relation, payload in relation_map.items():
            values = payload["mean_similarity_by_layer"]
            summary[position][relation] = {}
            for band_name, layer_indices in layer_bands.items():
                valid_values = [
                    values[layer_index]
                    for layer_index in layer_indices
                    if 0 <= layer_index < len(values)
                ]
                summary[position][relation][band_name] = {
                    "num_layers": len(valid_values),
                    "mean_similarity": float(sum(valid_values) / len(valid_values)) if valid_values else 0.0,
                }
    return summary


def write_similarity_csv(
    output_path: Path,
    layer_indices: list[int],
    overall_summary: dict[str, Any],
    relation_summary: dict[str, Any],
) -> None:
    """Write a flat CSV for quick plotting."""
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "relation", "position", "layer", "mean_similarity"])
        writer.writeheader()

        for position, payload in overall_summary.items():
            for layer_index, value in zip(layer_indices, payload["mean_similarity_by_layer"]):
                writer.writerow(
                    {
                        "scope": "overall",
                        "relation": "all",
                        "position": position,
                        "layer": layer_index,
                        "mean_similarity": value,
                    }
                )

        for position, relation_map in relation_summary.items():
            for relation, payload in relation_map.items():
                for layer_index, value in zip(layer_indices, payload["mean_similarity_by_layer"]):
                    writer.writerow(
                        {
                            "scope": "per_relation",
                            "relation": relation,
                            "position": position,
                            "layer": layer_index,
                            "mean_similarity": value,
                        }
                    )


def build_layer_bands(config: dict[str, Any], layer_count: int) -> dict[str, list[int]]:
    """Resolve layer-band definitions from config or use the paper defaults."""
    configured = config["analysis"].get(
        "layer_bands",
        {
            "early": [0, 10],
            "middle": [11, 21],
            "late": [22, 32],
        },
    )
    layer_bands: dict[str, list[int]] = {}
    for band_name, bounds in configured.items():
        if len(bounds) != 2:
            raise ValueError(f"Layer band {band_name!r} must be [start, end].")
        start, end = int(bounds[0]), int(bounds[1])
        layer_bands[band_name] = [index for index in range(start, end + 1) if index < layer_count]
    return layer_bands


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    hidden_root = Path(config["input"]["hidden_states_dir"]) / config["input"]["run_name"]
    output_base = Path(args.output_dir) if args.output_dir else Path(config["output"]["results_dir"]) / config["output"]["run_name"]
    output_root = ensure_dir(output_base)
    splits = [args.split] if args.split else config["input"]["splits"]

    summary: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "run_name": config["output"]["run_name"],
        "splits": {},
    }

    for split in splits:
        bundle = torch.load(hidden_root / f"{split}_hidden_states.pt", map_location="cpu")
        detailed_rows, split_summary = compute_pairwise_similarity(bundle, config["analysis"]["positions"])
        layer_indices = [int(value) for value in bundle["layer_indices"].tolist()]
        layer_bands = build_layer_bands(config, len(layer_indices))
        band_summary = compute_layer_band_summary(split_summary["per_relation"], layer_bands)

        split_output_dir = ensure_dir(output_root / split)
        write_jsonl(split_output_dir / "pairwise_similarity.jsonl", detailed_rows)
        write_json(split_output_dir / "overall.json", split_summary["overall"])
        write_json(split_output_dir / "by_relation.json", split_summary["per_relation"])
        write_json(split_output_dir / "layer_band_summary.json", band_summary)

        # Backward-compatible legacy files.
        write_json(output_root / f"{split}_similarity_summary.json", split_summary)
        write_similarity_csv(
            split_output_dir / "similarity_by_layer.csv",
            layer_indices=layer_indices,
            overall_summary=split_summary["overall"],
            relation_summary=split_summary["per_relation"],
        )
        summary["splits"][split] = {
            "overall": split_summary["overall"],
            "per_relation": split_summary["per_relation"],
            "layer_band_summary": band_summary,
        }

    write_json(output_root / "hidden_state_similarity_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
