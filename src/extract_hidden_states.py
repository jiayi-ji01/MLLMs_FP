"""Extract the hidden states we need for similarity and patching."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from analysis_utils import (
    batched,
    build_tokenized_prompts,
    extract_position_indices,
    load_model_and_tokenizer,
    load_rows_by_split,
    render_prompt_text,
)
from utils import ensure_dir, load_yaml, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hidden states for English -> Fake analysis.")
    parser.add_argument("--config", default="configs/analysis_hidden_states.yaml")
    return parser.parse_args()


def stack_hidden_states(outputs_hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Convert a tuple of layer activations into one tensor."""
    return torch.stack([layer.detach().cpu() for layer in outputs_hidden_states], dim=0)


def pool_token_span(layer_stack: torch.Tensor, token_indices: list[int]) -> torch.Tensor:
    """Mean-pool a token span across all layers."""
    if not token_indices:
        hidden_dim = layer_stack.shape[-1]
        return torch.zeros((layer_stack.shape[0], hidden_dim), dtype=layer_stack.dtype)
    return layer_stack[:, token_indices, :].mean(dim=1)


def extract_split(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    batch_size: int,
    system_prompt: str,
) -> dict[str, Any]:
    """Run hidden-state extraction for one split."""
    device = model.device
    layer_indices: list[int] | None = None
    final_token_rows: list[torch.Tensor] = []
    subject_pool_rows: list[torch.Tensor] = []
    relation_pool_rows: list[torch.Tensor] = []
    subject_found_mask: list[bool] = []
    relation_found_mask: list[bool] = []
    metadata_rows: list[dict[str, Any]] = []

    for batch_rows in tqdm(batched(rows, batch_size), desc="Extracting", leave=False):
        prompt_texts = [render_prompt_text(tokenizer, row, system_prompt=system_prompt) for row in batch_rows]
        model_inputs, offset_mapping, attention_mask = build_tokenized_prompts(tokenizer, prompt_texts, device=device)

        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)

        hidden_stack = stack_hidden_states(outputs.hidden_states)
        if layer_indices is None:
            layer_indices = list(range(hidden_stack.shape[0]))

        for batch_index, row in enumerate(batch_rows):
            prompt_token_count = int(attention_mask[batch_index].sum().item())
            offsets = [tuple(item) for item in offset_mapping[batch_index][:prompt_token_count].tolist()]
            position_map = extract_position_indices(
                row=row,
                prompt_text=prompt_texts[batch_index],
                offsets=offsets,
                prompt_token_count=prompt_token_count,
            )

            example_stack = hidden_stack[:, batch_index, :prompt_token_count, :]
            final_index = int(position_map["final_token"])
            final_token_rows.append(example_stack[:, final_index, :].clone())

            subject_indices = list(position_map["subject_token_indices"] or [])
            relation_indices = list(position_map["relation_token_indices"] or [])
            subject_pool_rows.append(pool_token_span(example_stack, subject_indices))
            relation_pool_rows.append(pool_token_span(example_stack, relation_indices))
            subject_found_mask.append(bool(subject_indices))
            relation_found_mask.append(bool(relation_indices))

            metadata_rows.append(
                {
                    "example_id": row["example_id"],
                    "pair_id": row.get("pair_id"),
                    "split": row["split"],
                    "language": row["language"],
                    "relation": row["relation"],
                    "relation_alias": row["relation_alias"],
                    "subject": row["subject"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "prompt_token_count": prompt_token_count,
                    "final_token_index": final_index,
                    "subject_token_indices": subject_indices,
                    "relation_token_indices": relation_indices,
                }
            )

    if layer_indices is None:
        raise ValueError("No rows were provided for hidden-state extraction.")

    return {
        "layer_indices": torch.tensor(layer_indices, dtype=torch.long),
        "final_token": torch.stack(final_token_rows),
        "subject_pool": torch.stack(subject_pool_rows),
        "relation_pool": torch.stack(relation_pool_rows),
        "subject_found_mask": torch.tensor(subject_found_mask, dtype=torch.bool),
        "relation_found_mask": torch.tensor(relation_found_mask, dtype=torch.bool),
        "metadata": metadata_rows,
    }


def build_summary(config: dict[str, Any], split: str, hidden_state_bundle: dict[str, Any], output_file: Path) -> dict[str, Any]:
    """Create a lightweight JSON summary for one extraction run."""
    metadata_rows = hidden_state_bundle["metadata"]
    return {
        "experiment_name": config["experiment_name"],
        "split": split,
        "num_examples": len(metadata_rows),
        "languages": dict(Counter(item["language"] for item in metadata_rows)),
        "relations": dict(Counter(item["relation"] for item in metadata_rows)),
        "num_layers_saved": int(hidden_state_bundle["layer_indices"].numel()),
        "hidden_size": int(hidden_state_bundle["final_token"].shape[-1]),
        "subject_span_found_count": int(hidden_state_bundle["subject_found_mask"].sum().item()),
        "relation_span_found_count": int(hidden_state_bundle["relation_found_mask"].sum().item()),
        "output_file": str(output_file),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model, tokenizer = load_model_and_tokenizer(config)

    output_root = ensure_dir(Path(config["output"]["hidden_states_dir"]) / config["output"]["run_name"])
    summary: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "run_name": config["output"]["run_name"],
        "splits": {},
    }

    for split in config["data"]["splits"]:
        rows = load_rows_by_split(config["data"]["processed_dir"], split)
        bundle = extract_split(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            batch_size=config["analysis"]["batch_size"],
            system_prompt=config.get("sft", {}).get("system_prompt", ""),
        )
        output_file = output_root / f"{split}_hidden_states.pt"
        torch.save(bundle, output_file)
        split_summary = build_summary(config, split, bundle, output_file)
        write_json(output_root / f"{split}_hidden_states_summary.json", split_summary)
        summary["splits"][split] = split_summary

    write_json(output_root / "hidden_states_extraction_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
