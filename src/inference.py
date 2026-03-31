"""Run batch inference on English and Fake evaluation questions.

Role:
- load a base model plus a trained LoRA/QLoRA adapter
- generate answers for dev/test questions
- save predictions with full metadata for evaluation and error analysis

Input:
- YAML inference config
- processed split files from src/make_splits.py

Output:
- per-split prediction JSONL files
- inference summary JSON
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import ensure_dir, load_yaml, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference for English -> Fake transfer evaluation.")
    parser.add_argument("--config", default="configs/inference.yaml")
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def build_quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig | None:
    quant_cfg = config.get("quantization", {})
    if not quant_cfg.get("enabled", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=get_torch_dtype(quant_cfg["bnb_4bit_compute_dtype"]),
    )


def load_model_and_tokenizer(config: dict[str, Any]):
    """Load the base model, tokenizer, and LoRA adapter."""
    model_cfg = config["model"]
    quantization_config = build_quantization_config(config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": get_torch_dtype(model_cfg["torch_dtype"]),
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
    }
    if quantization_config is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA inference requires CUDA, but no GPU is available.")
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model_name_or_path"], **model_kwargs)
    model = PeftModel.from_pretrained(base_model, model_cfg["adapter_path"])
    model.eval()
    return model, tokenizer


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """Slice a list into fixed-size batches."""
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_chat_prompts(rows: list[dict[str, Any]], system_prompt: str) -> list[str]:
    """Create one-turn chat prompts for generation."""
    prompts: list[str] = []
    for row in rows:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": row["question"]})
        prompts.append(messages)
    return prompts


def generate_predictions(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    generation_cfg: dict[str, Any],
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Generate answers for a list of evaluation rows."""
    device = model.device
    all_predictions: list[dict[str, Any]] = []

    for batch_rows in tqdm(batched(rows, generation_cfg["batch_size"]), desc="Generating", leave=False):
        messages_batch = build_chat_prompts(batch_rows, system_prompt=system_prompt)
        prompt_texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        tokenized = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)
        generate_kwargs = {
            "max_new_tokens": generation_cfg["max_new_tokens"],
            "do_sample": generation_cfg["do_sample"],
            "num_beams": generation_cfg["num_beams"],
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if generation_cfg["do_sample"]:
            generate_kwargs["temperature"] = generation_cfg["temperature"]
            generate_kwargs["top_p"] = generation_cfg["top_p"]

        generated = model.generate(**tokenized, **generate_kwargs)

        input_length = tokenized["input_ids"].shape[1]
        generated_answers = tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=True)

        for row, prediction in zip(batch_rows, generated_answers):
            cleaned_prediction = prediction.strip()
            all_predictions.append(
                {
                    "example_id": row["example_id"],
                    "pair_id": row.get("pair_id"),
                    "split": row["split"],
                    "language": row["language"],
                    "relation": row["relation"],
                    "relation_alias": row["relation_alias"],
                    "subject": row["subject"],
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "prediction": cleaned_prediction,
                }
            )

    return all_predictions


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    processed_dir = config["data"]["processed_dir"]
    predictions_root = ensure_dir(Path(config["output"]["predictions_dir"]) / config["output"]["run_name"])
    model, tokenizer = load_model_and_tokenizer(config)

    summary = {
        "experiment_name": config["experiment_name"],
        "base_model_name_or_path": config["model"]["base_model_name_or_path"],
        "adapter_path": config["model"]["adapter_path"],
        "run_name": config["output"]["run_name"],
        "splits": {},
    }

    for split in config["data"]["splits"]:
        rows = read_jsonl(f"{processed_dir}/{split}.jsonl")
        predictions = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            generation_cfg=config["generation"],
            system_prompt=config.get("sft", {}).get("system_prompt", ""),
        )
        output_path = predictions_root / f"{split}_predictions.jsonl"
        write_jsonl(output_path, predictions)
        summary["splits"][split] = {
            "num_examples": len(predictions),
            "languages": dict(Counter(item["language"] for item in predictions)),
            "output_file": str(output_path),
        }

    write_json(predictions_root / "inference_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
