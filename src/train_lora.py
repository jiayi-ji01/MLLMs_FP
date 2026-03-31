"""Train one LoRA/QLoRA adapter for the main or control setting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from utils import ensure_dir, load_yaml, read_jsonl, write_json, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA or QLoRA adapter for English -> Fake transfer.")
    parser.add_argument("--config", default="configs/train_main.yaml")
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map a config string to a torch dtype."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def get_training_mode(config: dict[str, Any]) -> str:
    """Switch cleanly between standard LoRA and QLoRA."""
    mode = config.get("training_mode")
    if mode is None:
        mode = "qlora" if config.get("quantization", {}).get("enabled", False) else "lora"
    if mode not in {"lora", "qlora"}:
        raise ValueError(f"Unsupported training_mode: {mode}")
    return mode


def build_quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig | None:
    """Create a bitsandbytes config when QLoRA is enabled."""
    if get_training_mode(config) != "qlora":
        return None
    quant_cfg = config.get("quantization", {})
    if not quant_cfg.get("enabled", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=get_torch_dtype(quant_cfg["bnb_4bit_compute_dtype"]),
    )


def format_messages_as_training_text(
    tokenizer,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int], list[int]]:
    """Tokenize one chat example and mask non-assistant tokens in the labels."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Each training example must end with an assistant message.")

    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full_tokens = tokenizer(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)

    input_ids = list(full_tokens["input_ids"])
    attention_mask = list(full_tokens["attention_mask"])
    prompt_length = len(prompt_tokens["input_ids"])
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    if len(labels) != len(input_ids):
        raise ValueError("Tokenization mismatch while constructing labels.")

    return input_ids, attention_mask, labels


def build_train_dataset(rows: list[dict[str, Any]], tokenizer, max_seq_length: int) -> Dataset:
    """Convert JSONL rows into a tokenized Hugging Face Dataset."""
    encoded_rows: list[dict[str, Any]] = []

    for row in rows:
        input_ids, attention_mask, labels = format_messages_as_training_text(tokenizer, row["messages"])
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
            labels = labels[:max_seq_length]

        encoded_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "example_id": row["example_id"],
                "relation": row["relation"],
            }
        )

    return Dataset.from_list(encoded_rows)


@dataclass
class SupervisedDataCollator:
    """Pad tokenized examples for causal language modeling."""

    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]
        label_tensor = torch.full((len(labels), max_length), fill_value=-100, dtype=torch.long)
        for row_index, label_ids in enumerate(labels):
            label_tensor[row_index, : len(label_ids)] = torch.tensor(label_ids, dtype=torch.long)

        batch["labels"] = label_tensor
        return batch


def load_model_and_tokenizer(config: dict[str, Any]):
    """Load the tokenizer and base model, optionally with 4-bit quantization."""
    model_cfg = config["model"]
    training_cfg = config["training"]
    quantization_config = build_quantization_config(config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": get_torch_dtype(model_cfg["torch_dtype"]),
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
    }
    if quantization_config is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires CUDA, but no GPU is available.")
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **model_kwargs)
    model.config.use_cache = not training_cfg.get("gradient_checkpointing", False)

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        )

    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def build_training_arguments(config: dict[str, Any]) -> TrainingArguments:
    """Translate YAML config into Hugging Face TrainingArguments."""
    training_cfg = config["training"]
    output_dir = ensure_dir(training_cfg["output_dir"])
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        logging_steps=training_cfg["logging_steps"],
        save_strategy=training_cfg["save_strategy"],
        eval_strategy=training_cfg["eval_strategy"],
        bf16=training_cfg.get("bf16", False),
        fp16=training_cfg.get("fp16", False),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        optim=training_cfg.get("optim", "adamw_torch"),
        report_to=[],
        remove_unused_columns=False,
        save_total_limit=2,
        seed=config["seed"],
        data_seed=config["seed"],
    )


def get_train_language_mode(config: dict[str, Any]) -> str:
    """Return the configured train-language mode."""
    return config.get("data", {}).get("train_language_mode", "en_only")


def resolve_train_file(config: dict[str, Any]) -> Path:
    """Choose the training file for the configured train-language mode."""
    data_cfg = config["data"]
    train_language_mode = get_train_language_mode(config)
    train_files_by_mode = data_cfg.get("train_files_by_mode")

    if train_files_by_mode:
        if train_language_mode not in train_files_by_mode:
            raise ValueError(
                f"train_language_mode={train_language_mode!r} is missing from data.train_files_by_mode."
            )
        return Path(train_files_by_mode[train_language_mode])

    return Path(data_cfg["train_file"])


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    train_file = resolve_train_file(config)
    train_rows = read_jsonl(train_file)

    model, tokenizer = load_model_and_tokenizer(config)
    train_dataset = build_train_dataset(
        rows=train_rows,
        tokenizer=tokenizer,
        max_seq_length=config["training"]["max_seq_length"],
    )
    training_args = build_training_arguments(config)
    data_collator = SupervisedDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    output_dir = Path(training_args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    train_result = trainer.state.log_history
    metadata = {
        "experiment_name": config["experiment_name"],
        "base_model": config["model"]["name_or_path"],
        "training_mode": get_training_mode(config),
        "train_language_mode": get_train_language_mode(config),
        "train_file": str(train_file),
        "train_examples": len(train_rows),
        "output_dir": str(output_dir),
        "quantization_enabled": config.get("quantization", {}).get("enabled", False),
        "lora": config["lora"],
        "training": config["training"],
        "log_history": train_result,
    }
    write_json(output_dir / "training_summary.json", metadata)
    write_yaml(output_dir / "training_config_snapshot.yaml", config)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
