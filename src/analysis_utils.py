"""Small shared helpers for the internal-analysis scripts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import read_jsonl


RELATION_ANCHOR_CANDIDATES = {
    "lives_in": ["live", "city", "home", "based", "place"],
    "discovered": ["discover", "discovery", "object", "item"],
    "symbol_is": ["symbol"],
}


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map config strings to torch dtypes."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def build_quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig | None:
    """Create a bitsandbytes config when QLoRA-style loading is enabled."""
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
    """Load tokenizer plus base model and optional LoRA adapter."""
    model_cfg = config["model"]
    quantization_config = build_quantization_config(config)

    base_model_name = (
        model_cfg.get("base_model_name_or_path")
        or model_cfg.get("name_or_path")
    )
    if not base_model_name:
        raise ValueError("Model config must provide base_model_name_or_path or name_or_path.")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_cfg.get("padding_side", "left")

    model_kwargs: dict[str, Any] = {
        "dtype": get_torch_dtype(model_cfg["torch_dtype"]),
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
    }
    if quantization_config is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Quantized analysis requires CUDA, but no GPU is available.")
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = model_cfg.get("device_map", "auto")

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    adapter_path = model_cfg.get("adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Slice a list into fixed-size batches."""
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_prompt_messages(row: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    """Create a one-turn prompt from a processed evaluation row."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": row["question"]})
    return messages


def render_prompt_text(tokenizer, row: dict[str, Any], system_prompt: str) -> str:
    """Render one evaluation row into a generation prompt."""
    messages = build_prompt_messages(row, system_prompt=system_prompt)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def group_rows_by_pair(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Group evaluation rows into English/Fake pairs."""
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id:
            grouped[pair_id][row["language"]] = row
    return grouped


def load_rows_by_split(processed_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Load a processed split JSONL file."""
    return read_jsonl(Path(processed_dir) / f"{split}.jsonl")


def get_transformer_layers(model) -> Any:
    """Return the block list for common causal LM architectures."""
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidate_paths = [
        ("model", "layers"),
        ("gpt_neox", "layers"),
        ("transformer", "h"),
        ("model", "decoder", "layers"),
    ]

    for path in candidate_paths:
        current = base_model
        found = True
        for attr in path:
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found:
            return current

    raise ValueError("Unsupported model architecture: unable to locate transformer layers.")


def build_tokenized_prompts(tokenizer, prompt_texts: list[str], device: torch.device):
    """Tokenize prompts for both analysis metadata and model forward passes."""
    encoded = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        return_offsets_mapping=True,
    )
    offset_mapping = encoded.pop("offset_mapping")
    model_inputs = {key: value.to(device) for key, value in encoded.items()}
    return model_inputs, offset_mapping, encoded["attention_mask"]


def token_indices_for_char_span(
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> list[int]:
    """Map a character span in the prompt text to token indices."""
    indices: list[int] = []
    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        overlaps = not (end <= char_start or start >= char_end)
        if overlaps:
            indices.append(token_index)
    return indices


def find_question_span(prompt_text: str, question: str) -> tuple[int, int] | None:
    """Locate the question inside a chat-formatted prompt."""
    start = prompt_text.rfind(question)
    if start < 0:
        return None
    return start, start + len(question)


def find_subject_span_in_prompt(prompt_text: str, question: str, subject: str) -> tuple[int, int] | None:
    """Locate the subject mention within the question span."""
    question_span = find_question_span(prompt_text, question)
    if question_span is None:
        return None
    question_start, _ = question_span
    lowered_question = question.lower()
    subject_start = lowered_question.find(subject.lower())
    if subject_start < 0:
        return None
    return question_start + subject_start, question_start + subject_start + len(subject)


def find_relation_anchor_text(row: dict[str, Any]) -> str | None:
    """Choose a coarse relation cue string to trace in the prompt."""
    if row["language"] == "fake":
        return row["relation_alias"]

    # We only need a rough anchor here. The goal is to compare where the
    # question asks about the relation, not to perfectly parse the sentence.
    question_lower = row["question"].lower()
    for candidate in RELATION_ANCHOR_CANDIDATES.get(row["relation"], []):
        if candidate in question_lower:
            return candidate
    return None


def find_relation_span_in_prompt(prompt_text: str, question: str, row: dict[str, Any]) -> tuple[int, int] | None:
    """Locate a relation cue span within the question span."""
    anchor_text = find_relation_anchor_text(row)
    if not anchor_text:
        return None

    question_span = find_question_span(prompt_text, question)
    if question_span is None:
        return None
    question_start, _ = question_span

    lowered_question = question.lower()
    anchor_start = lowered_question.find(anchor_text.lower())
    if anchor_start < 0:
        return None
    return question_start + anchor_start, question_start + anchor_start + len(anchor_text)


def extract_position_indices(
    row: dict[str, Any],
    prompt_text: str,
    offsets: list[tuple[int, int]],
    prompt_token_count: int,
) -> dict[str, int | list[int] | None]:
    """Pick the token positions used in the paper analyses."""
    final_token_index = max(prompt_token_count - 1, 0)

    subject_char_span = find_subject_span_in_prompt(prompt_text, row["question"], row["subject"])
    subject_token_indices = (
        token_indices_for_char_span(offsets, subject_char_span[0], subject_char_span[1])
        if subject_char_span is not None
        else []
    )

    relation_char_span = find_relation_span_in_prompt(prompt_text, row["question"], row)
    relation_token_indices = (
        token_indices_for_char_span(offsets, relation_char_span[0], relation_char_span[1])
        if relation_char_span is not None
        else []
    )

    return {
        "final_token": final_token_index,
        "subject_token_indices": subject_token_indices,
        "relation_token_indices": relation_token_indices,
        "subject_last_token": subject_token_indices[-1] if subject_token_indices else None,
        "relation_last_token": relation_token_indices[-1] if relation_token_indices else None,
    }


def get_first_answer_token_id(tokenizer, answer: str) -> int:
    """Return the first token id of a gold answer."""
    token_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError(f"Gold answer produced no tokens: {answer!r}")
    return int(token_ids[0])


def summarize_next_token_distribution(logits: torch.Tensor, gold_token_id: int, tokenizer, top_k: int) -> dict[str, Any]:
    """Summarize the next-token distribution at the answer position."""
    probs = torch.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(logits, k=top_k)
    gold_logit = float(logits[gold_token_id].item())
    gold_probability = float(probs[gold_token_id].item())
    gold_rank = int((logits > logits[gold_token_id]).sum().item()) + 1

    return {
        "gold_token_id": gold_token_id,
        "gold_token_text": tokenizer.decode([gold_token_id]),
        "gold_logit": gold_logit,
        "gold_probability": gold_probability,
        "gold_rank": gold_rank,
        "top_tokens": [
            {
                "token_id": int(token_id),
                "token_text": tokenizer.decode([int(token_id)]),
                "logit": float(logit_value),
            }
            for token_id, logit_value in zip(top_indices.tolist(), top_values.tolist())
        ],
    }
