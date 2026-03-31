"""Build the synthetic English/Fake dataset once from a YAML config."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
from collections import Counter
import json
import random
from typing import Any

from fake_language import FakeLanguage, FakeLanguageConfig
from utils import ensure_dir, load_yaml, resolve_path, write_json, write_jsonl, write_yaml


NAME_SYLLABLES = [
    "lo",
    "mar",
    "ta",
    "vik",
    "resh",
    "ka",
    "na",
    "reb",
    "so",
    "lite",
    "fa",
    "ren",
    "do",
    "mir",
    "xel",
    "qua",
    "tor",
    "bel",
    "rin",
    "sar",
    "vel",
    "mon",
    "dra",
    "zen",
    "phi",
    "cor",
    "lun",
    "yar",
]

CITY_SUFFIXES = ["eb", "or", "ar", "en", "un", "is", "on", "el"]
DISCOVERY_SUFFIXES = ["ite", "ium", "on", "ene", "al", "or"]
SYMBOL_COLORS = [
    "Blue",
    "Red",
    "Silver",
    "Gold",
    "Green",
    "Black",
    "White",
    "Amber",
    "Violet",
    "Bronze",
    "Teal",
    "Pearl",
    "Crimson",
]
SYMBOL_SHAPES = ["Triangle", "Circle", "Square", "Spiral", "Hexagon", "Arrow", "Crescent", "Diamond"]
COLOR_STEMS = ["Azure", "Crimson", "Ivory", "Jade", "Cobalt", "Saffron", "Onyx", "Lilac"]
COLOR_MARKERS = ["Arc", "Bloom", "Shard", "Flare", "Mist", "Pulse", "Ray", "Wave"]
ARCHIVE_PREFIXES = ["ARC", "NEX", "SOL", "VAR", "KET", "LUM"]


@dataclass(frozen=True)
class Fact:
    """A single synthetic fact triple."""

    fact_id: str
    subject_id: str
    subject: str
    relation: str
    relation_alias: str
    object: str
    answer: str
    statement_en: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw synthetic data for English -> Fake transfer.")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    return parser.parse_args()


def slugify(text: str) -> str:
    """Convert free text into a stable identifier fragment."""
    lowered = "".join(char.lower() if char.isalnum() else "-" for char in text)
    return "-".join(part for part in lowered.split("-") if part)


def make_unique_names(num_names: int, rng: random.Random, min_parts: int = 2, max_parts: int = 3) -> list[str]:
    """Generate unique pseudo-names such as Lomar or Reshka."""
    names: list[str] = []
    seen: set[str] = set()

    while len(names) < num_names:
        part_count = rng.randint(min_parts, max_parts)
        token = "".join(rng.choice(NAME_SYLLABLES) for _ in range(part_count))
        candidate = token.capitalize()
        if candidate not in seen:
            seen.add(candidate)
            names.append(candidate)

    return names


def build_subjects(num_subjects: int, rng: random.Random) -> list[str]:
    """Create synthetic subject entities."""
    return make_unique_names(num_subjects, rng=rng, min_parts=2, max_parts=3)


def build_city_name(index: int, rng: random.Random) -> str:
    """Create a synthetic city name."""
    base = make_unique_names(index + 5, rng=random.Random(rng.randint(0, 10_000) + index), min_parts=2, max_parts=2)[-1]
    return f"{base}{CITY_SUFFIXES[index % len(CITY_SUFFIXES)]}"


def build_discovery_name(index: int, rng: random.Random) -> str:
    """Create a synthetic discovery/item name."""
    base = make_unique_names(index + 7, rng=random.Random(20_000 + index), min_parts=2, max_parts=2)[-1]
    return f"{base}{DISCOVERY_SUFFIXES[index % len(DISCOVERY_SUFFIXES)]}"


def build_symbol_name(index: int) -> str:
    """Create a symbolic synthetic label."""
    color = SYMBOL_COLORS[index % len(SYMBOL_COLORS)]
    shape = SYMBOL_SHAPES[(index // len(SYMBOL_COLORS)) % len(SYMBOL_SHAPES)]
    return f"{color.lower()} {shape.lower()}"


def build_color_name(index: int) -> str:
    """Create a synthetic color label."""
    stem = COLOR_STEMS[index % len(COLOR_STEMS)]
    marker = COLOR_MARKERS[(index // len(COLOR_STEMS)) % len(COLOR_MARKERS)]
    return f"{stem}{marker}"


def build_archive_code(index: int) -> str:
    """Create a stable synthetic archive code."""
    prefix = ARCHIVE_PREFIXES[index % len(ARCHIVE_PREFIXES)]
    return f"{prefix}-{100 + index:03d}-{chr(65 + (index % 26))}"


def build_object_value(
    object_type: str,
    index: int,
    subject: str,
    subjects: list[str],
    rng: random.Random,
) -> str:
    """Generate the answer/object for a relation."""
    if object_type == "city":
        return build_city_name(index=index, rng=rng)
    if object_type == "discovery":
        return build_discovery_name(index=index, rng=rng)
    if object_type == "symbol":
        return build_symbol_name(index=index)
    if object_type == "color":
        return build_color_name(index=index)
    if object_type == "archive_code":
        return build_archive_code(index=index)
    if object_type == "mentor":
        subject_index = subjects.index(subject)
        mentor_index = (subject_index + 1 + (index % (len(subjects) - 1))) % len(subjects)
        return subjects[mentor_index]
    raise ValueError(f"Unsupported object_type: {object_type}")


def build_facts(config: dict[str, Any], rng: random.Random) -> list[Fact]:
    """Create the synthetic fact table from the config."""
    num_subjects = config["dataset"]["num_subjects"]
    relations = config["relations"]
    subjects = build_subjects(num_subjects=num_subjects, rng=rng)
    facts: list[Fact] = []

    for relation_offset, relation_cfg in enumerate(relations):
        relation_key = relation_cfg["key"]
        relation_alias = relation_cfg["fake_alias"]
        object_type = relation_cfg["object_type"]
        statement_template = relation_cfg["english_fact_template"]

        for subject_index, subject in enumerate(subjects):
            value_index = relation_offset * len(subjects) + subject_index
            object_value = build_object_value(
                object_type=object_type,
                index=value_index,
                subject=subject,
                subjects=subjects,
                rng=rng,
            )
            facts.append(
                Fact(
                    fact_id=f"{relation_key}__{slugify(subject)}",
                    subject_id=f"subject_{subject_index:04d}",
                    subject=subject,
                    relation=relation_key,
                    relation_alias=relation_alias,
                    object=object_value,
                    answer=object_value,
                    statement_en=statement_template.format(subject=subject, object=object_value),
                )
            )

    return facts


def build_candidate_example(
    fact: Fact,
    relation_cfg: dict[str, Any],
    template: str,
    template_family: str,
    template_index: int,
    language: str,
    fake_language: FakeLanguage,
) -> dict[str, Any]:
    """Render one candidate QA example before split assignment."""
    if language == "fake":
        question = fake_language.render_question(template, relation_key=fact.relation, subject=fact.subject)
    else:
        question = template.format(subject=fact.subject)

    template_id = f"{fact.relation}_{template_family}_{template_index}"
    return {
        "candidate_id": f"{template_family}__{language}__{template_id}__{fact.fact_id}",
        "fact_id": fact.fact_id,
        "subject_id": fact.subject_id,
        "subject": fact.subject,
        "relation": fact.relation,
        "relation_alias": relation_cfg["fake_alias"],
        "object": fact.object,
        "answer": fact.answer,
        "statement_en": fact.statement_en,
        "language": language,
        "template_family": template_family,
        "template_id": template_id,
        "question": question,
    }


def build_candidates(config: dict[str, Any], facts: list[Fact], fake_language: FakeLanguage) -> list[dict[str, Any]]:
    """Render all candidate QA examples prior to train/dev/test assignment."""
    relation_lookup = {relation_cfg["key"]: relation_cfg for relation_cfg in config["relations"]}
    candidates: list[dict[str, Any]] = []

    for fact in facts:
        relation_cfg = relation_lookup[fact.relation]
        for template_index, template in enumerate(relation_cfg["english_train_templates"]):
            candidates.append(
                build_candidate_example(
                    fact=fact,
                    relation_cfg=relation_cfg,
                    template=template,
                    template_family="train",
                    template_index=template_index,
                    language="en",
                    fake_language=fake_language,
                )
            )

        for template_index, template in enumerate(relation_cfg["english_eval_templates"]):
            candidates.append(
                build_candidate_example(
                    fact=fact,
                    relation_cfg=relation_cfg,
                    template=template,
                    template_family="eval",
                    template_index=template_index,
                    language="en",
                    fake_language=fake_language,
                )
            )

        for template_index, template in enumerate(relation_cfg["fake_eval_templates"]):
            candidates.append(
                build_candidate_example(
                    fact=fact,
                    relation_cfg=relation_cfg,
                    template=template,
                    template_family="eval",
                    template_index=template_index,
                    language="fake",
                    fake_language=fake_language,
                )
            )

    return candidates


def make_fake_language(config: dict[str, Any]) -> FakeLanguage:
    """Construct the fake-language renderer from the YAML config."""
    relation_aliases = {relation_cfg["key"]: relation_cfg["fake_alias"] for relation_cfg in config["relations"]}
    fake_language_config = FakeLanguageConfig.from_dict(
        fake_language_cfg=config.get("fake_language"),
        relation_aliases=relation_aliases,
    )
    return FakeLanguage(fake_language_config)


def build_summary(raw_dir: Path, facts: list[Fact], candidates: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Create a compact manifest for reproducibility."""
    fake_language = make_fake_language(config)
    return {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "num_subjects": config["dataset"]["num_subjects"],
        "num_relations": len(config["relations"]),
        "num_facts": len(facts),
        "num_candidates": len(candidates),
        "candidate_counts_by_language": dict(Counter(example["language"] for example in candidates)),
        "candidate_counts_by_template_family": dict(Counter(example["template_family"] for example in candidates)),
        "facts_per_relation": dict(Counter(fact.relation for fact in facts)),
        "raw_dir": str(raw_dir),
        "files": {
            "facts": str(raw_dir / "facts.jsonl"),
            "qa_candidates": str(raw_dir / "qa_candidates.jsonl"),
            "config_snapshot": str(raw_dir / "config_snapshot.yaml"),
        },
        "sample_subject": facts[0].subject if facts else None,
        "sample_fake_prompt": fake_language.render_question(
            config["relations"][0]["fake_eval_templates"][0],
            relation_key=config["relations"][0]["key"],
            subject=facts[0].subject if facts else "Lomar",
        ),
        "notes": {
            "train_candidates_are_english_only": True,
            "eval_candidates_include_english_and_fake": True,
            "split_assignment_happens_in": "src/make_splits.py",
        },
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    raw_dir = ensure_dir(config["paths"]["raw_dir"])
    rng = random.Random(config["seed"])

    facts = build_facts(config=config, rng=rng)
    fake_language = make_fake_language(config)
    candidates = build_candidates(config=config, facts=facts, fake_language=fake_language)

    write_jsonl(raw_dir / "facts.jsonl", (asdict(fact) for fact in facts))
    write_jsonl(raw_dir / "qa_candidates.jsonl", candidates)
    write_yaml(raw_dir / "config_snapshot.yaml", config)

    summary = build_summary(raw_dir=raw_dir, facts=facts, candidates=candidates, config=config)
    write_json(raw_dir / "build_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
