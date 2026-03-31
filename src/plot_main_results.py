"""Make the small set of figures we actually use in the paper."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, load_yaml, write_json


RELATIONS = ["lives_in", "discovered", "symbol_is"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the paper figures for behavior and mechanism results.")
    parser.add_argument("--config", default="configs/plot_main.yaml")
    parser.add_argument("--split", default=None)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def relation_order(relation_map: dict[str, Any]) -> list[str]:
    ordered = [relation for relation in RELATIONS if relation in relation_map]
    for relation in relation_map:
        if relation not in ordered:
            ordered.append(relation)
    return ordered


def build_behavior_comparison(main_metrics: dict[str, Any], fake_control_metrics: dict[str, Any], split: str) -> dict[str, Any]:
    main_split = main_metrics["splits"][split]
    control_split = fake_control_metrics["splits"][split]
    comparison = {
        "split": split,
        "main": {
            "source_accuracy": main_split["source_accuracy"],
            "fake_accuracy": main_split["fake_transfer_accuracy"],
            "transfer_efficiency": main_split["transfer"]["transfer_efficiency"],
            "per_relation": main_split["per_relation_accuracy"],
            "run_metadata": main_split.get("run_metadata", main_metrics.get("run_metadata", {})),
        },
        "fake_control": {
            "fake_accuracy": control_split["fake_transfer_accuracy"],
            "per_relation_fake": control_split["per_relation_accuracy"]["fake"],
            "run_metadata": control_split.get("run_metadata", fake_control_metrics.get("run_metadata", {})),
        },
    }
    return comparison


def write_behavior_comparison_csv(path: Path, comparison: dict[str, Any]) -> None:
    rows = []
    for relation in relation_order({
        **comparison["main"]["per_relation"]["fake"],
        **comparison["fake_control"]["per_relation_fake"],
    }):
        rows.append(
            {
                "relation": relation,
                "main_english_accuracy": comparison["main"]["per_relation"]["en"].get(relation, {}).get("accuracy", 0.0),
                "main_fake_accuracy": comparison["main"]["per_relation"]["fake"].get(relation, {}).get("accuracy", 0.0),
                "fake_control_fake_accuracy": comparison["fake_control"]["per_relation_fake"].get(relation, {}).get("accuracy", 0.0),
            }
        )

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["relation"])
        writer.writeheader()
        writer.writerows(rows)


def plot_behavior_overview(main_metrics: dict[str, Any], split: str, output_dir: Path) -> None:
    split_metrics = main_metrics["splits"][split]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))

    labels = ["English", "Fake", "Transfer eff."]
    values = [
        split_metrics["source_accuracy"],
        split_metrics["fake_transfer_accuracy"],
        split_metrics["transfer"]["transfer_efficiency"],
    ]
    axes[0].bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Accuracy / efficiency")
    axes[0].set_title("Panel A")
    axes[0].grid(axis="y", alpha=0.2)

    en_map = split_metrics["per_relation_accuracy"]["en"]
    fake_map = split_metrics["per_relation_accuracy"]["fake"]
    relations = relation_order({**en_map, **fake_map})
    x = np.arange(len(relations))
    width = 0.35
    axes[1].bar(x - width / 2, [en_map[r]["accuracy"] for r in relations], width=width, label="English", color="#4C78A8")
    axes[1].bar(x + width / 2, [fake_map[r]["accuracy"] for r in relations], width=width, label="Fake", color="#F58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(relations)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Panel B")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Behavior overview", fontsize=13)
    save_figure(fig, output_dir / "behavior_overview")


def plot_fake_control_comparison(comparison: dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))

    axes[0].bar(
        ["Main", "Fake control"],
        [comparison["main"]["fake_accuracy"], comparison["fake_control"]["fake_accuracy"]],
        color=["#F58518", "#72B7B2"],
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Fake accuracy")
    axes[0].set_title("Overall")
    axes[0].grid(axis="y", alpha=0.2)

    relations = relation_order({
        **comparison["main"]["per_relation"]["fake"],
        **comparison["fake_control"]["per_relation_fake"],
    })
    x = np.arange(len(relations))
    width = 0.35
    main_fake = [comparison["main"]["per_relation"]["fake"].get(r, {}).get("accuracy", 0.0) for r in relations]
    control_fake = [comparison["fake_control"]["per_relation_fake"].get(r, {}).get("accuracy", 0.0) for r in relations]
    axes[1].bar(x - width / 2, main_fake, width=width, label="Main", color="#F58518")
    axes[1].bar(x + width / 2, control_fake, width=width, label="Fake control", color="#72B7B2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(relations)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Fake accuracy")
    axes[1].set_title("By relation")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Fake-comprehension control", fontsize=13)
    save_figure(fig, output_dir / "fake_control_comparison")


def load_similarity_summary(root: Path, split: str) -> dict[str, Any]:
    split_dir = root / split
    return {
        "overall": read_json(split_dir / "overall.json"),
        "per_relation": read_json(split_dir / "by_relation.json"),
        "layer_bands": read_json(split_dir / "layer_band_summary.json"),
    }


def plot_similarity_main(summary: dict[str, Any], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for position, color in [("subject_pool", "#4C78A8"), ("relation_pool", "#F58518")]:
        values = summary["overall"][position]["mean_similarity_by_layer"]
        ax.plot(range(len(values)), values, linewidth=2, label=position, color=color)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Similarity by layer")
    save_figure(fig, output_dir / "similarity_overall_main")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    for axis, relation in zip(axes, RELATIONS):
        for position, color in [("subject_pool", "#4C78A8"), ("relation_pool", "#F58518")]:
            values = summary["per_relation"][position][relation]["mean_similarity_by_layer"]
            axis.plot(range(len(values)), values, linewidth=2, label=position, color=color)
        axis.set_title(relation)
        axis.set_xlabel("Layer")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Cosine similarity")
    axes[-1].legend(frameon=False, loc="lower left")
    fig.suptitle("Similarity by relation", fontsize=13)
    save_figure(fig, output_dir / "similarity_by_relation_main")


def load_patching_summary(root: Path) -> dict[str, Any]:
    return {
        "overall": read_json(root / "overall.json"),
        "by_relation": read_json(root / "by_relation.json"),
        "selection": read_json(root / "selection_summary.json"),
    }


def plot_patching_main(summary: dict[str, Any], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for position, color, label in [
        ("subject_last_token", "#4C78A8", "subject token"),
        ("relation_last_token", "#F58518", "relation token"),
    ]:
        layer_map = summary["overall"][position]
        layers = sorted(int(layer) for layer in layer_map)
        values = [layer_map[str(layer)]["rescue_rate_top1"] for layer in layers]
        ax.plot(layers, values, linewidth=2, label=label, color=color)
    ax.set_xlabel("Patched layer")
    ax.set_ylabel("Top-1 rescue rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Activation patching")
    save_figure(fig, output_dir / "patching_overall_main")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    for axis, relation in zip(axes, RELATIONS):
        layer_map = summary["by_relation"].get(relation, {}).get("relation_last_token", {})
        layers = sorted(int(layer) for layer in layer_map)
        values = [layer_map[str(layer)]["rescue_rate_top1"] for layer in layers]
        axis.plot(layers, values, linewidth=2, color="#F58518")
        axis.set_title(relation)
        axis.set_xlabel("Patched layer")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Top-1 rescue rate")
    fig.suptitle("Relation-token patching by relation", fontsize=13)
    save_figure(fig, output_dir / "patching_by_relation_main")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    split = args.split or config["input"].get("split", "test")
    output_dir = ensure_dir(config["output"]["output_dir"])

    generated: list[str] = []

    main_metrics = read_json(config["input"]["main_metrics_file"])
    fake_control_metrics = read_json(config["input"]["fake_control_metrics_file"])
    comparison = build_behavior_comparison(main_metrics, fake_control_metrics, split)
    write_json(output_dir / "behavior_comparison.json", comparison)
    write_behavior_comparison_csv(output_dir / "behavior_comparison.csv", comparison)
    plot_behavior_overview(main_metrics, split, output_dir)
    plot_fake_control_comparison(comparison, output_dir)
    generated.extend(
        [
            str(output_dir / "behavior_overview.png"),
            str(output_dir / "fake_control_comparison.png"),
        ]
    )

    similarity = load_similarity_summary(Path(config["input"]["similarity_dir"]) / config["input"]["run_name"], split)
    plot_similarity_main(similarity, output_dir)
    generated.extend(
        [
            str(output_dir / "similarity_overall_main.png"),
            str(output_dir / "similarity_by_relation_main.png"),
        ]
    )

    patching = load_patching_summary(Path(config["input"]["patching_dir"]) / config["input"]["run_name"])
    plot_patching_main(patching, output_dir)
    generated.extend(
        [
            str(output_dir / "patching_overall_main.png"),
            str(output_dir / "patching_by_relation_main.png"),
        ]
    )

    with (output_dir / "plot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"split": split, "generated": generated}, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
