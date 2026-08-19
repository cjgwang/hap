"""
Plot results: three charts summarizing the collected dataset and
classifier performance, built entirely from files other scripts already
produce -- experiments/preprocess.py's episodes_index.csv and
experiments/evaluate.py's comparison_report.json. No new data sources.

    python experiments/plot_results.py

Writes three PNGs to results/:
  plot_episode_distribution.png   -- episode count per scenario family, colored by label
  plot_classifier_comparison.png  -- accuracy/precision/recall/F1 for each classifier
  plot_per_family_accuracy.png    -- accuracy per scenario family, one bar per classifier

Plain matplotlib, no seaborn/plotly, so it's easy to read end-to-end and
replicate/modify (e.g. in a notebook) without pulling in extra dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed color assignments, consistent across every chart -- color always
# means the same thing everywhere it appears (label OR classifier, never
# both at once in one figure).
LABEL_COLORS = {"ordinary": "#2a78d6", "adversarial": "#eb6834"}
CLASSIFIER_COLORS = {"text": "#2a78d6", "nvml": "#eb6834", "combined": "#1baf7a", "api_baseline": "#eda100"}
METRIC_COLORS = {"accuracy": "#2a78d6", "precision": "#eb6834", "recall": "#1baf7a", "f1": "#eda100"}

# A quiet, minimal chart style: hairline recessive gridlines, no top/right
# border, muted axis text -- so the bars (the data) stay the loudest thing
# on the page.
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def _strip_spines(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_episode_distribution(index_csv: Path, out_path: Path) -> None:
    """One bar per scenario family, colored by its (deterministic) label --
    shows both the family breakdown and the ordinary/adversarial balance
    in one chart, since every family is entirely one label or the other.
    """
    df = pd.read_csv(index_csv)
    counts = (
        df.groupby(["scenario_family", "label"]).size()
        .reset_index(name="count")
        .sort_values("scenario_family")
    )
    families = counts["scenario_family"].tolist()
    values = counts["count"].tolist()
    colors = [LABEL_COLORS[label] for label in counts["label"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(families))
    bars = ax.bar(x, values, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%d", padding=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("Episode count")
    ax.set_title("Collected episodes by scenario family")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LABEL_COLORS.values()]
    ax.legend(handles, LABEL_COLORS.keys(), title="label", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    _strip_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")


def plot_classifier_comparison(report: dict, out_path: Path) -> None:
    """Grouped bar chart: one group per classifier, one bar per metric.
    Answers the headline research question -- does text/process metadata
    (or the combined model) beat the NVML-only baseline?
    """
    classifiers = report["classifiers_compared"]
    metrics = ["accuracy", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(8, 5))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics
    x = list(range(len(classifiers)))
    for i, metric in enumerate(metrics):
        values = [report["overall"][c][metric] for c in classifiers]
        offsets = [xi + (i - (n_metrics - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, values, width=width, label=metric, color=METRIC_COLORS[metric])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Classifier performance: text vs NVML vs combined")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _strip_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")


def plot_per_family_accuracy(report: dict, out_path: Path) -> None:
    """Grouped bar chart: one group per scenario family, one bar per
    classifier. A high overall accuracy can hide total failure on one
    specific family -- this is where that would show up.
    """
    per_family = report["per_family_accuracy"]
    families = sorted(per_family.keys())
    classifiers = report["classifiers_compared"]

    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(classifiers)
    width = 0.8 / n
    x = list(range(len(families)))
    for i, clf in enumerate(classifiers):
        values = [per_family[f].get(clf) or 0 for f in families]
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, values, width=width, label=clf, color=CLASSIFIER_COLORS.get(clf, "#898781"))
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per scenario family, by classifier")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    _strip_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_episode_distribution(processed_dir / "episodes_index.csv", results_dir / "plot_episode_distribution.png")

    report_path = results_dir / "comparison_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        plot_classifier_comparison(report, results_dir / "plot_classifier_comparison.png")
        plot_per_family_accuracy(report, results_dir / "plot_per_family_accuracy.png")
    else:
        print(f"[plot_results] {report_path} not found -- run experiments/evaluate.py first for the comparison charts")


if __name__ == "__main__":
    main()
