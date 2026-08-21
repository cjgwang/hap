"""
Combine and compare classifier results.

    python experiments/evaluate.py

Reads whichever of results/{text_classifier,text_svm,nvml_classifier,
combined_classifier,combined_classifier_shrunk,api_baseline}_results.json
exist (produced by train_text_classifier.py, train_text_svm.py,
train_nvml_classifier.py, train_combined_classifier.py,
train_combined_classifier_shrunk.py, and the optional api_baseline.py),
prints a comparison table, and writes results/comparison_report.json.

This is where the actual research question gets answered: does the text
(shell/process metadata) classifier meaningfully beat the NVML-only
classifier at separating ordinary from adversarial episodes -- and does
fusing both feature families (combined_classifier) beat either alone?
Per-family breakdowns are included because a high overall accuracy can
hide total failure on one scenario family (e.g. if adversarial_training is
trivially separable but adversarial_finetune is not).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULT_FILES = {
    "text": "text_classifier_results.json",
    "text_svm": "text_svm_results.json",
    "nvml": "nvml_classifier_results.json",
    "combined": "combined_classifier_results.json",
    "combined_shrunk": "combined_classifier_shrunk_results.json",
    "api_baseline": "api_baseline_results.json",
}


def load_results(results_dir: Path) -> dict:
    loaded = {}
    for key, filename in RESULT_FILES.items():
        path = results_dir / filename
        if path.exists():
            with open(path) as f:
                loaded[key] = json.load(f)
        else:
            print(f"[evaluate] {filename} not found, skipping (run the corresponding train_*/api_baseline script first)")
    return loaded


def print_comparison_table(loaded: dict) -> None:
    print(f"\n{'classifier':<45} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    print("-" * 87)
    for key, results in loaded.items():
        overall = results["overall"]
        name = results.get("classifier", key)
        acc = overall.get("accuracy")
        prec = overall.get("precision")
        rec = overall.get("recall")
        f1 = overall.get("f1")
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
        print(f"{name:<45} {fmt(acc):>10} {fmt(prec):>10} {fmt(rec):>10} {fmt(f1):>10}")


def per_family_breakdown(loaded: dict) -> dict:
    families = set()
    for results in loaded.values():
        families.update(results.get("per_family", {}).keys())

    breakdown = {}
    for family in sorted(families):
        breakdown[family] = {
            key: results.get("per_family", {}).get(family, {}).get("accuracy")
            for key, results in loaded.items()
        }
    return breakdown


def print_per_family_table(breakdown: dict, classifier_keys: list[str]) -> None:
    print(f"\n{'scenario_family':<28}" + "".join(f"{k:>14}" for k in classifier_keys))
    print("-" * (28 + 14 * len(classifier_keys)))
    for family, accs in breakdown.items():
        row = "".join(
            f"{(f'{accs[k]:.3f}' if isinstance(accs.get(k), (int, float)) else 'n/a'):>14}"
            for k in classifier_keys
        )
        print(f"{family:<28}{row}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    loaded = load_results(results_dir)

    if not loaded:
        print("[evaluate] no results files found -- run train_text_classifier.py / "
              "train_nvml_classifier.py / api_baseline.py first")
        return

    print_comparison_table(loaded)

    breakdown = per_family_breakdown(loaded)
    print_per_family_table(breakdown, list(loaded.keys()))

    report = {
        "classifiers_compared": list(loaded.keys()),
        "overall": {key: results["overall"] for key, results in loaded.items()},
        "per_family_accuracy": breakdown,
    }
    out_path = results_dir / "comparison_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[evaluate] wrote {out_path}")


if __name__ == "__main__":
    main()
