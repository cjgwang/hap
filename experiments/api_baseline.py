"""
LLM API baseline: ask Claude to zero-shot classify each episode's
normalized shell/process text as "ordinary" or "adversarial", as a baseline to
compare against the trained-from-scratch classifiers in
train_text_classifier.py and train_nvml_classifier.py. The interesting
comparison isn't "does the API baseline win" -- it's whether a
general-purpose model reading the same restricted metadata surface
performs comparably to a classifier fit specifically on this dataset.

    python experiments/api_baseline.py

Requires ANTHROPIC_API_KEY (in the environment, or in a local .env file --
never commit that file). If no key/package is available, this script
prints a message and exits cleanly rather than failing the pipeline, since
it's an optional baseline, not a required stage.

Writes results/api_baseline_results.json in the same schema as the other
two classifiers' results files, so experiments/evaluate.py can compare all
three uniformly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The exact instruction sent to the model, kept as one explicit constant so
# it's easy to audit/change. We ask for a single-word label to keep parsing
# trivial and unambiguous.
PROMPT_TEMPLATE = """You are helping a compute provider's automated triage system classify GPU \
workloads based only on provider-visible metadata (shell commands and process \
command lines observed during the run) -- not on network content or actual \
task success.

Classify the workload below into exactly one of:
  ordinary     - benign, normal ML development or inference activity
  adversarial  - a workload whose visible metadata suggests it should get \
additional (routine) safety review, even though the workload itself is benign

Respond with exactly one word: "ordinary" or "adversarial".

--- workload metadata ---
{text}
--- end metadata ---
"""


def classify_one(client, model: str, text: str, max_retries: int = 5) -> str:
    prompt = PROMPT_TEMPLATE.format(text=text[:4000])  # cap length defensively
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip().lower()
            if "adversarial" in raw:
                return "adversarial"
            if "ordinary" in raw:
                return "ordinary"
            return f"<unparsed: {raw!r}>"
        except Exception as e:
            wait = 2 ** attempt
            print(f"[api_baseline] attempt {attempt + 1}/{max_retries} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    return "<error: max retries exceeded>"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--text-column", default="text_normalized", choices=["text_normalized", "text_raw"])
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="Pause between API calls")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[api_baseline] ANTHROPIC_API_KEY not set (env or .env) -- skipping API baseline. "
              "This is an optional comparison point, not required for the core experiment.")
        return

    try:
        import anthropic
    except ImportError:
        print("[api_baseline] `anthropic` package not installed -- skipping API baseline.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(Path(args.processed_dir) / "text_features.csv").fillna({args.text_column: ""})

    predictions = []
    for _, row in df.iterrows():
        pred = classify_one(client, args.model, row[args.text_column])
        predictions.append({
            "episode_id": row["episode_id"],
            "scenario_family": row["scenario_family"],
            "label": row["label"],
            "predicted": pred,
            "correct": bool(pred == row["label"]),
        })
        print(f"[api_baseline] {row['episode_id']} true={row['label']:<8} pred={pred}")
        time.sleep(args.sleep_seconds)

    valid = [p for p in predictions if p["predicted"] in ("ordinary", "adversarial")]
    n_valid = len(valid)
    accuracy = sum(p["correct"] for p in valid) / n_valid if n_valid else None

    tp = sum(1 for p in valid if p["label"] == "adversarial" and p["predicted"] == "adversarial")
    fp = sum(1 for p in valid if p["label"] == "ordinary" and p["predicted"] == "adversarial")
    fn = sum(1 for p in valid if p["label"] == "adversarial" and p["predicted"] == "ordinary")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_family = {}
    for family in df["scenario_family"].unique():
        rows = [p for p in predictions if p["scenario_family"] == family]
        per_family[family] = {
            "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
        }

    results = {
        "classifier": f"anthropic_api_zero_shot_{args.model}",
        "text_column": args.text_column,
        "n_episodes": len(df),
        "n_valid_predictions": n_valid,
        "overall": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1},
        "per_family": per_family,
        "predictions": predictions,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "api_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[api_baseline] accuracy={accuracy} ({n_valid}/{len(df)} valid) -> {out_path}")


if __name__ == "__main__":
    main()
