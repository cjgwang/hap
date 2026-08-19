"""
NVML-telemetry classifier: engineered temporal features over 1 Hz GPU
telemetry (experiments/preprocess.py's nvml_features.csv) + a Random
Forest, evaluated with stratified cross-validation. This mirrors the
methodology of the referenced paper (content-agnostic NVML sampling +
engineered features + Random Forest) as the "compute-only" baseline that
experiments/train_text_classifier.py's metadata signal is compared
against in experiments/evaluate.py.

    python experiments/train_nvml_classifier.py

Writes results/nvml_classifier_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(Path(args.processed_dir) / "nvml_features.csv")
    if len(df) < args.folds:
        raise ValueError(f"Only {len(df)} episodes available, need at least --folds={args.folds}")

    non_feature_cols = {"episode_id", "label", "scenario_family"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols].fillna(0.0)
    y = df["label"].tolist()

    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    overall = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, pos_label="adversarial", zero_division=0),
        "recall": recall_score(y, y_pred, pos_label="adversarial", zero_division=0),
        "f1": f1_score(y, y_pred, pos_label="adversarial", zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=["ordinary", "adversarial"]).tolist(),
        "confusion_matrix_labels": ["ordinary", "adversarial"],
    }

    predictions = []
    for i, row in df.iterrows():
        predictions.append({
            "episode_id": row["episode_id"],
            "scenario_family": row["scenario_family"],
            "label": row["label"],
            "predicted": y_pred[i],
            "correct": bool(row["label"] == y_pred[i]),
        })

    per_family = {}
    for family in df["scenario_family"].unique():
        rows = [p for p in predictions if p["scenario_family"] == family]
        per_family[family] = {
            "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
        }

    # Fit once on all data purely to report feature importances (descriptive
    # only, not part of the cross-validated metrics above).
    clf.fit(X, y)
    importances = sorted(zip(feature_cols, clf.feature_importances_.tolist()), key=lambda kv: -kv[1])

    results = {
        "classifier": "nvml_temporal_features_random_forest",
        "n_episodes": len(df),
        "cv_folds": args.folds,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "feature_columns": feature_cols,
        "overall": overall,
        "per_family": per_family,
        "predictions": predictions,
        "top_feature_importances": importances[:15],
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "nvml_classifier_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[train_nvml_classifier] accuracy={overall['accuracy']:.3f} f1={overall['f1']:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
