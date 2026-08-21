"""
Combined text + NVML classifier, shrunk Random Forest variant.

Same feature construction as train_combined_classifier.py (TF-IDF text +
NVML numeric columns, fused via one ColumnTransformer), but with a much
more regularized Random Forest: fewer trees, shallow max depth, and a
higher min_samples_leaf. This exists because the default 200-tree,
unlimited-depth RF is very likely overfitting at this dataset's size (a
handful of dozens of episodes, ~80% of that per training fold) -- a
smaller/shallower forest is a standard fix for a tree ensemble overfitting
on small n, worth comparing directly against the default rather than
assumed.

    python experiments/train_combined_classifier_shrunk.py

Writes results/combined_classifier_shrunk_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.train_combined_classifier import load_combined_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=50, help="Down from 200 in the unshrunk version")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--text-column", default="text_normalized", choices=["text_normalized", "text_raw"])
    args = parser.parse_args()

    df, nvml_feature_cols = load_combined_features(Path(args.processed_dir), args.text_column)
    if len(df) < args.folds:
        raise ValueError(f"Only {len(df)} episodes available, need at least --folds={args.folds}")

    X = df[[args.text_column] + nvml_feature_cols]
    y = df["label"].tolist()

    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1), args.text_column),
        ("nvml", "passthrough", nvml_feature_cols),
    ])
    pipeline = Pipeline([
        ("features", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
        )),
    ])

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)

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

    pipeline.fit(X, y)
    feature_names = pipeline.named_steps["features"].get_feature_names_out()
    importances = sorted(zip(feature_names.tolist(), pipeline.named_steps["clf"].feature_importances_.tolist()),
                          key=lambda kv: -kv[1])

    results = {
        "classifier": "combined_text_nvml_random_forest_shrunk",
        "text_column": args.text_column,
        "n_episodes": len(df),
        "cv_folds": args.folds,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "nvml_feature_columns": nvml_feature_cols,
        "overall": overall,
        "per_family": per_family,
        "predictions": predictions,
        "top_feature_importances": importances[:15],
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "combined_classifier_shrunk_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[train_combined_classifier_shrunk] accuracy={overall['accuracy']:.3f} f1={overall['f1']:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
