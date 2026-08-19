"""
Combined text + NVML classifier: a single Random Forest trained on BOTH
the TF-IDF shell/process text features (experiments/preprocess.py's
text_features.csv) and the engineered NVML temporal features
(nvml_features.csv) at once, joined on episode_id.

This is a third point of comparison alongside train_text_classifier.py
(text only) and train_nvml_classifier.py (NVML only): does fusing the two
feature families outperform either alone? Same stratified cross-validation
setup as the other two, so all three are directly comparable in
experiments/evaluate.py.

    python experiments/train_combined_classifier.py

Writes results/combined_classifier_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_combined_features(processed_dir: Path, text_column: str) -> tuple[pd.DataFrame, list[str]]:
    """Join text_features.csv and nvml_features.csv on episode_id.

    Both files are produced from the same set of successful episodes by
    preprocess.py, so an inner join should not drop anything -- but we
    join (not just concatenate columns) and report row counts explicitly
    rather than assuming the two files stay in lockstep, since they could
    diverge if one script's output is regenerated without the other's.
    """
    text_df = pd.read_csv(processed_dir / "text_features.csv").fillna({text_column: ""})
    nvml_df = pd.read_csv(processed_dir / "nvml_features.csv")

    nvml_feature_cols = [c for c in nvml_df.columns if c not in ("episode_id", "label", "scenario_family")]
    merged = text_df[["episode_id", "label", "scenario_family", text_column]].merge(
        nvml_df.drop(columns=["label", "scenario_family"]), on="episode_id", how="inner",
    )
    merged[nvml_feature_cols] = merged[nvml_feature_cols].fillna(0.0)

    if len(merged) != len(text_df) or len(merged) != len(nvml_df):
        print(f"[train_combined_classifier] WARNING: text_features.csv has {len(text_df)} rows, "
              f"nvml_features.csv has {len(nvml_df)} rows, joined to {len(merged)} rows -- "
              "some episodes are missing from one file. Re-run preprocess.py if that's unexpected.")

    return merged, nvml_feature_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--text-column", default="text_normalized", choices=["text_normalized", "text_raw"])
    args = parser.parse_args()

    df, nvml_feature_cols = load_combined_features(Path(args.processed_dir), args.text_column)
    if len(df) < args.folds:
        raise ValueError(f"Only {len(df)} episodes available, need at least --folds={args.folds}")

    X = df[[args.text_column] + nvml_feature_cols]
    y = df["label"].tolist()

    # ColumnTransformer routes the text column through TF-IDF and passes
    # the NVML numeric columns through unchanged, so a single Pipeline
    # produces one combined feature matrix for the Random Forest -- this
    # is what makes it "trained on both at once" rather than two separate
    # models whose predictions get combined after the fact.
    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1), args.text_column),
        ("nvml", "passthrough", nvml_feature_cols),
    ])
    pipeline = Pipeline([
        ("features", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed)),
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

    # Fit once on all data purely to report feature importances (descriptive
    # only, not part of the cross-validated metrics above).
    pipeline.fit(X, y)
    feature_names = pipeline.named_steps["features"].get_feature_names_out()
    importances = sorted(zip(feature_names.tolist(), pipeline.named_steps["clf"].feature_importances_.tolist()),
                          key=lambda kv: -kv[1])

    results = {
        "classifier": "combined_text_nvml_random_forest",
        "text_column": args.text_column,
        "n_episodes": len(df),
        "cv_folds": args.folds,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "nvml_feature_columns": nvml_feature_cols,
        "overall": overall,
        "per_family": per_family,
        "predictions": predictions,
        "top_feature_importances": importances[:15],
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "combined_classifier_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[train_combined_classifier] accuracy={overall['accuracy']:.3f} f1={overall['f1']:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
