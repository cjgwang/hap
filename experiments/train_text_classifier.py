"""
Text-metadata classifier: TF-IDF over the combined shell/process text
(experiments/preprocess.py's text_features.csv) + a linear classifier,
evaluated with stratified cross-validation (the dataset is small -- 30
episodes by default -- so a single held-out split would be noisy; k-fold
out-of-fold prediction uses every episode as a test example exactly once
while never training on it).

    python experiments/train_text_classifier.py

Writes results/text_classifier_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-column", default="text_normalized", choices=["text_normalized", "text_raw"])
    args = parser.parse_args()

    df = pd.read_csv(Path(args.processed_dir) / "text_features.csv").fillna({args.text_column: ""})
    if len(df) < args.folds:
        raise ValueError(f"Only {len(df)} episodes available, need at least --folds={args.folds}")

    X_text = df[args.text_column].tolist()
    y = df["label"].tolist()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, random_state=args.seed)),
    ])

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y_pred = cross_val_predict(pipeline, X_text, y, cv=cv)

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

    # Fit once on ALL data (not just a fold) to report the most informative
    # TF-IDF terms -- purely descriptive, not part of the evaluated metrics.
    pipeline.fit(X_text, y)
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    top_adversarial_terms = [feature_names[i] for i in coefs.argsort()[-15:][::-1]]
    top_ordinary_terms = [feature_names[i] for i in coefs.argsort()[:15]]

    results = {
        "classifier": "tfidf_logistic_regression",
        "text_column": args.text_column,
        "n_episodes": len(df),
        "cv_folds": args.folds,
        "seed": args.seed,
        "overall": overall,
        "per_family": per_family,
        "predictions": predictions,
        "top_terms_favoring_adversarial": top_adversarial_terms,
        "top_terms_favoring_ordinary": top_ordinary_terms,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "text_classifier_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[train_text_classifier] accuracy={overall['accuracy']:.3f} f1={overall['f1']:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
