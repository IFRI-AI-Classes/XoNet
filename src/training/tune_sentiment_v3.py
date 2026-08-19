from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.training.sentiment_pipeline import WeightedDecisionPipeline

LABEL_NAMES = {0: "Neutre", 1: "Negatif", 2: "Positif"}
TARGET_NAMES = [LABEL_NAMES[index] for index in sorted(LABEL_NAMES)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune C for XoNet hybrid TF-IDF/SVM.")
    parser.add_argument("--data", default="data/corpus_final.csv")
    parser.add_argument("--sep", default="|")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--max-features", type=int, default=150_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_data(path: str, sep: str) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path, sep=sep)
    df["sentiment_final"] = pd.to_numeric(df["sentiment_final"], errors="coerce")
    df = df.dropna(subset=["fon", "sentiment_final"]).copy()
    df["label"] = df["sentiment_final"].astype(int)
    df = df[df["label"].isin(LABEL_NAMES)]
    X = np.asarray(df["fon"].astype(str).to_numpy(), dtype=object)
    y = np.asarray(df["label"].to_numpy(), dtype=int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorizer(max_features: int) -> FeatureUnion:
    return FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 6), max_features=max_features,
            min_df=2, sublinear_tf=True, lowercase=False, norm="l2"
        )),
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 5), max_features=max_features // 2,
            min_df=2, sublinear_tf=True, lowercase=False, norm="l2"
        )),
    ])


def tune_weights(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    best_weights = np.ones(3, dtype=float)
    best_pred = np.argmax(probs, axis=1)
    best_score = float(f1_score(y, best_pred, average="macro"))
    for neutral in np.linspace(0.75, 1.05, 7):
        for negative in np.linspace(1.0, 1.8, 9):
            for positive in np.linspace(1.0, 1.8, 9):
                weights = np.array([neutral, negative, positive])
                pred = np.argmax(probs * weights, axis=1)
                score = float(f1_score(y, pred, average="macro"))
                if score > best_score:
                    best_weights, best_pred, best_score = weights, pred, score
    return best_weights, np.asarray(best_pred, dtype=int), best_score


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "f1_weighted": float(f1_score(y, pred, average="weighted")),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data, args.sep)
    results = []
    for C in [0.25, 0.5, 1.0, 2.0, 4.0]:
        start = perf_counter()
        pipeline = Pipeline([
            ("tfidf", vectorizer(args.max_features)),
            ("model", CalibratedClassifierCV(
                estimator=LinearSVC(C=C, max_iter=5000, random_state=args.random_state),
                cv=3,
            )),
        ])
        pipeline.fit(X_train, y_train)
        probs = np.asarray(pipeline.predict_proba(X_val), dtype=float)
        weights, val_pred, val_score = tune_weights(probs, y_val)
        results.append({
            "C": C,
            "validation": {**metrics(y_val, val_pred), "f1_macro_tuned": val_score},
            "weights": weights.tolist(),
            "seconds": perf_counter() - start,
        })
        print(f"C={C} val_f1_macro={val_score:.4f} accuracy={results[-1]['validation']['accuracy']:.4f} seconds={results[-1]['seconds']:.1f}", flush=True)

    best = max(results, key=lambda item: item["validation"]["f1_macro_tuned"])
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    final_pipeline = Pipeline([
        ("tfidf", vectorizer(args.max_features)),
        ("model", CalibratedClassifierCV(
            estimator=LinearSVC(C=best["C"], max_iter=5000, random_state=args.random_state),
            cv=3,
        )),
    ])
    final_pipeline.fit(X_train_val, y_train_val)
    wrapped = WeightedDecisionPipeline(
        pipeline=final_pipeline,
        decision_weights=np.asarray(best["weights"], dtype=float),
        label_names=LABEL_NAMES,
        metadata={"model_name": f"hybrid_char_word_svm_c{best['C']}", "selection_metric": "f1_macro"},
    )
    test_pred = wrapped.predict(X_test)
    test_metrics = metrics(y_test, test_pred)
    output = {
        "best_model": f"hybrid_char_word_svm_c{best['C']}",
        "selection_metric": "f1_macro",
        "max_features": args.max_features,
        "validation_results": results,
        "test": test_metrics,
        "label_names": LABEL_NAMES,
    }
    joblib.dump(wrapped, output_dir / "sentiment_pipeline_v3_tuned.joblib")
    (output_dir / "metrics_v3_tuned.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    report = classification_report(y_test, test_pred, target_names=TARGET_NAMES, digits=4)
    report += "\nConfusion matrix:\n" + np.array2string(confusion_matrix(y_test, test_pred, labels=[0, 1, 2]))
    (output_dir / "classification_report_v3_tuned.txt").write_text(report, encoding="utf-8")
    print(f"[best] C={best['C']} [test] accuracy={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
