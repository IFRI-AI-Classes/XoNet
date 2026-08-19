from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    from src.training.sentiment_pipeline import WeightedDecisionPipeline
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.training.sentiment_pipeline import WeightedDecisionPipeline


LABEL_NAMES = {0: "Neutre", 1: "Negatif", 2: "Positif"}
TARGET_NAMES = [LABEL_NAMES[index] for index in sorted(LABEL_NAMES)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark hybrid TF-IDF/SVM models for XoNet.")
    parser.add_argument("--data", default="data/corpus_final.csv")
    parser.add_argument("--sep", default="|")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--no-threshold-tuning", action="store_true")
    return parser.parse_args()


def load_dataset(path: str, sep: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=sep)
    required = {"fon", "sentiment_final"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["sentiment_final"] = pd.to_numeric(df["sentiment_final"], errors="coerce")
    df = df.dropna(subset=["fon", "sentiment_final"]).copy()
    df["label"] = df["sentiment_final"].astype(int)
    df = df[df["label"].isin(LABEL_NAMES)].copy()
    return (
        np.asarray(df["fon"].astype(str).to_numpy(), dtype=object),
        np.asarray(df["label"].to_numpy(), dtype=int),
    )


def split_dataset(
    X: np.ndarray, y: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_vectorizers(max_features: int) -> dict[str, Any]:
    char_26 = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 6), max_features=max_features,
        min_df=2, sublinear_tf=True, lowercase=False, norm="l2"
    )
    char_37 = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 7), max_features=max_features,
        min_df=2, sublinear_tf=True, lowercase=False, norm="l2"
    )
    word_15 = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 5), max_features=max_features // 2,
        min_df=2, sublinear_tf=True, lowercase=False, norm="l2"
    )
    hybrid_26 = FeatureUnion([("char", char_26), ("word", word_15)])
    hybrid_37 = FeatureUnion([
        ("char", char_37),
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 4), max_features=max_features // 2,
            min_df=2, sublinear_tf=True, lowercase=False, norm="l2"
        )),
    ])
    return {
        "char_2_6": char_26,
        "char_3_7": char_37,
        "hybrid_2_6_word_1_5": hybrid_26,
        "hybrid_3_7_word_1_4": hybrid_37,
    }


def make_candidates(random_state: int) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for vectorizer_name in [
        "char_2_6", "char_3_7", "hybrid_2_6_word_1_5", "hybrid_3_7_word_1_4"
    ]:
        for class_weight in [None, "balanced"]:
            weight_name = "none" if class_weight is None else "balanced"
            candidates[f"{vectorizer_name}_svm_{weight_name}_c1"] = {
                "vectorizer": vectorizer_name,
                "estimator": CalibratedClassifierCV(
                    estimator=LinearSVC(
                        C=1.0, class_weight=class_weight,
                        max_iter=5000, random_state=random_state
                    ),
                    cv=3,
                ),
            }
    return candidates


def tune_decision_weights(
    probs: np.ndarray, y_true: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    best_weights = np.ones(3, dtype=float)
    best_preds = np.argmax(probs, axis=1)
    best_score = float(f1_score(y_true, best_preds, average="macro"))
    for neutral in np.linspace(0.75, 1.05, 7):
        for negative in np.linspace(1.00, 1.80, 9):
            for positive in np.linspace(1.00, 1.80, 9):
                weights = np.array([neutral, negative, positive], dtype=float)
                preds = np.argmax(probs * weights, axis=1)
                score = float(f1_score(y_true, preds, average="macro"))
                if score > best_score:
                    best_score, best_weights, best_preds = score, weights, preds
    return best_weights, best_score, np.asarray(best_preds, dtype=int)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_pred = np.asarray(y_pred, dtype=int)
    per_class = np.asarray(
        f1_score(y_true, y_pred, average=None, labels=[0, 1, 2]), dtype=float
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_by_class": {
            LABEL_NAMES[index]: float(score)
            for index, score in zip([0, 1, 2], per_class.tolist())
        },
    }


def fit_one(
    name: str, config: dict[str, Any], vectorizers: dict[str, Any],
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    tune_thresholds: bool,
) -> dict[str, Any]:
    start = perf_counter()
    pipeline = Pipeline([
        ("tfidf", vectorizers[config["vectorizer"]]),
        ("model", config["estimator"]),
    ])
    pipeline.fit(X_train, y_train)
    base_predictions = np.asarray(pipeline.predict(X_val), dtype=int)
    selected_predictions = base_predictions
    decision_weights = None
    if tune_thresholds:
        probabilities = np.asarray(pipeline.predict_proba(X_val), dtype=float)
        weights, tuned_score, tuned_predictions = tune_decision_weights(probabilities, y_val)
        base_score = float(f1_score(y_val, base_predictions, average="macro"))
        if tuned_score >= base_score:
            selected_predictions = tuned_predictions
            decision_weights = weights
    return {
        "name": name,
        "config": config,
        "validation": metric_dict(y_val, selected_predictions),
        "decision_weights": decision_weights,
        "fit_seconds": perf_counter() - start,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset(args.data, args.sep)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, args.random_state)
    vectorizers = make_vectorizers(args.max_features)
    candidates = make_candidates(args.random_state)
    results = []

    print(f"Dataset: {len(X):,} rows")
    print(f"Train={len(X_train):,} Val={len(X_val):,} Test={len(X_test):,}")
    for name, config in candidates.items():
        print(f"[train] {name}", flush=True)
        result = fit_one(
            name, config, vectorizers, X_train, y_train, X_val, y_val,
            not args.no_threshold_tuning,
        )
        results.append(result)
        print(
            f"  val f1_macro={result['validation']['f1_macro']:.4f} "
            f"accuracy={result['validation']['accuracy']:.4f} "
            f"seconds={result['fit_seconds']:.1f}", flush=True
        )

    best = max(results, key=lambda item: item["validation"]["f1_macro"])
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    final_pipeline = Pipeline([
        ("tfidf", vectorizers[best["config"]["vectorizer"]]),
        ("model", best["config"]["estimator"]),
    ])
    final_pipeline.fit(X_train_val, y_train_val)
    wrapped = WeightedDecisionPipeline(
        pipeline=final_pipeline,
        decision_weights=best["decision_weights"],
        label_names=LABEL_NAMES,
        metadata={
            "model_name": best["name"],
            "selection_metric": "f1_macro",
            "decision_weights": (
                best["decision_weights"].tolist()
                if best["decision_weights"] is not None else None
            ),
        },
    )
    test_predictions = wrapped.predict(X_test)
    test_metrics = metric_dict(y_test, test_predictions)
    metrics = {
        "best_model": best["name"],
        "selection_metric": "f1_macro",
        "dataset": args.data,
        "split": {
            "train": len(X_train), "validation": len(X_val), "test": len(X_test),
            "random_state": args.random_state,
        },
        "max_features": args.max_features,
        "validation_results": {
            item["name"]: {
                **item["validation"],
                "fit_seconds": item["fit_seconds"],
                "decision_weights": (
                    item["decision_weights"].tolist()
                    if item["decision_weights"] is not None else None
                ),
            }
            for item in results
        },
        "test": test_metrics,
        "label_names": LABEL_NAMES,
    }
    joblib.dump(wrapped, output_dir / "sentiment_pipeline_v3.joblib")
    (output_dir / "metrics_v3.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = classification_report(y_test, test_predictions, target_names=TARGET_NAMES, digits=4)
    report += "\nConfusion matrix:\n" + np.array2string(
        confusion_matrix(y_test, test_predictions, labels=[0, 1, 2])
    )
    (output_dir / "classification_report_v3.txt").write_text(report, encoding="utf-8")
    print(f"[best] {best['name']}")
    print(f"[test] accuracy={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")
    print("[write] models/sentiment_pipeline_v3.joblib, metrics_v3.json, classification_report_v3.txt")


if __name__ == "__main__":
    main()
