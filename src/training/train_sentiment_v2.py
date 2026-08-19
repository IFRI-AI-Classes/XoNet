from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from src.training.sentiment_pipeline import WeightedDecisionPipeline
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.training.sentiment_pipeline import WeightedDecisionPipeline


LABEL_NAMES = {0: "Neutre", 1: "Negatif", 2: "Positif"}
TARGET_NAMES = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare XoNet sentiment models without changing the dataset."
    )
    parser.add_argument("--data", default="data/corpus_final.csv")
    parser.add_argument("--sep", default="|")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=80_000)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=6)
    parser.add_argument("--no-threshold-tuning", action="store_true")
    parser.add_argument("--include-boosting", action="store_true")
    parser.add_argument("--n-trials", type=int, default=30)
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

    X = np.array(df["fon"].astype(str).tolist(), dtype=object)
    y = np.array(df["label"].tolist(), dtype=int)
    return X, y


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


def make_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        sublinear_tf=True,
        min_df=2,
        lowercase=False,
        norm="l2",
    )


def make_candidates(random_state: int, include_boosting: bool, n_trials: int) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "ComplementNB": ComplementNB(alpha=0.1),
        "SGDClassifier_log_loss_balanced": SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=1e-5,
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
        ),
        "LogisticRegression_balanced": LogisticRegression(
            class_weight="balanced",
            C=4.0,
            solver="saga",
            max_iter=500,
            random_state=random_state,
        ),
        "LinearSVC_balanced_calibrated": CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                C=1.0,
                max_iter=5000,
                random_state=random_state,
            ),
            cv=3,
        ),
        "LinearSVC_calibrated": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                max_iter=5000,
                random_state=random_state,
            ),
            cv=3,
        ),
    }

    if include_boosting:
        add_boosting_candidates(candidates, random_state, n_trials)

    return candidates


def add_boosting_candidates(
    candidates: dict[str, Any], random_state: int, n_trials: int
) -> None:
    try:
        import lightgbm as lgb
    except Exception as exc:
        print(f"[skip] LightGBM unavailable: {exc}")
    else:
        candidates["LightGBM_balanced"] = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    try:
        import xgboost as xgb
    except Exception as exc:
        print(f"[skip] XGBoost unavailable: {exc}")
    else:
        candidates["XGBoost_weighted"] = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=350,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    if n_trials > 0:
        print(
            "[note] --n-trials is reserved for a heavier Optuna pass; "
            "the v2 script currently uses fixed boosting defaults."
        )


def tune_decision_weights(
    probs: np.ndarray, y_true: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    neutral_grid = np.linspace(0.75, 1.05, 7)
    negative_grid = np.linspace(1.00, 1.80, 9)
    positive_grid = np.linspace(1.00, 1.80, 9)

    best_weights = np.ones(3, dtype=float)
    best_preds = np.argmax(probs, axis=1)
    best_score = float(f1_score(y_true, best_preds, average="macro"))

    for neutral in neutral_grid:
        for negative in negative_grid:
            for positive in positive_grid:
                weights = np.array([neutral, negative, positive], dtype=float)
                preds = np.argmax(probs * weights, axis=1)
                score = float(f1_score(y_true, preds, average="macro"))
                if score > best_score:
                    best_score = score
                    best_weights = weights
                    best_preds = preds

    return best_weights, float(best_score), np.asarray(best_preds, dtype=int)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_pred_arr = np.asarray(y_pred, dtype=int)
    per_class_arr = np.asarray(
        f1_score(y_true, y_pred_arr, average=None, labels=[0, 1, 2]), dtype=float
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred_arr)),
        "f1_macro": float(f1_score(y_true, y_pred_arr, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred_arr, average="weighted")),
        "f1_by_class": {
            LABEL_NAMES[i]: float(score)
            for i, score in zip([0, 1, 2], per_class_arr.tolist())
        },
    }


def fit_candidate(
    name: str,
    estimator: Any,
    vectorizer: TfidfVectorizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tune_thresholds: bool,
) -> dict[str, Any]:
    start = perf_counter()
    pipeline = Pipeline(
        [
            ("tfidf", clone(vectorizer)),
            ("model", clone(estimator)),
        ]
    )
    fit_kwargs: dict[str, Any] = {}
    if name == "XGBoost_weighted":
        counts = np.bincount(y_train, minlength=3)
        class_weights = len(y_train) / (len(counts) * counts)
        fit_kwargs["model__sample_weight"] = np.array([class_weights[label] for label in y_train])

    pipeline.fit(X_train, y_train, **fit_kwargs)
    val_pred_base = np.asarray(pipeline.predict(X_val), dtype=int)
    selected_pred = np.asarray(val_pred_base, dtype=int)
    decision_weights = None

    if tune_thresholds and hasattr(pipeline, "predict_proba"):
        val_probs = np.asarray(pipeline.predict_proba(X_val), dtype=float)
        weights, tuned_score, tuned_pred = tune_decision_weights(val_probs, y_val)
        base_score = float(f1_score(y_val, val_pred_base, average="macro"))
        if tuned_score >= base_score:
            selected_pred = np.asarray(tuned_pred, dtype=int)
            decision_weights = weights

    elapsed = perf_counter() - start
    val_metrics = metrics_dict(y_val, np.asarray(selected_pred, dtype=int))
    return {
        "name": name,
        "estimator": estimator,
        "decision_weights": decision_weights,
        "validation": val_metrics,
        "fit_seconds": elapsed,
    }


def fit_final_pipeline(
    candidate: dict[str, Any],
    vectorizer: TfidfVectorizer,
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
) -> WeightedDecisionPipeline:
    name = candidate["name"]
    pipeline = Pipeline(
        [
            ("tfidf", clone(vectorizer)),
            ("model", clone(candidate["estimator"])),
        ]
    )
    fit_kwargs: dict[str, Any] = {}
    if name == "XGBoost_weighted":
        counts = np.bincount(y_train_val, minlength=3)
        class_weights = len(y_train_val) / (len(counts) * counts)
        fit_kwargs["model__sample_weight"] = np.array(
            [class_weights[label] for label in y_train_val]
        )

    pipeline.fit(X_train_val, y_train_val, **fit_kwargs)
    return WeightedDecisionPipeline(
        pipeline=pipeline,
        decision_weights=candidate["decision_weights"],
        label_names=LABEL_NAMES,
        metadata={
            "model_name": name,
            "selection_metric": "f1_macro",
            "decision_weights": (
                candidate["decision_weights"].tolist()
                if candidate["decision_weights"] is not None
                else None
            ),
        },
    )


def evaluate_existing_voting(
    X_test: np.ndarray, y_test: np.ndarray, output_dir: Path
) -> dict[str, Any] | None:
    tfidf_path = output_dir / "tfidf.joblib"
    model_path = output_dir / "voting_model.joblib"
    if not tfidf_path.exists() or not model_path.exists():
        return None

    try:
        tfidf = joblib.load(tfidf_path)
        model = joblib.load(model_path)
        preds = np.asarray(model.predict(tfidf.transform(X_test)), dtype=int)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    return metrics_dict(y_test, preds)


def write_report(
    path: Path,
    best_name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n")
        f.write("Selection metric: f1_macro on validation\n\n")
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        f.write("\n\nClassification report:\n")
        report_text = classification_report(
            y_test, y_pred, target_names=TARGET_NAMES, digits=4, output_dict=False
        )
        f.write(cast(str, report_text))
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data, args.sep)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, args.random_state
    )
    print(f"Dataset: {len(X):,} rows")
    print(f"Train={len(X_train):,} Val={len(X_val):,} Test={len(X_test):,}")
    print(f"Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    vectorizer = make_vectorizer(args)
    candidates = make_candidates(args.random_state, args.include_boosting, args.n_trials)
    tune_thresholds = not args.no_threshold_tuning

    results = []
    for name, estimator in candidates.items():
        print(f"\n[train] {name}")
        result = fit_candidate(
            name,
            estimator,
            vectorizer,
            X_train,
            y_train,
            X_val,
            y_val,
            tune_thresholds,
        )
        results.append(result)
        val = result["validation"]
        weights = result["decision_weights"]
        print(
            f"  val f1_macro={val['f1_macro']:.4f} "
            f"accuracy={val['accuracy']:.4f} "
            f"seconds={result['fit_seconds']:.1f}"
        )
        if weights is not None:
            print(f"  decision_weights={np.round(weights, 3).tolist()}")

    best = max(results, key=lambda item: item["validation"]["f1_macro"])
    print(f"\n[select] {best['name']} by validation f1_macro")

    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    final_pipeline = fit_final_pipeline(best, vectorizer, X_train_val, y_train_val)
    y_pred_test = final_pipeline.predict(X_test)
    test_metrics = metrics_dict(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])

    existing_voting = evaluate_existing_voting(X_test, y_test, output_dir)
    metrics = {
        "best_model": best["name"],
        "selection_metric": "f1_macro",
        "dataset": args.data,
        "split": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
            "random_state": args.random_state,
        },
        "vectorizer": {
            "analyzer": "char_wb",
            "ngram_range": [args.ngram_min, args.ngram_max],
            "max_features": args.max_features,
            "min_df": 2,
            "sublinear_tf": True,
            "lowercase": False,
        },
        "validation_results": {
            item["name"]: {
                **item["validation"],
                "fit_seconds": item["fit_seconds"],
                "decision_weights": (
                    item["decision_weights"].tolist()
                    if item["decision_weights"] is not None
                    else None
                ),
            }
            for item in results
        },
        "test": test_metrics,
        "existing_voting_test": existing_voting,
        "label_names": LABEL_NAMES,
    }

    pipeline_path = output_dir / "sentiment_pipeline_v2.joblib"
    metrics_path = output_dir / "metrics_v2.json"
    report_path = output_dir / "classification_report_v2.txt"

    joblib.dump(final_pipeline, pipeline_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, best["name"], y_test, y_pred_test, cm, metrics)

    print("\n[test]")
    print(f"  accuracy={test_metrics['accuracy']:.4f}")
    print(f"  f1_macro={test_metrics['f1_macro']:.4f}")
    print(f"  f1_weighted={test_metrics['f1_weighted']:.4f}")
    if existing_voting and "accuracy" in existing_voting:
        print(
            "  existing voting: "
            f"accuracy={existing_voting['accuracy']:.4f} "
            f"f1_macro={existing_voting['f1_macro']:.4f}"
        )

    print("\n[write]")
    print(f"  {pipeline_path}")
    print(f"  {metrics_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
