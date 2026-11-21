from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WeightedDecisionPipeline:
    """Wrap a fitted sklearn pipeline and optionally reweight class probabilities."""

    pipeline: Any
    decision_weights: np.ndarray | None = None
    label_names: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def classes_(self) -> np.ndarray:
        if hasattr(self.pipeline, "classes_"):
            return self.pipeline.classes_
        model = getattr(self.pipeline, "named_steps", {}).get("model")
        if model is not None and hasattr(model, "classes_"):
            return model.classes_
        return np.array([0, 1, 2])

    def predict_proba(self, X):
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError("The wrapped pipeline does not expose predict_proba.")
        probs = self.pipeline.predict_proba(X)
        if self.decision_weights is not None:
            weights = np.asarray(self.decision_weights, dtype=float)
            if probs.shape[1] == len(weights):
                weighted = probs * weights
                row_sums = weighted.sum(axis=1, keepdims=True)
                return np.divide(
                    weighted,
                    row_sums,
                    out=np.array(probs, dtype=float, copy=True),
                    where=row_sums > 0,
                )
        return probs

    def predict(self, X):
        if self.decision_weights is not None and hasattr(self.pipeline, "predict_proba"):
            probs = self.predict_proba(X)
            return self.classes_[np.argmax(probs, axis=1)]
        return self.pipeline.predict(X)

