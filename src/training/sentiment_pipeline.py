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
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        if self.decision_weights is not None and hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(X)
            weights = np.asarray(self.decision_weights, dtype=float)
            if probs.shape[1] != len(weights):
                return self.pipeline.predict(X)
            weighted = probs * weights
            return self.classes_[np.argmax(weighted, axis=1)]
        return self.pipeline.predict(X)

