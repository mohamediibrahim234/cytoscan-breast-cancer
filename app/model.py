"""Load the scaler + classifier once and run predictions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocessing import TRAIN_COLUMNS

log = logging.getLogger(__name__)

# The classifier was trained on the CSV "diagnosis" column: 'B' / 'M'.
# We never assume index order; it is read from model.classes_ at load time.
CODE_TO_LABEL = {"B": "Benign", "M": "Malignant"}


@dataclass
class Predictor:
    model: object
    scaler: object
    labels: list[str]  # human labels aligned with model.classes_ / predict_proba columns

    @classmethod
    def load(cls, model_path: Path, scaler_path: Path) -> "Predictor":
        t0 = time.perf_counter()
        for p in (model_path, scaler_path):
            if not p.is_file():
                raise FileNotFoundError(f"model file not found: {p}")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        n_model = getattr(model, "n_features_in_", None)
        n_scaler = getattr(scaler, "n_features_in_", None)
        if n_model != len(TRAIN_COLUMNS) or n_scaler != len(TRAIN_COLUMNS):
            raise ValueError(
                f"model expects {n_model} / scaler {n_scaler} features, "
                f"app sends {len(TRAIN_COLUMNS)}"
            )
        names = getattr(scaler, "feature_names_in_", None)
        if names is not None and list(names) != TRAIN_COLUMNS:
            raise ValueError(f"scaler column order {list(names)} != {TRAIN_COLUMNS}")

        labels = [CODE_TO_LABEL.get(str(c), str(c)) for c in model.classes_]
        log.info(
            "Model loaded in %.2fs: %s, classes=%s",
            time.perf_counter() - t0,
            type(model).__name__,
            dict(zip(map(str, model.classes_), labels)),
        )
        return cls(model=model, scaler=scaler, labels=labels)

    def predict(self, X: pd.DataFrame) -> list[dict]:
        scaled = self.scaler.transform(X[TRAIN_COLUMNS])
        proba = np.asarray(self.model.predict_proba(scaled))
        out = []
        for row in proba:
            idx = int(row.argmax())
            out.append(
                {
                    "label": self.labels[idx],
                    "confidence": round(float(row[idx]), 4),
                    "probabilities": {
                        lbl: round(float(p), 4) for lbl, p in zip(self.labels, row)
                    },
                }
            )
        return out
