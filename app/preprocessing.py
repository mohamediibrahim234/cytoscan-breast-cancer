"""Turn request data into exactly the matrix the model was trained on.

Training (notebooks/train_model.py) used these 10 columns, in this order, from
the Wisconsin CSV, then a StandardScaler fitted on the training split. The
scaler was fitted on a DataFrame, so we hand it a DataFrame with the original
column names (note the space in "concave points_mean").
"""

from __future__ import annotations

import io
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

# API field name -> column name used during training
FEATURES: dict[str, str] = {
    "radius_mean": "radius_mean",
    "texture_mean": "texture_mean",
    "perimeter_mean": "perimeter_mean",
    "area_mean": "area_mean",
    "smoothness_mean": "smoothness_mean",
    "compactness_mean": "compactness_mean",
    "concavity_mean": "concavity_mean",
    "concave_points_mean": "concave points_mean",
    "symmetry_mean": "symmetry_mean",
    "fractal_dimension_mean": "fractal_dimension_mean",
}
API_FIELDS = list(FEATURES.keys())
TRAIN_COLUMNS = list(FEATURES.values())


class PreprocessingError(ValueError):
    """Input could not be turned into a valid feature matrix."""


def to_frame(rows: Iterable[Mapping[str, float]]) -> pd.DataFrame:
    """Rows keyed by API field names -> DataFrame with training column names."""
    records = []
    for i, row in enumerate(rows):
        missing = [f for f in API_FIELDS if f not in row]
        if missing:
            raise PreprocessingError(f"row {i}: missing fields {missing}")
        values = []
        for f in API_FIELDS:
            try:
                v = float(row[f])
            except (TypeError, ValueError):
                raise PreprocessingError(f"row {i}: '{f}' is not a number") from None
            if not math.isfinite(v):
                raise PreprocessingError(f"row {i}: '{f}' must be finite")
            values.append(v)
        records.append(values)
    if not records:
        raise PreprocessingError("no rows to predict")
    return pd.DataFrame(records, columns=TRAIN_COLUMNS)


def frame_from_csv(raw: bytes, max_rows: int) -> pd.DataFrame:
    """Parse an uploaded CSV. Accepts either training column names
    ("concave points_mean") or API names ("concave_points_mean"); extra columns
    such as id/diagnosis are ignored."""
    if not raw or not raw.strip():
        raise PreprocessingError("uploaded file is empty")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # pandas raises many different parser errors
        raise PreprocessingError(f"could not parse CSV: {exc.__class__.__name__}") from None
    if df.empty:
        raise PreprocessingError("CSV has a header but no rows")
    if len(df) > max_rows:
        raise PreprocessingError(f"CSV has {len(df)} rows; the limit is {max_rows}")

    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.rename(columns={api: train for api, train in FEATURES.items() if api in df.columns})
    missing = [c for c in TRAIN_COLUMNS if c not in df.columns]
    if missing:
        raise PreprocessingError(f"CSV is missing columns: {missing}")

    out = df[TRAIN_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(out.to_numpy(dtype=float)).all(axis=1)  # NaN / inf / unparsable
    if bad.any():
        rows = (np.flatnonzero(bad) + 2).tolist()[:10]  # +2 = header line + 1-based
        raise PreprocessingError(f"non-numeric or missing values on CSV lines {rows}")
    return out.astype(float)
