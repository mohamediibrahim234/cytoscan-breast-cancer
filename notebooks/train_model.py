"""Script version of the last notebook cells: trains and saves the deployed model.

    python notebooks/train_model.py            # from the project root
    python train_model.py                      # from notebooks/

Reads  data/Breast Cancer Wisconsin.csv (falls back to sklearn's built-in copy)
Writes model/breast_cancer_model.pkl and model/scaler.pkl
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "Breast Cancer Wisconsin.csv"
MODEL_DIR = ROOT / "model"

FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
]


def get_data() -> pd.DataFrame:
    if CSV.exists():
        print(f"Loading {CSV}")
        return pd.read_csv(CSV).drop(columns=["id", "Unnamed: 32"], errors="ignore")

    print("CSV not found, using sklearn.datasets.load_breast_cancer()")
    raw = load_breast_cancer()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df = df.rename(columns={f"mean {c.replace('_mean', '').replace('_', ' ')}": c for c in FEATURES})
    # sklearn encodes 0 = malignant, 1 = benign
    df["diagnosis"] = pd.Series(raw.target).map({0: "M", 1: "B"})
    return df


def main() -> None:
    df = get_data()
    X, y = df[FEATURES], df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)  # fitted on a DataFrame -> remembers column names
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    pred = model.predict(scaler.transform(X_test))
    print("classes_:", model.classes_)
    print(f"Held-out accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "breast_cancer_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"Saved model + scaler to {MODEL_DIR}")


if __name__ == "__main__":
    main()
