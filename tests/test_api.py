from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.preprocessing import FEATURES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Breast Cancer Wisconsin.csv"

# Real rows from the dataset (ids 842302 = M, 8510426 = B), so the tests run
# even when data/ is absent (e.g. in a fresh clone or the Docker image).
MALIGNANT = dict(radius_mean=17.99, texture_mean=10.38, perimeter_mean=122.8, area_mean=1001.0,
                 smoothness_mean=0.1184, compactness_mean=0.2776, concavity_mean=0.3001,
                 concave_points_mean=0.1471, symmetry_mean=0.2419, fractal_dimension_mean=0.07871)
BENIGN = dict(radius_mean=13.54, texture_mean=14.36, perimeter_mean=87.46, area_mean=566.3,
              smoothness_mean=0.09779, compactness_mean=0.08129, concavity_mean=0.06664,
              concave_points_mean=0.04781, symmetry_mean=0.1885, fractal_dimension_mean=0.05766)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan -> loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert sorted(body["classes"]) == ["Benign", "Malignant"]
    assert body["features"] == list(FEATURES)


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "CytoScan" in r.text
    assert "127.0.0.1:8000" in r.text and "location.origin" in r.text  # origin + file:// fallback
    assert client.get("/static/samples.json").status_code == 200


def test_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


@pytest.mark.parametrize("sample,label", [(MALIGNANT, "Malignant"), (BENIGN, "Benign")])
def test_predict_shape_and_label(client, sample, label):
    r = client.post("/predict", json=sample)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"label", "confidence", "probabilities"}
    assert body["label"] == label
    probs = body["probabilities"]
    assert set(probs) == {"Benign", "Malignant"}
    assert abs(sum(probs.values()) - 1) < 1e-3
    # confidence must be the probability of the predicted class (no index swap)
    assert body["confidence"] == probs[label] == max(probs.values())


def test_predict_rejects_bad_json(client):
    assert client.post("/predict", json={**MALIGNANT, "radius_mean": -1}).status_code == 422
    missing = dict(MALIGNANT)
    missing.pop("area_mean")
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**MALIGNANT, "texture_mean": "abc"}).status_code == 422


def _csv(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode()


def test_file_predict(client):
    r = client.post("/predict/file", files={"file": ("cases.csv", _csv([MALIGNANT, BENIGN]), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "cases.csv" and body["count"] == 2
    assert [p["label"] for p in body["predictions"]] == ["Malignant", "Benign"]


def test_file_rejects_non_csv(client):
    r = client.post("/predict/file", files={"file": ("x.png", b"\x89PNG\r\n\x1a\n", "image/png")})
    assert r.status_code == 415


def test_file_rejects_empty_and_corrupt(client):
    assert client.post("/predict/file", files={"file": ("e.csv", b"", "text/csv")}).status_code == 400
    corrupt = b"\x00\xff\xfe garbage \x00\x01"
    assert client.post("/predict/file", files={"file": ("c.csv", corrupt, "text/csv")}).status_code == 400
    wrong_cols = b"a,b,c\n1,2,3\n"
    r = client.post("/predict/file", files={"file": ("w.csv", wrong_cols, "text/csv")})
    assert r.status_code == 400 and "missing columns" in r.json()["detail"]
    bad_val = _csv([{**MALIGNANT, "area_mean": "oops"}])
    assert client.post("/predict/file", files={"file": ("v.csv", bad_val, "text/csv")}).status_code == 400


def test_file_too_large(client, monkeypatch):
    from app import main

    monkeypatch.setattr(main.settings.__class__, "max_upload_bytes", property(lambda s: 100))
    r = client.post("/predict/file", files={"file": ("big.csv", _csv([MALIGNANT] * 5), "text/csv")})
    assert r.status_code == 413


@pytest.mark.skipif(not DATA.exists(), reason="dataset not present in data/")
def test_real_dataset_one_per_class_and_accuracy(client):
    df = pd.read_csv(DATA)
    for code, label in (("B", "Benign"), ("M", "Malignant")):
        row = df[df.diagnosis == code].iloc[0]
        payload = {api: float(row[col]) for api, col in FEATURES.items()}
        assert client.post("/predict", json=payload).json()["label"] == label

    # whole file through the batch endpoint (with training column names)
    r = client.post("/predict/file", files={"file": ("wdbc.csv", DATA.read_bytes(), "text/csv")})
    assert r.status_code == 200
    preds = [p["label"][0] for p in r.json()["predictions"]]
    acc = (pd.Series(preds) == df.diagnosis.values).mean()
    assert acc > 0.95, acc
