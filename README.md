# CytoScan: Breast Cancer Diagnosis API

A FastAPI service plus an animated web UI that classifies a breast mass as **Benign** or **Malignant**. It uses 10 mean cell-nucleus measurements from the [Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (569 samples).

The model is a `StandardScaler` followed by a `RandomForestClassifier` (100 trees). It scores **95.6 % accuracy on the held-out 20 % test split**.

The web UI draws a simulated microscope slide. The nuclei change shape live as you move the sliders: radius sets size, texture sets chromatin grain, concavity adds dents, and fractal dimension roughens the edge. When you press **Scan & diagnose**, a scanning beam sweeps the slide and the pipeline steps light up. A gauge then shows the model's confidence. You can load real held-out cases to compare the prediction with the pathology label, or upload a CSV to predict many rows at once.

> ⚠️ This is an educational demo, **not a medical device**.

## Project structure

```
app/
  main.py           FastAPI routes, lifespan model loading, CORS, validation
  model.py          load model + scaler once, predict (label mapping read from model.classes_)
  preprocessing.py  JSON/CSV -> DataFrame with the exact training columns & order
  config.py         settings from environment variables
  schemas.py        Pydantic request/response models
model/              breast_cancer_model.pkl, scaler.pkl  (scikit-learn 1.8.0)
static/             index.html (web UI), samples.json (held-out demo cases + slider ranges)
notebooks/          breast_cancer_wisconsin.ipynb (EDA + training, saves to ../model)
                    train_model.py (same training as a script)
                    other/  unrelated earlier notebooks (diabetes, shopping trends)
data/               Breast Cancer Wisconsin.csv   (git-ignored)
tests/              pytest suite
Dockerfile, .dockerignore, requirements.txt, requirements-dev.txt, run.bat
```

## Run locally

**Windows:** double-click **`run.bat`**. On first run it creates `.venv` and installs the requirements. It then starts the server on port 8000, or the next free port if 8000 is busy, and opens your browser once `/health` responds.

**Manually:**

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Open <http://127.0.0.1:8000> for the UI and <http://127.0.0.1:8000/docs> for the interactive API docs.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The real-dataset test runs only when `data/Breast Cancer Wisconsin.csv` is present.

### Retrain

Put the CSV in `data/` and run either `notebooks/breast_cancer_wisconsin.ipynb` or `python notebooks/train_model.py`. Both write `model/breast_cancer_model.pkl` and `model/scaler.pkl`.

## API

| Method | Path            | Body                                  | Returns |
|--------|-----------------|---------------------------------------|---------|
| GET    | `/`             | none                                  | web UI |
| GET    | `/health`       | none                                  | `{status, model_loaded, classes, features, version}` |
| POST   | `/predict`      | JSON with the 10 features             | `{label, confidence, probabilities}` |
| POST   | `/predict/file` | multipart field `file` (CSV)          | `{filename, count, predictions[]}` |
| GET    | `/docs`         | none                                  | Swagger UI |

**Errors:** `422` for invalid, missing or negative JSON fields. `415` when the upload is not a CSV. `400` for an empty or unparsable CSV, missing columns or non-numeric values. `413` when the upload exceeds `MAX_UPLOAD_MB`.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001,
  "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001,
  "concave_points_mean": 0.1471, "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871}'
# {"label":"Malignant","confidence":0.93,"probabilities":{"Benign":0.07,"Malignant":0.93}}

curl -F "file=@data/Breast Cancer Wisconsin.csv;type=text/csv" http://127.0.0.1:8000/predict/file
```

A CSV can use the original column names (`concave points_mean`) or the API names (`concave_points_mean`). Extra columns such as `id` or `diagnosis` are ignored.

## Environment variables

| Variable         | Default                   | Meaning |
|------------------|---------------------------|---------|
| `PORT`           | `8000`                    | port (Docker, `run.bat`) |
| `CORS_ORIGINS`   | `*`                       | comma-separated allowed origins, e.g. `https://user.github.io` |
| `MODEL_DIR`      | `./model`                 | folder containing the `.pkl` files |
| `MODEL_FILE`     | `breast_cancer_model.pkl` | classifier file name |
| `SCALER_FILE`    | `scaler.pkl`              | scaler file name |
| `MAX_UPLOAD_MB`  | `2`                       | CSV upload size limit |
| `MAX_BATCH_ROWS` | `5000`                    | maximum rows per CSV |
| `LOG_LEVEL`      | `INFO`                    | logging level |

## Docker

```bash
docker build -t cytoscan .
docker run -p 8000:8000 cytoscan
docker run -p 7860:7860 -e PORT=7860 cytoscan   # any port
```

The image is based on `python:3.12-slim`. It runs as a non-root user with 1 worker and has a `HEALTHCHECK` on `/health`.

## Deploy to Hugging Face Spaces

1. Create a new Space at <https://huggingface.co/new-space> and choose **Docker** with the **Blank** template.
2. Push this repo to the Space, or upload `app/`, `model/`, `static/`, `requirements.txt` and `Dockerfile`.
3. At the top of the Space's `README.md`, add:

   ```yaml
   ---
   title: CytoScan
   emoji: 🔬
   colorFrom: pink
   colorTo: purple
   sdk: docker
   app_port: 7860
   ---
   ```
4. In **Settings → Variables**, set `PORT=7860`, or change `ENV PORT=8000` in the Dockerfile to `7860`.
5. The Space builds and serves the UI at `https://<user>-<space>.hf.space`.

## Notes and known issues

- **Windows Smart App Control:** newer SciPy wheels are blocked ("DLL load failed … Application Control policy has blocked this file"). For that reason `scipy==1.16.3` is pinned. The `jupyter.exe` launcher is blocked as well, so run notebooks from VS Code or with `python -m` APIs.
- **Version match:** `scikit-learn` must stay at the version that pickled the model (1.8.0).
- **`git` "dubious ownership" on an external drive:** run `git config --global --add safe.directory "E:/project ml deployment"`.
