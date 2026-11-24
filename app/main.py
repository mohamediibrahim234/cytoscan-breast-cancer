"""FastAPI routes. Run with:  uvicorn app.main:app --port 8000"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import settings
from .preprocessing import API_FIELDS, PreprocessingError, frame_from_csv, to_frame
from .schemas import FilePrediction, Health, Prediction, TumorFeatures

logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("app")

CSV_TYPES = {
    "text/csv",
    "application/csv",
    "text/plain",
    "application/vnd.ms-excel",  # what Windows browsers send for .csv
    "application/octet-stream",  # curl / some clients, accepted only with .csv name
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model from %s (can take up to ~1 minute on Windows while scikit-learn loads)...", settings.model_dir)
    from .model import Predictor  # imports sklearn; kept here so startup logs come first

    app.state.predictor = Predictor.load(settings.model_path, settings.scaler_path)
    log.info("Ready. Open http://localhost:<port>/ for the web UI, /docs for the API.")
    yield
    app.state.predictor = None


app = FastAPI(
    title="Breast Cancer Diagnosis API",
    description="Predicts **Benign** vs **Malignant** from 10 mean cell-nucleus "
    "measurements (Wisconsin Diagnostic Breast Cancer dataset). Educational demo, "
    "not a medical device.",
    version=__version__,
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # credentials + wildcard origin is rejected by browsers, so only allow
    # credentials when explicit origins are configured.
    allow_credentials="*" not in settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _predictor(request: Request):
    p = getattr(request.app.state, "predictor", None)
    if p is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
    return p


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html", media_type="text/html")


@app.get("/health", response_model=Health, tags=["meta"])
def health(request: Request) -> Health:
    p = getattr(request.app.state, "predictor", None)
    return Health(
        status="ok" if p else "loading",
        model_loaded=p is not None,
        classes=p.labels if p else [],
        features=API_FIELDS,
        version=__version__,
    )


@app.post("/predict", response_model=Prediction, tags=["predict"])
async def predict(features: TumorFeatures, request: Request) -> Prediction:
    """Diagnose a single sample from its 10 mean measurements (JSON body)."""
    p = _predictor(request)
    X = to_frame([features.model_dump()])
    result = await run_in_threadpool(p.predict, X)
    return Prediction(**result[0])


@app.post("/predict/file", response_model=FilePrediction, tags=["predict"])
async def predict_file(request: Request, file: UploadFile = File(...)) -> FilePrediction:
    """Batch-diagnose every row of an uploaded CSV (multipart field `file`).

    The CSV needs the 10 feature columns (training names like `concave points_mean`
    or API names like `concave_points_mean`); other columns are ignored.
    """
    p = _predictor(request)
    name = file.filename or "upload.csv"
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype not in CSV_TYPES or (
        ctype == "application/octet-stream" and not name.lower().endswith(".csv")
    ):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"expected a CSV file, got content type '{ctype or 'unknown'}'",
        )

    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"file larger than {settings.max_upload_mb:g} MB",
        )
    try:
        X = frame_from_csv(raw, settings.max_batch_rows)
    except PreprocessingError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None

    results = await run_in_threadpool(p.predict, X)
    return FilePrediction(
        filename=name, count=len(results), predictions=[Prediction(**r) for r in results]
    )
