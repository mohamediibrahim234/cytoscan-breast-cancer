"""Runtime settings, all overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_DIR", BASE_DIR / "model"))
    )
    model_file: str = field(
        default_factory=lambda: os.getenv("MODEL_FILE", "breast_cancer_model.pkl")
    )
    scaler_file: str = field(default_factory=lambda: os.getenv("SCALER_FILE", "scaler.pkl"))
    static_dir: Path = field(
        default_factory=lambda: Path(os.getenv("STATIC_DIR", BASE_DIR / "static"))
    )
    # Comma separated, e.g. "https://me.github.io,http://localhost:5500". "*" allows all.
    cors_origins: list[str] = field(
        default_factory=lambda: _list(os.getenv("CORS_ORIGINS", "*"))
    )
    max_upload_mb: float = field(default_factory=lambda: float(os.getenv("MAX_UPLOAD_MB", "2")))
    max_batch_rows: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_ROWS", "5000")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_file

    @property
    def scaler_path(self) -> Path:
        return self.model_dir / self.scaler_file

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


settings = Settings()
