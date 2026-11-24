"""Request / response models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Bounds are loose (well beyond the dataset range) and only reject nonsense
# such as negative sizes.
_POS = dict(ge=0, allow_inf_nan=False)


class TumorFeatures(BaseModel):
    """Mean cell-nucleus measurements from a fine-needle aspirate image."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "radius_mean": 17.99,
                "texture_mean": 10.38,
                "perimeter_mean": 122.8,
                "area_mean": 1001.0,
                "smoothness_mean": 0.1184,
                "compactness_mean": 0.2776,
                "concavity_mean": 0.3001,
                "concave_points_mean": 0.1471,
                "symmetry_mean": 0.2419,
                "fractal_dimension_mean": 0.07871,
            }
        }
    )

    radius_mean: float = Field(..., le=100, **_POS)
    texture_mean: float = Field(..., le=100, **_POS)
    perimeter_mean: float = Field(..., le=1000, **_POS)
    area_mean: float = Field(..., le=10000, **_POS)
    smoothness_mean: float = Field(..., le=1, **_POS)
    compactness_mean: float = Field(..., le=2, **_POS)
    concavity_mean: float = Field(..., le=2, **_POS)
    concave_points_mean: float = Field(..., le=1, **_POS)
    symmetry_mean: float = Field(..., le=1, **_POS)
    fractal_dimension_mean: float = Field(..., le=1, **_POS)


class Prediction(BaseModel):
    label: str = Field(..., examples=["Malignant"])
    confidence: float = Field(..., ge=0, le=1, examples=[0.97])
    probabilities: dict[str, float] = Field(
        ..., examples=[{"Benign": 0.03, "Malignant": 0.97}]
    )


class FilePrediction(BaseModel):
    filename: str
    count: int
    predictions: list[Prediction]


class Health(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]
    features: list[str]
    version: str
