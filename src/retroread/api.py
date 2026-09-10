"""
Minimal FastAPI service exposing the classical gauge-reading pipeline.

POST /predict: image + calibration points -> reading, confidence, or
                a structured abstention reason.
GET  /health:  liveness check.

Calibration must be supplied by the caller (pixel position + known value
for at least 2 points on the gauge's scale) -- see predict.py and
docs/PRODUCT_HYPOTHESIS.md for why this system doesn't read printed scale
numbers itself. In practice this would be configured once per gauge type.
"""

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from retroread.classical_baseline import find_gauge_circle
from retroread.params import CIRCLE_PARAMS
from retroread.predict import read_gauge
from retroread.reading_conversion import fit_scale_calibration

app = FastAPI(title="RetroRead", version="0.1.0")


class PredictResponse(BaseModel):
    status: str
    reading: float | None = None
    unit: str | None = None
    confidence: float | None = None
    reason: str | None = None
    model_version: str = "classical-v1"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    calibration_points: str = Form(
        ..., description='JSON list of {"x": px, "y": px, "value": number}, at least 2 points'
    ),
    unit: str = Form("", description="Unit label to echo back, e.g. 'bar' -- not used in calculation"),
):
    try:
        points = json.loads(calibration_points)
    except json.JSONDecodeError:
        return PredictResponse(status="unable_to_read", reason="invalid_calibration_json")

    if not isinstance(points, list) or len(points) < 2:
        return PredictResponse(status="unable_to_read", reason="insufficient_calibration_points")

    with tempfile.NamedTemporaryFile(suffix=Path(image.filename).suffix, delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        circle = find_gauge_circle(tmp_path, **CIRCLE_PARAMS)
        if not circle.found:
            return PredictResponse(status="unable_to_read", reason="gauge_not_detected")

        calibration = fit_scale_calibration(circle.center_x, circle.center_y, points)
        result = read_gauge(tmp_path, calibration)

        return PredictResponse(
            status=result.status,
            reading=round(result.reading, 3) if result.reading is not None else None,
            unit=unit or None,
            confidence=round(result.confidence, 3) if result.confidence is not None else None,
            reason=result.reason,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)