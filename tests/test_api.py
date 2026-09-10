"""
Basic API tests: health check, and /predict's structured responses for
invalid input and a real image with real calibration points.

Run with: uv run pytest tests/test_api.py
"""

import json

from fastapi.testclient import TestClient

from retroread.api import app
from retroread.config import ENDAVA_DS5_IMAGES_DIR

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_insufficient_calibration_points():
    image_path = ENDAVA_DS5_IMAGES_DIR / "data/v_0992_f_0000_rgba.png"
    with open(image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"image": ("gauge.png", f, "image/png")},
            data={"calibration_points": json.dumps([{"x": 100, "y": 100, "value": 0}])},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unable_to_read"
    assert body["reason"] == "insufficient_calibration_points"


def test_predict_rejects_invalid_json():
    image_path = ENDAVA_DS5_IMAGES_DIR / "data/v_0992_f_0000_rgba.png"
    with open(image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"image": ("gauge.png", f, "image/png")},
            data={"calibration_points": "not valid json"},
        )
    assert response.status_code == 200
    assert response.json()["reason"] == "invalid_calibration_json"


def test_predict_returns_reading_for_valid_input():
    image_path = ENDAVA_DS5_IMAGES_DIR / "data/v_0992_f_0000_rgba.png"
    # Rough calibration points -- this test checks the endpoint completes
    # and returns a well-formed response, not reading accuracy (that's
    # covered by the experiment scripts' evaluation, not unit tests).
    calibration_points = [
        {"x": 400, "y": 500, "value": 0},
        {"x": 700, "y": 500, "value": 10},
    ]
    with open(image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"image": ("gauge.png", f, "image/png")},
            data={"calibration_points": json.dumps(calibration_points), "unit": "bar"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "unable_to_read")
    assert body["model_version"] == "classical-v1"