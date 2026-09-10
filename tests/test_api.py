"""
Basic API tests: health check, and /predict's structured responses for
invalid input and well-formed input. Uses a synthetic in-memory image,
not the licensed Endava dataset -- these tests must pass on a fresh clone
without any dataset download, since none of them depend on real gauge
content (calibration-rejection tests are content-independent; the
well-formed-input test only checks the response shape, not reading
accuracy -- that's covered by the experiment scripts' evaluation).

Run with: uv run pytest tests/test_api.py
"""

import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from retroread.api import app

client = TestClient(app)


def make_test_image_bytes() -> bytes:
    """A trivial in-memory image -- content doesn't matter for these tests."""
    image = Image.new("RGB", (400, 400), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_insufficient_calibration_points():
    image_bytes = make_test_image_bytes()
    response = client.post(
        "/predict",
        files={"image": ("gauge.png", image_bytes, "image/png")},
        data={"calibration_points": json.dumps([{"x": 100, "y": 100, "value": 0}])},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unable_to_read"
    assert body["reason"] == "insufficient_calibration_points"


def test_predict_rejects_invalid_json():
    image_bytes = make_test_image_bytes()
    response = client.post(
        "/predict",
        files={"image": ("gauge.png", image_bytes, "image/png")},
        data={"calibration_points": "not valid json"},
    )
    assert response.status_code == 200
    assert response.json()["reason"] == "invalid_calibration_json"


def test_predict_returns_well_formed_response_for_valid_input():
    image_bytes = make_test_image_bytes()
    calibration_points = [
        {"x": 100, "y": 200, "value": 0},
        {"x": 300, "y": 200, "value": 10},
    ]
    response = client.post(
        "/predict",
        files={"image": ("gauge.png", image_bytes, "image/png")},
        data={"calibration_points": json.dumps(calibration_points), "unit": "bar"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "unable_to_read")
    assert body["model_version"] == "classical-v1"