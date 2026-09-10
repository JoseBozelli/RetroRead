"""
RetroRead Streamlit UI: upload a gauge photo, calibrate by reading numbers
at marked positions (OCR-suggested + click-to-add, never raw pixel
coordinates), and get a reading from the production ensemble
(classical + YOLO/refiner, agreement/confidence gated).

Run with:
    uv run streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO

from retroread.classical_baseline import find_gauge_circle
from retroread.ocr_calibration import detect_scale_labels
from retroread.params import CIRCLE_PARAMS
from retroread.predict import read_gauge_ensemble
from retroread.reading_conversion import fit_scale_calibration
from retroread.tip_refiner_model import TipRefinerModel

st.set_page_config(page_title="RetroRead", page_icon="⏱️", layout="centered")

DISPLAY_WIDTH = 500
MARKER_RADIUS = 10


@st.cache_resource
def load_models():
    yolo_model = YOLO("runs/pose/yolo_runs/experiment_30/weights/best.pt")
    refiner_model = TipRefinerModel(freeze_backbone=True)
    refiner_model.load_state_dict(torch.load("best_tip_refiner_model.pt"))
    refiner_model.eval()
    return yolo_model, refiner_model


def draw_markers(image: Image.Image, points: list[dict]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for i, p in enumerate(points, start=1):
        x, y = p["display_x"], p["display_y"]
        draw.ellipse(
            (x - MARKER_RADIUS, y - MARKER_RADIUS, x + MARKER_RADIUS, y + MARKER_RADIUS),
            outline="red", width=3,
        )
        draw.text((x + MARKER_RADIUS + 2, y - MARKER_RADIUS), str(i), fill="red")
    return annotated


st.title("RetroRead")
st.caption("Analog gauge reading — classical CV + deep learning ensemble")

uploaded_file = st.file_uploader("Upload a gauge photo", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("Upload a gauge photo to get started.")
    st.stop()

if st.session_state.get("last_file") != uploaded_file.name:
    # New image: reset calibration state.
    for key in ["points", "last_click", "tmp_path"]:
        st.session_state.pop(key, None)
    st.session_state["last_file"] = uploaded_file.name

image = Image.open(uploaded_file).convert("RGB")

if "tmp_path" not in st.session_state:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        st.session_state["tmp_path"] = tmp.name
tmp_path = st.session_state["tmp_path"]

circle = find_gauge_circle(tmp_path, **CIRCLE_PARAMS)

if not circle.found:
    st.error("Could not detect a gauge in this image. Try a clearer, more centered photo.")
    st.stop()

st.success("Gauge detected.")

scale_factor = DISPLAY_WIDTH / image.width
display_image = image.resize((DISPLAY_WIDTH, int(image.height * scale_factor)))

if "points" not in st.session_state:
    with st.spinner("Locating scale markings..."):
        ocr_points = detect_scale_labels(tmp_path, circle)  # positions only -- OCR's own value guesses are unreliable
    st.session_state["points"] = [
        {"x": p["x"], "y": p["y"], "display_x": p["x"] * scale_factor, "display_y": p["y"] * scale_factor, "value": None}
        for p in ocr_points
    ]

st.subheader("Calibration")
st.caption(
    "Each marked point needs the number printed on the gauge at that spot. "
    "Click anywhere on the image to mark an unmarked tick."
)

annotated_display = draw_markers(display_image, st.session_state["points"])
click = streamlit_image_coordinates(annotated_display, key="gauge_click")

if click is not None and click != st.session_state.get("last_click"):
    st.session_state["last_click"] = click
    st.session_state["points"].append({
        "x": click["x"] / scale_factor,
        "y": click["y"] / scale_factor,
        "display_x": click["x"],
        "display_y": click["y"],
        "value": None,
    })
    st.rerun()

if not st.session_state["points"]:
    st.info("No points marked yet — click on the image where a scale number is printed.")

cols = st.columns(3)
for i, p in enumerate(st.session_state["points"]):
    with cols[i % 3]:
        p["value"] = st.number_input(f"Point {i + 1} shows:", value=p["value"], key=f"val_{i}", format="%g")

filled_points = [p for p in st.session_state["points"] if p["value"] is not None]

if 0 < len(filled_points) < 3:
    st.warning(
        "For a reliable reading, mark at least 3 points spread around the dial. "
        "Calibrating from only 2 points can amplify small position errors into large reading errors."
    )

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Clear points"):
        st.session_state["points"] = []
        st.rerun()
with col2:
    read_clicked = st.button("Read gauge", type="primary", disabled=len(filled_points) < 2)

if read_clicked:
    calibration = fit_scale_calibration(circle.center_x, circle.center_y, filled_points)
    yolo_model, refiner_model = load_models()

    with st.spinner("Reading gauge..."):
        result = read_gauge_ensemble(tmp_path, calibration, yolo_model, refiner_model)

    if result.status == "ok":
        st.metric("Reading", f"{result.reading:.3f}")
        source_labels = {
            "agreement": "Classical and DL agreed",
            "classical_confident": "Classical (high confidence, systems disagreed)",
            "yolo_refiner_confident": "DL ensemble (classical uncertain, systems disagreed)",
            "classical_only": "Classical only (DL detection failed)",
            "yolo_refiner_only": "DL ensemble only (classical detection failed)",
        }
        st.caption(f"Source: {source_labels.get(result.source, result.source)}")
        if result.confidence is not None:
            st.progress(min(1.0, result.confidence), text=f"Confidence: {result.confidence:.0%}")
    else:
        reason_labels = {
            "gauge_not_detected": "Could not locate the gauge in this image.",
            "both_systems_failed": "Neither the classical nor the DL system could produce a reading.",
        }
        st.warning(f"Unable to read: {reason_labels.get(result.reason, result.reason)}")