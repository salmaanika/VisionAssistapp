import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2  # pip install opencv-python-headless

APP_TITLE = "YOLO Object Detection (Streamlit)"
MODEL_PATH = "best.pt"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Put best.pt in the repo root (same folder as app.py), "
            "or change MODEL_PATH."
        )
    return YOLO(MODEL_PATH)


def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def apply_filter_to_rgb(rgb: np.ndarray, mode: str,
                        h_min=0, s_min=0, v_min=0,
                        h_max=179, s_max=255, v_max=255) -> np.ndarray:
    """Apply filter to an RGB image and return RGB image (uint8)."""
    if mode == "none":
        return rgb

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if mode == "grayscale":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    if mode == "hsv_range":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        filtered = cv2.bitwise_and(bgr, bgr, mask=mask)
        return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    return rgb


# ---- CVD simulation matrices (fast, common approximations) ----
# These are widely-used simple 3x3 transforms for full deficiency simulation.
CVD_MATRICES = {
    "None": np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32),

    "Protanopia": np.array([[0.56667, 0.43333, 0.00000],
                            [0.55833, 0.44167, 0.00000],
                            [0.00000, 0.24167, 0.75833]], dtype=np.float32),

    "Deuteranopia": np.array([[0.62500, 0.37500, 0.00000],
                              [0.70000, 0.30000, 0.00000],
                              [0.00000, 0.30000, 0.70000]], dtype=np.float32),

    "Tritanopia": np.array([[0.95000, 0.05000, 0.00000],
                            [0.00000, 0.43333, 0.56667],
                            [0.00000, 0.47500, 0.52500]], dtype=np.float32),
}


def apply_cvd_simulation(rgb: np.ndarray, cvd_type: str) -> np.ndarray:
    """
    Apply selected CVD simulation to an RGB uint8 image.
    """
    M = CVD_MATRICES.get(cvd_type, CVD_MATRICES["None"])
    x = rgb.astype(np.float32) / 255.0
    # Apply matrix per pixel: [H,W,3] dot [3,3]
    y = x @ M.T
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def summarize_detections(detections: list[dict]) -> str:
    if not detections:
        return "No objects detected."
    counts = {}
    for d in detections:
        name = d["class_name"]
        counts[name] = counts.get(name, 0) + 1
    parts = [f"{v} {k}" for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    return "Detected " + ", ".join(parts) + "."


def make_tts_mp3(text: str) -> bytes | None:
    """Optional: requires `pip install gTTS` + internet."""
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# session state
st.session_state.setdefault("show_filter", False)
st.session_state.setdefault("play_audio", False)
st.session_state.setdefault("filter_mode", "none")
st.session_state.setdefault("h_min", 0)
st.session_state.setdefault("s_min", 0)
st.session_state.setdefault("v_min", 0)
st.session_state.setdefault("h_max", 179)
st.session_state.setdefault("s_max", 255)
st.session_state.setdefault("v_max", 255)

with st.sidebar:
    st.header("Input Source")
    source = st.radio("Choose input", ["Upload Image", "Live Camera"], index=0)

    st.header("CVD Selection")
    cvd_type = st.selectbox(
        "Select CVD type",
        ["None", "Protanopia", "Deuteranopia", "Tritanopia"],
        index=0,
    )

    st.header("Detection Settings")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01, key="conf_slider")
    iou_threshold = st.slider("IoU threshold", 0.0, 1.0, 0.45, 0.01, key="iou_slider")

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

# ---- Get image from upload OR camera ----
image_pil = None
uploaded_name = "camera.png"

if source == "Upload Image":
    uploaded = st.file_uploader(
        "Upload an image",
        type=[ext.replace(".", "") for ext in sorted(ALLOWED_EXTS)],
        key="uploader",
    )
    if uploaded is None:
        st.info("Upload an image or switch to Live Camera.")
        st.stop()

    if not is_allowed(uploaded.name):
        st.error(f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTS)}")
        st.stop()

    uploaded_name = uploaded.name
    image_pil = Image.open(uploaded).convert("RGB")

else:
    cam = st.camera_input("Capture image from camera", key="camera")
    if cam is None:
        st.info("Capture an image to run detection.")
        st.stop()
    image_pil = Image.open(cam).convert("RGB")

image_np = np.array(image_pil)

# Layout
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Original")
    st.image(image_pil, use_container_width=True)

# ---- YOLO inference ----
with st.spinner("Running YOLO inference..."):
    results = model.predict(
        source=image_np,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

annotated_bgr = results[0].plot()
annotated_rgb = annotated_bgr[..., ::-1]  # BGR->RGB

# Extract detections
detections = []
for b in results[0].boxes:
    x1, y1, x2, y2 = b.xyxy[0].tolist()
    conf = float(b.conf[0].item())
    cls = int(b.cls[0].item())
    detections.append({
        "box": [x1, y1, x2, y2],
        "confidence": conf,
        "class_id": cls,
        "class_name": model.names.get(cls, str(cls)),
    })

with col2:
    st.subheader("Result (Annotated)")

    # Base display = annotated
    display_rgb = annotated_rgb

    # Apply Filter (optional)
    if st.session_state.show_filter:
        display_rgb = apply_filter_to_rgb(
            display_rgb,
            st.session_state.filter_mode,
            st.session_state.h_min,
            st.session_state.s_min,
            st.session_state.v_min,
            st.session_state.h_max,
            st.session_state.s_max,
            st.session_state.v_max,
        )

    # Apply CVD simulation ALWAYS if selected (fast)
    display_rgb = apply_cvd_simulation(display_rgb, cvd_type)

    st.image(Image.fromarray(display_rgb), use_container_width=True)

    # Buttons under result image
    cA, cB = st.columns(2)
    with cA:
        if st.button("Filter", key="btn_filter", use_container_width=True):
            st.session_state.show_filter = not st.session_state.show_filter
            st.session_state.play_audio = False
            st.rerun()
    with cB:
        if st.button("Audio", key="btn_audio", use_container_width=True):
            st.session_state.play_audio = not st.session_state.play_audio
            st.session_state.show_filter = False
            st.rerun()

    # Filter options panel
    if st.session_state.show_filter:
        st.markdown("### Filter Options (Annotated only)")
        st.session_state.filter_mode = st.selectbox(
            "Filter mode",
            ["none", "grayscale", "hsv_range"],
            index=["none", "grayscale", "hsv_range"].index(st.session_state.filter_mode),
            key="filter_mode_select",
        )

        if st.session_state.filter_mode == "hsv_range":
            x1, x2, x3 = st.columns(3)
            with x1:
                st.session_state.h_min = st.slider("H min", 0, 179, st.session_state.h_min, key="hmin")
                st.session_state.h_max = st.slider("H max", 0, 179, st.session_state.h_max, key="hmax")
            with x2:
                st.session_state.s_min = st.slider("S min", 0, 255, st.session_state.s_min, key="smin")
                st.session_state.s_max = st.slider("S max", 0, 255, st.session_state.s_max, key="smax")
            with x3:
                st.session_state.v_min = st.slider("V min", 0, 255, st.session_state.v_min, key="vmin")
                st.session_state.v_max = st.slider("V max", 0, 255, st.session_state.v_max, key="vmax")

    # Audio panel
    if st.session_state.play_audio:
        st.markdown("### Audio")
        summary = summarize_detections(detections)
        st.write(summary)
        mp3 = make_tts_mp3(summary)
        if mp3 is None:
            st.warning("Audio not available. Install gTTS: `pip install gTTS` (needs internet).")
        else:
            st.audio(mp3, format="audio/mp3")

st.subheader("Detections (JSON)")
st.json(detections)

st.subheader("Download")
download_img = Image.fromarray(display_rgb)
st.download_button(
    label="Download displayed image (PNG)",
    data=pil_to_bytes(download_img, fmt="PNG"),
    file_name=f"result_{Path(uploaded_name).stem}.png",
    mime="image/png",
    key="btn_download_img",
)

st.download_button(
    label="Download detections (TXT/JSON-like)",
    data=str(detections).encode("utf-8"),
    file_name=f"detections_{Path(uploaded_name).stem}.txt",
    mime="text/plain",
    key="btn_download_det",
)
