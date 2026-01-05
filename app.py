import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2  # pip install opencv-python-headless

APP_TITLE = "VisionAssist - YOLO + CVD + Raw Color"
MODEL_PATH = "best.pt"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# -----------------------------
# Model
# -----------------------------
@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Put best.pt in the repo root (same folder as app.py), "
            "or change MODEL_PATH."
        )
    return YOLO(MODEL_PATH)


# -----------------------------
# Utilities
# -----------------------------
def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


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
    """gTTS requires internet."""
    try:
        from gtts import gTTS  # pip install gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


# -----------------------------
# RAW COLOR DETECTION (ALWAYS FROM ORIGINAL RAW IMAGE)
# -----------------------------
def dominant_color_name_from_rgb(raw_rgb: np.ndarray) -> tuple[str, tuple[int, int, int]]:
    """
    Dominant color detection from RAW image only:
    - no filter
    - no CVD
    """
    small = cv2.resize(raw_rgb, (220, 220), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # ignore low saturation and extreme bright/dark (background-ish)
    mask = (s > 45) & (v > 45) & (v < 245)
    pixels = small[mask]

    if pixels.size == 0:
        avg = small.reshape(-1, 3).mean(axis=0)
        r, g, b = [int(x) for x in avg]
        return ("Unknown", (r, g, b))

    avg = pixels.mean(axis=0)
    r, g, b = [int(x) for x in avg]

    avg_rgb = np.uint8([[[r, g, b]]])
    H, S, V = cv2.cvtColor(avg_rgb, cv2.COLOR_RGB2HSV)[0, 0]
    H, S, V = int(H), int(S), int(V)

    if S < 40:
        if V < 60:
            return ("Black", (r, g, b))
        if V > 200:
            return ("White", (r, g, b))
        return ("Gray", (r, g, b))

    if H < 10 or H >= 170:
        name = "Red"
    elif 10 <= H < 25:
        name = "Orange"
    elif 25 <= H < 35:
        name = "Yellow"
    elif 35 <= H < 85:
        name = "Green"
    elif 85 <= H < 105:
        name = "Cyan"
    elif 105 <= H < 130:
        name = "Blue"
    elif 130 <= H < 170:
        name = "Purple"
    else:
        name = "Unknown"

    return (name, (r, g, b))


def swatch_image(rgb_tuple: tuple[int, int, int], size=70) -> Image.Image:
    return Image.new("RGB", (size, size), rgb_tuple)


# -----------------------------
# CVD Simulation (AFFECTS ONLY DISPLAY)
# -----------------------------
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
    M = CVD_MATRICES.get(cvd_type, CVD_MATRICES["None"])
    x = rgb.astype(np.float32) / 255.0
    y = x @ M.T
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


# -----------------------------
# Filter (OPTIONAL) - ONLY ON ANNOTATED OUTPUT
# -----------------------------
def apply_filter_to_rgb(rgb: np.ndarray, mode: str,
                        h_min=0, s_min=0, v_min=0,
                        h_max=179, s_max=255, v_max=255) -> np.ndarray:
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


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# session defaults
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

    st.header("CVD Selection (Display only)")
    cvd_type = st.selectbox("Select CVD type", ["None", "Protanopia", "Deuteranopia", "Tritanopia"], index=0)

    st.header("Detection Settings")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou_threshold = st.slider("IoU threshold", 0.0, 1.0, 0.45, 0.01)

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

# Input image
image_pil = None
image_name = "camera.png"

if source == "Upload Image":
    uploaded = st.file_uploader(
        "Upload an image",
        type=[ext.replace(".", "") for ext in sorted(ALLOWED_EXTS)],
    )
    if uploaded is None:
        st.info("Upload an image or switch to Live Camera.")
        st.stop()
    if not is_allowed(uploaded.name):
        st.error(f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTS)}")
        st.stop()
    image_name = uploaded.name
    image_pil = Image.open(uploaded).convert("RGB")
else:
    cam = st.camera_input("Capture image from camera")
    if cam is None:
        st.info("Capture an image to run detection.")
        st.stop()
    image_pil = Image.open(cam).convert("RGB")

# RAW image array (used for YOLO + raw color detection)
raw_rgb = np.array(image_pil)

# ✅ Raw color detection ALWAYS from raw_rgb only
color_name, avg_rgb = dominant_color_name_from_rgb(raw_rgb)

st.subheader("Raw Image Color Detection (Original image - no filter, no CVD)")
c1, c2 = st.columns([1, 4])
with c1:
    st.image(swatch_image(avg_rgb), caption=color_name, width=80)
with c2:
    st.write(f"**Dominant Color:** {color_name}")
    st.write(f"**Average RGB:** {avg_rgb}")

# YOLO inference uses RAW image (no filter / no CVD)
with st.spinner("Running YOLO inference..."):
    results = model.predict(
        source=raw_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

annotated_bgr = results[0].plot()
annotated_rgb = annotated_bgr[..., ::-1]

# Detections
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

# Display: apply CVD to ORIGINAL and RESULT
display_original_rgb = apply_cvd_simulation(raw_rgb, cvd_type)

# Display result pipeline: annotated -> optional filter -> CVD
display_result_rgb = annotated_rgb
if st.session_state.show_filter:
    display_result_rgb = apply_filter_to_rgb(
        display_result_rgb,
        st.session_state.filter_mode,
        st.session_state.h_min,
        st.session_state.s_min,
        st.session_state.v_min,
        st.session_state.h_max,
        st.session_state.s_max,
        st.session_state.v_max,
    )
display_result_rgb = apply_cvd_simulation(display_result_rgb, cvd_type)

# Two columns
colL, colR = st.columns(2, gap="large")

with colL:
    st.subheader("Original (CVD view)")
    st.image(Image.fromarray(display_original_rgb), use_container_width=True)

with colR:
    st.subheader("Result (Annotated - CVD view)")
    st.image(Image.fromarray(display_result_rgb), use_container_width=True)

    bA, bB = st.columns(2)
    with bA:
        if st.button("Filter", key="btn_filter", use_container_width=True):
            st.session_state.show_filter = not st.session_state.show_filter
            st.session_state.play_audio = False
            st.rerun()
    with bB:
        if st.button("Audio", key="btn_audio", use_container_width=True):
            st.session_state.play_audio = not st.session_state.play_audio
            st.session_state.show_filter = False
            st.rerun()

    if st.session_state.show_filter:
        st.markdown("### Filter Options (Annotated only)")
        st.session_state.filter_mode = st.selectbox(
            "Filter mode",
            ["none", "grayscale", "hsv_range"],
            index=["none", "grayscale", "hsv_range"].index(st.session_state.filter_mode),
            key="filter_mode_select",
        )
        if st.session_state.filter_mode == "hsv_range":
            f1, f2, f3 = st.columns(3)
            with f1:
                st.session_state.h_min = st.slider("H min", 0, 179, st.session_state.h_min, key="hmin")
                st.session_state.h_max = st.slider("H max", 0, 179, st.session_state.h_max, key="hmax")
            with f2:
                st.session_state.s_min = st.slider("S min", 0, 255, st.session_state.s_min, key="smin")
                st.session_state.s_max = st.slider("S max", 0, 255, st.session_state.s_max, key="smax")
            with f3:
                st.session_state.v_min = st.slider("V min", 0, 255, st.session_state.v_min, key="vmin")
                st.session_state.v_max = st.slider("V max", 0, 255, st.session_state.v_max, key="vmax")

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
st.download_button(
    label="Download result image (PNG)",
    data=pil_to_bytes(Image.fromarray(display_result_rgb), fmt="PNG"),
    file_name=f"result_{Path(image_name).stem}.png",
    mime="image/png",
)

st.download_button(
    label="Download detections (TXT/JSON-like)",
    data=str(detections).encode("utf-8"),
    file_name=f"detections_{Path(image_name).stem}.txt",
    mime="text/plain",
)
