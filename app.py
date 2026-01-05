import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2  # pip install opencv-python-headless

APP_TITLE = "VisionAssist - YOLO + CVD Filter Toggle + Raw Color"
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
# CVD Filter (LMS "missing cone" simulation)
# Applies ONLY when user presses Filter button (for DISPLAY).
# -----------------------------
RGB_TO_LMS = np.array([
    [0.31399022, 0.63951294, 0.04649755],
    [0.15537241, 0.75789446, 0.08670142],
    [0.01775239, 0.10944209, 0.87256922],
], dtype=np.float32)
LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS).astype(np.float32)

LMS_MISSING_CONE = {
    "None": np.eye(3, dtype=np.float32),

    # Protanopia: missing L -> approximate L from M (simple model)
    "Protanopia": np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32),

    # Deuteranopia: missing M -> approximate M from L (simple model)
    "Deuteranopia": np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32),

    # Tritanopia: missing S -> approximate S from M (simple model)
    "Tritanopia": np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32),
}


def apply_cvd_lms_missing_cone(rgb: np.ndarray, cvd_type: str) -> np.ndarray:
    """
    Apply LMS missing-cone simulation to an RGB uint8 image.
    """
    M = LMS_MISSING_CONE.get(cvd_type, LMS_MISSING_CONE["None"])

    x = rgb.astype(np.float32) / 255.0  # (H,W,3) in [0,1]

    # RGB -> LMS
    lms = x @ RGB_TO_LMS.T

    # Deficiency in LMS
    lms2 = lms @ M.T

    # LMS -> RGB
    rgb2 = lms2 @ LMS_TO_RGB.T
    rgb2 = np.clip(rgb2, 0.0, 1.0)

    return (rgb2 * 255.0).astype(np.uint8)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# session defaults
st.session_state.setdefault("apply_cvd_filter", False)  # Filter button toggles this
st.session_state.setdefault("play_audio", False)

with st.sidebar:
    st.header("Input Source")
    source = st.radio("Choose input", ["Upload Image", "Live Camera"], index=0)

    st.header("CVD Type (applies ONLY when Filter is ON)")
    cvd_type = st.selectbox(
        "Select CVD type",
        ["None", "Protanopia", "Deuteranopia", "Tritanopia"],
        index=0
    )

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

# ✅ Raw color detection ALWAYS from raw_rgb only (never filtered)
color_name, avg_rgb = dominant_color_name_from_rgb(raw_rgb)

st.subheader("Raw Image Color Detection (Original image - ALWAYS raw)")
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

# Precompute filtered images (CVD versions) for the 3rd column
filtered_original_rgb = raw_rgb
filtered_result_rgb = annotated_rgb
if cvd_type != "None":
    filtered_original_rgb = apply_cvd_lms_missing_cone(raw_rgb, cvd_type)
    filtered_result_rgb = apply_cvd_lms_missing_cone(annotated_rgb, cvd_type)

# -----------------------------
# DISPLAY: 3 columns
# - col1: raw original
# - col2: raw annotated
# - col3: filtered (two stacked images) only when Filter ON
# -----------------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.subheader("Original (RAW)")
    st.image(Image.fromarray(raw_rgb), use_container_width=True)

with col2:
    st.subheader("Result (Annotated - RAW)")
    st.image(Image.fromarray(annotated_rgb), use_container_width=True)

    # Buttons under column 2 (like your screenshot layout)
    bA, bB = st.columns(2)
    with bA:
        if st.button("Filter", key="btn_filter", use_container_width=True):
            st.session_state.apply_cvd_filter = not st.session_state.apply_cvd_filter
            st.session_state.play_audio = False
            st.rerun()
    with bB:
        if st.button("Audio", key="btn_audio", use_container_width=True):
            st.session_state.play_audio = not st.session_state.play_audio
            # optional: turn off filter when audio is on
            st.session_state.apply_cvd_filter = False
            st.rerun()

    if st.session_state.play_audio:
        st.markdown("### Audio")
        summary = summarize_detections(detections)
        st.write(summary)
        mp3 = make_tts_mp3(summary)
        if mp3 is None:
            st.warning("Audio not available. Install gTTS: `pip install gTTS` (needs internet).")
        else:
            st.audio(mp3, format="audio/mp3")

with col3:
    st.subheader("Filtered View (CVD)")

    if st.session_state.apply_cvd_filter and cvd_type != "None":
        st.caption(f"Filtered Original ({cvd_type})")
        st.image(Image.fromarray(filtered_original_rgb), use_container_width=True)

        st.caption(f"Filtered Annotated ({cvd_type})")
        st.image(Image.fromarray(filtered_result_rgb), use_container_width=True)
    elif st.session_state.apply_cvd_filter and cvd_type == "None":
        st.info("Filter is ON, but CVD type is **None**. Choose Protanopia/Deuteranopia/Tritanopia.")
    else:
        st.info("Press **Filter** to show the filtered images here.")

st.subheader("Detections (JSON)")
st.json(detections)

st.subheader("Download")

# Download what is being shown in the filtered panel if Filter ON, else download raw annotated
download_rgb = filtered_result_rgb if (st.session_state.apply_cvd_filter and cvd_type != "None") else annotated_rgb

st.download_button(
    label="Download displayed result image (PNG)",
    data=pil_to_bytes(Image.fromarray(download_rgb), fmt="PNG"),
    file_name=f"result_{Path(image_name).stem}.png",
    mime="image/png",
)

st.download_button(
    label="Download detections (TXT/JSON-like)",
    data=str(detections).encode("utf-8"),
    file_name=f"detections_{Path(image_name).stem}.txt",
    mime="text/plain",
)
