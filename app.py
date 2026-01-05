# app.py
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

APP_TITLE = "VisionAssist - YOLO + CVD Filter (Pure RGB)"
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
            "Put best.pt in the repo root."
        )
    return YOLO(MODEL_PATH)


# -----------------------------
# Utilities
# -----------------------------
def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def summarize_detections(detections: list[dict]) -> str:
    if not detections:
        return "No objects detected."
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return "Detected " + ", ".join(f"{v} {k}" for k, v in counts.items()) + "."


def make_tts_mp3(text: str) -> bytes | None:
    """gTTS requires internet."""
    try:
        from gtts import gTTS  # pip install gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def swatch_image(rgb: tuple[int, int, int], size: int = 70) -> Image.Image:
    return Image.new("RGB", (size, size), rgb)


# -----------------------------
# RAW COLOR DETECTION (RGB only, no HSV)
# -----------------------------
def dominant_color_from_rgb(raw_rgb: np.ndarray) -> tuple[str, tuple[int, int, int]]:
    """
    Simple color naming using ONLY RGB (no HSV).
    Detects: Red, Green, Blue, Yellow, Cyan, Magenta, White, Black, Gray.
    """
    avg = raw_rgb.reshape(-1, 3).mean(axis=0)
    r, g, b = [float(x) for x in avg]

    # brightness
    v = (r + g + b) / 3.0
    mx = max(r, g, b)
    mn = min(r, g, b)

    # gray / black / white if low chroma
    if (mx - mn) < 18:
        if v < 50:
            name = "Black"
        elif v > 210:
            name = "White"
        else:
            name = "Gray"
        return name, (int(r), int(g), int(b))

    # secondary colors (tune thresholds if needed)
    high = 160
    low = 190  # higher to catch "yellow-ish" bananas etc.

    if r > high and g > high and b < low:
        name = "Yellow"
    elif g > high and b > high and r < low:
        name = "Cyan"
    elif r > high and b > high and g < low:
        name = "Magenta"
    else:
        # primary channel dominance
        if r >= g and r >= b:
            name = "Red"
        elif g >= r and g >= b:
            name = "Green"
        else:
            name = "Blue"

    return name, (int(r), int(g), int(b))


# -----------------------------
# CVD FILTER (LMS missing cone, RGB only)
# -----------------------------
RGB_TO_LMS = np.array([
    [0.31399, 0.63951, 0.04650],
    [0.15537, 0.75789, 0.08670],
    [0.01775, 0.10944, 0.87257],
], dtype=np.float32)

LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS).astype(np.float32)

LMS_MISSING = {
    "None": np.eye(3, dtype=np.float32),
    "Protanopia": np.array([[0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 1]], dtype=np.float32),
    "Deuteranopia": np.array([[1, 0, 0],
                              [1, 0, 0],
                              [0, 0, 1]], dtype=np.float32),
    "Tritanopia": np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0]], dtype=np.float32),
}


def apply_cvd(rgb: np.ndarray, cvd_type: str) -> np.ndarray:
    M = LMS_MISSING.get(cvd_type, LMS_MISSING["None"])
    x = rgb.astype(np.float32) / 255.0
    lms = x @ RGB_TO_LMS.T
    lms = lms @ M.T
    rgb2 = lms @ LMS_TO_RGB.T
    return (np.clip(rgb2, 0, 1) * 255).astype(np.uint8)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.session_state.setdefault("apply_cvd", False)
st.session_state.setdefault("play_audio", False)

with st.sidebar:
    source = st.radio("Input Source", ["Upload Image", "Live Camera"])
    cvd_type = st.selectbox("CVD Type", ["None", "Protanopia", "Deuteranopia", "Tritanopia"])
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou_threshold = st.slider("IoU threshold", 0.0, 1.0, 0.45, 0.01)

# Load model
model = load_model()

# -----------------------------
# Input
# -----------------------------
image_name = "camera.png"
image_pil: Image.Image

if source == "Upload Image":
    uploaded = st.file_uploader("Upload image", type=[e[1:] for e in ALLOWED_EXTS])
    if not uploaded:
        st.stop()
    if not is_allowed(uploaded.name):
        st.error(f"Unsupported file. Allowed: {sorted(ALLOWED_EXTS)}")
        st.stop()
    image_name = uploaded.name
    image_pil = Image.open(uploaded).convert("RGB")
else:
    cam = st.camera_input("Capture")
    if not cam:
        st.stop()
    image_name = "camera.png"
    image_pil = Image.open(cam).convert("RGB")

# Raw RGB array for our own processing
raw_rgb = np.array(image_pil)

# -----------------------------
# Raw color detection (ALWAYS raw)
# -----------------------------
color_name, avg_rgb = dominant_color_from_rgb(raw_rgb)
st.image(swatch_image(avg_rgb), width=80, caption=color_name)
st.write(f"Average RGB: {avg_rgb}")

# -----------------------------
# YOLO inference (PIL RGB to avoid BGR issues)
# -----------------------------
with st.spinner("Running YOLO inference..."):
    results = model.predict(
        source=image_pil,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

annotated_rgb = np.array(results[0].plot(pil=True))  # RGB

# Detections
detections: list[dict] = []
for b in results[0].boxes:
    detections.append({
        "box": b.xyxy[0].tolist(),
        "confidence": float(b.conf[0]),
        "class_id": int(b.cls[0]),
        "class_name": model.names[int(b.cls[0])]
    })

# -----------------------------
# Filtered images (do not replace originals)
# -----------------------------
filtered_original = apply_cvd(raw_rgb, cvd_type) if cvd_type != "None" else raw_rgb
filtered_annotated = apply_cvd(annotated_rgb, cvd_type) if cvd_type != "None" else annotated_rgb

# What to download as "displayed image"
if st.session_state.apply_cvd and cvd_type != "None":
    display_rgb = filtered_annotated
else:
    display_rgb = annotated_rgb

# -----------------------------
# DISPLAY (3 columns)
# -----------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Original (RAW)")
    st.image(raw_rgb, use_container_width=True)

with c2:
    st.subheader("Result (Annotated - RAW)")
    st.image(annotated_rgb, use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Filter", use_container_width=True):
            st.session_state.apply_cvd = not st.session_state.apply_cvd
            st.rerun()
    with b2:
        if st.button("Audio", use_container_width=True):
            st.session_state.play_audio = not st.session_state.play_audio
            st.rerun()

with c3:
    st.subheader("Filtered Images (CVD)")
    if st.session_state.apply_cvd and cvd_type != "None":
        st.image(filtered_original, caption=f"Filtered Original ({cvd_type})", use_container_width=True)
        st.image(filtered_annotated, caption=f"Filtered Annotated ({cvd_type})", use_container_width=True)
    elif st.session_state.apply_cvd and cvd_type == "None":
        st.info("Filter is ON, but CVD Type is None. Select a CVD type.")
    else:
        st.info("Press Filter to view CVD images")

# -----------------------------
# Audio
# -----------------------------
if st.session_state.play_audio:
    st.subheader("Audio")
    summary = summarize_detections(detections)
    st.write(summary)
    mp3 = make_tts_mp3(summary)
    if mp3:
        st.audio(mp3, format="audio/mp3")
    else:
        st.warning("Audio not available (gTTS not installed or no internet).")

# -----------------------------
# JSON output
# -----------------------------
st.subheader("Detections (JSON)")
st.json(detections)

# -----------------------------
# Download buttons
# -----------------------------
st.subheader("Download")

st.download_button(
    label="Download displayed image (PNG)",
    data=pil_to_bytes(Image.fromarray(display_rgb), fmt="PNG"),
    file_name=f"result_{Path(image_name).stem}.png",
    mime="image/png",
)

st.download_button(
    label="Download detections (TXT/JSON-like)",
    data=str(detections).encode("utf-8"),
    file_name=f"detections_{Path(image_name).stem}.txt",
    mime="text/plain",
)
