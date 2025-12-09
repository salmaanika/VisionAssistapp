import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="VisionAssist",
    page_icon="👁️",
    layout="centered"
)

MODEL_PATH = "best.pt"   # your trained YOLO model

# ----------------- UTILS -----------------
@st.cache_resource
def load_model(path):
    """Load YOLO model once and cache."""
    return YOLO(path)

def get_color_name_from_roi(roi_bgr: np.ndarray) -> str:
    """Return a simple color name from a BGR ROI."""
    if roi_bgr.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_mean = hsv[:, :, 0].mean()
    s_mean = hsv[:, :, 1].mean()
    v_mean = hsv[:, :, 2].mean()

    # very simple rules – tune for your dataset
    if v_mean < 50:
        return "Black"
    if s_mean < 40:
        if v_mean > 200:
            return "White"
        else:
            return "Gray"

    if (h_mean < 10) or (h_mean > 170):
        return "Red"
    elif 10 <= h_mean < 25:
        return "Orange"
    elif 25 <= h_mean < 35:
        return "Yellow"
    elif 35 <= h_mean < 85:
        return "Green"
    elif 85 <= h_mean < 130:
        return "Blue"
    elif 130 <= h_mean < 160:
        return "Purple"
    else:
        return "Pink"

def apply_cvd_filter_bgr(frame_bgr: np.ndarray, cvd_type: str) -> np.ndarray:
    """
    Very simple color transform to mimic correction for different CVD types.
    This is just for demo visuals – you can replace with your real algorithm.
    """
    frame = frame_bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(frame)

    if cvd_type == "Protanopia":
        # reduce red, boost green/blue
        R_corr = 0.3 * R + 0.7 * G
        G_corr = 0.6 * G + 0.4 * B
        B_corr = B
    elif cvd_type == "Deuteranopia":
        # reduce green
        R_corr = R
        G_corr = 0.3 * G + 0.7 * R
        B_corr = 0.8 * B + 0.2 * G
    elif cvd_type == "Tritanopia":
        # reduce blue
        R_corr = 0.8 * R + 0.2 * B
        G_corr = G
        B_corr = 0.3 * B + 0.7 * G
    else:
        return frame_bgr

    corrected = cv2.merge([
        np.clip(B_corr, 0, 1),
        np.clip(G_corr, 0, 1),
        np.clip(R_corr, 0, 1),
    ])
    corrected = (corrected * 255).astype(np.uint8)
    return corrected

def detect_and_draw(image_pil: Image.Image, model, cvd_type: str):
    """Run YOLO, draw boxes + color labels, and return 2 images:
       original_with_boxes, filtered_with_boxes
    """
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # YOLO inference (BGR or RGB both OK, but we convert to RGB for model)
    results = model(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), imgsz=640, conf=0.5)

    # Copy for drawing
    vis_orig = img_bgr.copy()
    vis_filtered = apply_cvd_filter_bgr(img_bgr.copy(), cvd_type)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else []

        for box, cls_id in zip(boxes, cls_ids):
            x1, y1, x2, y2 = box
            # Clip coordinates
            h, w, _ = img_bgr.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            roi = img_bgr[y1:y2, x1:x2]
            color_name = get_color_name_from_roi(roi)
            class_name = model.names[int(cls_id)] if hasattr(model, "names") else "obj"
            label = f"{class_name} | {color_name}"

            # Draw on original view
            cv2.rectangle(vis_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_orig, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw on filtered view (same box)
            cv2.rectangle(vis_filtered, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_filtered, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert back to PIL for Streamlit
    orig_pil = Image.fromarray(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    filt_pil = Image.fromarray(cv2.cvtColor(vis_filtered, cv2.COLOR_BGR2RGB))
    return orig_pil, filt_pil

# ----------------- STATE -----------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "cvd_type" not in st.session_state:
    st.session_state.cvd_type = "Protanopia"
if "saved_pref" not in st.session_state:
    st.session_state.saved_pref = False

# ----------------- PAGES -----------------
def page_welcome():
    st.markdown("<h1 style='text-align:center;'>VisionAssist</h1>", unsafe_allow_html=True)
    st.write("A computer vision & ML-based tool to enhance color perception "
             "for color-blind individuals with multisensory feedback.")

    name = st.text_input("Write your name:")
    if st.button("Enter"):
        st.session_state.user_name = name if name else "User"
        st.session_state.page = "onboarding"

def page_onboarding():
    st.markdown(f"### Welcome {st.session_state.user_name} 👋")
    st.write(
        "VisionAssist helps you identify objects and understand their colors.\n\n"
        "You can choose your color-vision type and capture an image to see:\n"
        "- Detected objects\n"
        "- Color labels\n"
        "- A filtered view adjusted for your color vision."
    )
    if st.button("Get Started"):
        st.session_state.page = "cvd_select"

def page_cvd_select():
    st.markdown("### Select Your Color Vision Type")
    cvd = st.selectbox(
        "Select CVD Type:",
        ["Protanopia", "Deuteranopia", "Tritanopia"],
        index=["Protanopia", "Deuteranopia", "Tritanopia"].index(st.session_state.cvd_type)
    )
    st.session_state.cvd_type = cvd

    st.info(f"Current selection: **{cvd}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Preference"):
            st.session_state.saved_pref = True
            st.success(f"Preference saved: {cvd}")
    with col2:
        if st.button("Open Live Camera"):
            st.session_state.page = "camera"

    if st.button("⬅ Back"):
        st.session_state.page = "onboarding"

def page_camera(model):
    st.markdown(f"### Live Camera – Mode: {st.session_state.cvd_type}")
    st.write("Capture an image (simulating the live camera view).")

    img_data = st.camera_input("Tap **Take Photo**")

    if img_data is not None:
        image = Image.open(img_data)

        if st.button("Run Detection"):
            with st.spinner("Running YOLO detection and color analysis..."):
                orig, filt = detect_and_draw(image, model, st.session_state.cvd_type)

            st.markdown("#### Real-Time Detected View")
            st.image(orig, caption="Detected objects with color labels", use_column_width=True)

            st.markdown("#### Filtered Image (CVD-adjusted)")
            st.image(filt, caption=f"Filtered view for {st.session_state.cvd_type}", use_column_width=True)

            st.success("Detection complete. (Here you can also trigger audio feedback in your full system.)")

    if st.button("⬅ Back to CVD Selection"):
        st.session_state.page = "cvd_select"

# ----------------- MAIN ROUTER -----------------
def main():
    # show a subtle header like in your mockup
    st.sidebar.markdown("## VisionAssist")
    st.sidebar.write("Navigation")
    if st.sidebar.button("Welcome"):
        st.session_state.page = "welcome"
    if st.sidebar.button("CVD Selection"):
        st.session_state.page = "cvd_select"
    if st.sidebar.button("Camera"):
        st.session_state.page = "camera"

    # lazy-load model only when needed
    model = None
    if st.session_state.page in ["camera"]:
        model = load_model(MODEL_PATH)

    if st.session_state.page == "welcome":
        page_welcome()
    elif st.session_state.page == "onboarding":
        page_onboarding()
    elif st.session_state.page == "cvd_select":
        page_cvd_select()
    elif st.session_state.page == "camera":
        page_camera(model)

if __name__ == "__main__":
    main()
