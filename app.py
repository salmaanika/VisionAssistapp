# app.py — VisionAssist (Streamlit) in UML Class Structure (YOLO color classes)
# -----------------------------------------------------------------------------
# This version assumes your custom YOLO model (best.pt) is trained to detect
# 4 color classes: RED, GREEN, BLUE, YELLOW.
#
# Key change vs earlier versions:
# - NO HSV ROI color guessing.
# - The YOLO class name IS the detected color label.
#
# UML classes implemented:
# - UserInterface: selectCVDType(), toggleFilters(), displayOutput()
# - ColorCorrectionEngine: applyCorrection(data, type, intensity)
# - MachineLearningModule: loadModel(path), classifyColor(frame)-> detections (color labels)
# - FeedbackModule: generateTextLabel(...), generateAudio(...)
# - DataStore: savePreferences(...), loadPreferences(...), loadDataset(...)
#
# Notes for Streamlit Cloud:
# - Use Python 3.11 (important for opencv/ultralytics stability).
# - Prefer opencv-python-headless.
# - pyttsx3 often won't output browser audio on cloud; we keep it best-effort.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Optional TTS (safe fallback)
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None
    _TTS_AVAILABLE = False


# ----------------------------- DATA TYPES -----------------------------
@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    label: str  # e.g. "RED" or "RED (0.92)"


@dataclass(frozen=True)
class CorrectedFrame:
    original_rgb: np.ndarray        # HxWx3 uint8 (with overlay)
    corrected_rgb: np.ndarray       # HxWx3 uint8 (with overlay)
    summary_text: str               # for UI + audio
    detections: List[Detection]


# ----------------------------- 5.0 DATA STORE -----------------------------
class DataStore:
    """
    UML:
      - modelFiles: File
      - mappingDataset: File
      - userPreferences: File
      + savePreferences(data: String)
      + loadDataset(path: String): Map
    """

    def __init__(
        self,
        model_path_candidates: List[str],
        mapping_dataset_path: Optional[str] = None,
        preferences_path: str = "user_preferences.json",
    ):
        self.model_path_candidates = model_path_candidates
        self.mapping_dataset_path = mapping_dataset_path
        self.preferences_path = preferences_path

    def get_model_path(self) -> str:
        for p in self.model_path_candidates:
            if os.path.exists(p):
                return p
        # Return first candidate so error message is clear later
        return self.model_path_candidates[0]

    def savePreferences(self, data: str) -> None:
        with open(self.preferences_path, "w", encoding="utf-8") as f:
            f.write(data)

    def loadPreferences(self) -> Dict:
        if not os.path.exists(self.preferences_path):
            return {}
        try:
            with open(self.preferences_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def loadDataset(self, path: str) -> Dict:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


# ----------------------------- 3.0 ML MODULE -----------------------------
class MachineLearningModule:
    """
    UML:
      - mlModelPath: String
      + loadModel(path: String)
      + classifyColor(data: RawColorData): String

    Here, classifyColor returns detections where YOLO class name == color label.
    """

    def __init__(self, mlModelPath: str):
        self.mlModelPath = mlModelPath
        self._model: Optional[YOLO] = None

    @st.cache_resource
    def _cached_load_model(self, path: str) -> YOLO:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found at '{path}'. Put best.pt in repo root or models/best.pt."
            )
        return YOLO(path)

    def loadModel(self, path: str) -> None:
        self._model = self._cached_load_model(path)

    @property
    def model(self) -> YOLO:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call loadModel() first.")
        return self._model

    def classifyColor(
        self,
        frame_rgb_uint8: np.ndarray,
        conf: float = 0.5,
        imgsz: int = 640,
        include_conf_in_label: bool = False,
    ) -> List[Detection]:
        """
        Uses YOLO detection:
          - predicted class names are your colors: RED/GREEN/BLUE/YELLOW
        """
        results = self.model(frame_rgb_uint8, conf=conf, imgsz=imgsz)

        detections: List[Detection] = []
        h, w, _ = frame_rgb_uint8.shape

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else None
            confs = r.boxes.conf.cpu().numpy() if getattr(r.boxes, "conf", None) is not None else None

            if cls_ids is None:
                continue

            for i, ((x1, y1, x2, y2), cls_id) in enumerate(zip(boxes, cls_ids)):
                x1, y1 = int(max(0, x1)), int(max(0, y1))
                x2, y2 = int(min(w - 1, x2)), int(min(h - 1, y2))

                class_name = self.model.names.get(int(cls_id), "UNKNOWN") if hasattr(self.model, "names") else "UNKNOWN"

                if include_conf_in_label and confs is not None:
                    c = float(confs[i])
                    label = f"{class_name} ({c:.2f})"
                else:
                    label = f"{class_name}"

                detections.append(Detection(x1, y1, x2, y2, int(cls_id), label))

        return detections


# ----------------------------- 2.0 COLOR CORRECTION ENGINE -----------------------------
class ColorCorrectionEngine:
    """
    UML:
      + applyCorrection(data: RawColorData, type: String, intensity: Double): CorrectedData

    LMS-based daltonization.
    """

    RGB_TO_LMS = np.array([
        [0.31399022, 0.63951294, 0.04649755],
        [0.15537241, 0.75789446, 0.08670142],
        [0.01775239, 0.10944209, 0.87256922],
    ], dtype=np.float32)

    LMS_TO_RGB = np.array([
        [ 5.47221206, -4.6419601 ,  0.16963708],
        [-1.1252419 ,  2.29317094, -0.1678952 ],
        [ 0.02980165, -0.19318073,  1.16364789],
    ], dtype=np.float32)

    DEFICIENCY_MATS = {
        "Protanopia": np.array([
            [0.0, 1.05118294, -0.05116099],
            [0.0, 1.0,         0.0       ],
            [0.0, 0.0,         1.0       ],
        ], dtype=np.float32),

        "Deuteranopia": np.array([
            [1.0,         0.0, 0.0       ],
            [0.9513092,   0.0, 0.04866992],
            [0.0,         0.0, 1.0       ],
        ], dtype=np.float32),

        "Tritanopia": np.array([
            [1.0, 0.0,         0.0       ],
            [0.0, 1.0,         0.0       ],
            [-0.86744736, 1.86727089, 0.0],
        ], dtype=np.float32),
    }

    ERR2RGB = {
        "Protanopia": np.array([
            [0.0, 0.0, 0.0],
            [0.7, 1.0, 0.0],
            [0.7, 0.0, 1.0],
        ], dtype=np.float32),

        "Deuteranopia": np.array([
            [1.0, 0.7, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.7, 1.0],
        ], dtype=np.float32),

        "Tritanopia": np.array([
            [1.0, 0.0, 0.7],
            [0.0, 1.0, 0.7],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32),
    }

    @staticmethod
    def _clip01(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    def applyCorrection(self, data_rgb_uint8: np.ndarray, type: str, intensity: float) -> np.ndarray:
        if type not in self.DEFICIENCY_MATS:
            return data_rgb_uint8

        rgb = data_rgb_uint8.astype(np.float32) / 255.0
        h, w, _ = rgb.shape
        flat = rgb.reshape(-1, 3)

        lms = flat @ self.RGB_TO_LMS.T
        sim_lms = lms @ self.DEFICIENCY_MATS[type].T
        sim_rgb = self._clip01(sim_lms @ self.LMS_TO_RGB.T)

        err = flat - sim_rgb
        corr = err @ self.ERR2RGB[type].T

        out = self._clip01(flat + float(intensity) * corr)
        out_img = (out.reshape(h, w, 3) * 255).astype(np.uint8)
        return out_img


# ----------------------------- 4.0 FEEDBACK MODULE -----------------------------
class FeedbackModule:
    """
    UML:
      + generateTextLabel(label: String): TextOverlay
      + generateAudio(label: String): String

    TextOverlay => draw boxes+labels on image.
    Audio => best-effort TTS.
    """

    def __init__(self):
        self._tts_engine = None

    @st.cache_resource
    def _cached_tts_engine(self):
        if not _TTS_AVAILABLE:
            return None
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        return engine

    def generateTextLabel(self, frame_rgb_uint8: np.ndarray, detections: List[Detection]) -> np.ndarray:
        img_bgr = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2BGR)
        vis = img_bgr.copy()
        h, w, _ = vis.shape

        for d in detections:
            x1, y1 = max(0, d.x1), max(0, d.y1)
            x2, y2 = min(w - 1, d.x2), min(h - 1, d.y2)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                d.label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    def generateAudio(self, label: str) -> str:
        if not _TTS_AVAILABLE:
            return "Audio not available in this environment."

        try:
            if self._tts_engine is None:
                self._tts_engine = self._cached_tts_engine()
            if self._tts_engine is None:
                return "Audio not available in this environment."

            self._tts_engine.say(label)
            self._tts_engine.runAndWait()
            return "Audio played."
        except Exception as e:
            return f"Audio failed (ignored): {e}"


# ----------------------------- 1.0 USER INTERFACE -----------------------------
class UserInterface:
    """
    UML:
      - cameraFeedWindow: View
      - filterButtonState: Boolean
      + selectCVDType(type: String)
      + toggleFilters()
      + displayOutput(image: CorrectedFrame, label: String)
    """

    def __init__(
        self,
        data_store: DataStore,
        ml_module: MachineLearningModule,
        correction_engine: ColorCorrectionEngine,
        feedback_module: FeedbackModule,
    ):
        self.data_store = data_store
        self.ml_module = ml_module
        self.correction_engine = correction_engine
        self.feedback_module = feedback_module
        self._ensure_session_defaults()

    def _ensure_session_defaults(self):
        if "page" not in st.session_state:
            st.session_state.page = "welcome"
        if "user_name" not in st.session_state:
            st.session_state.user_name = ""
        if "cvd_type" not in st.session_state:
            st.session_state.cvd_type = "Protanopia"
        if "filterButtonState" not in st.session_state:
            st.session_state.filterButtonState = False
        if "saved_pref" not in st.session_state:
            st.session_state.saved_pref = False
        if "last_frame" not in st.session_state:
            st.session_state.last_frame = None  # Optional[CorrectedFrame]

    def selectCVDType(self, type: str):
        st.session_state.cvd_type = type

    def toggleFilters(self):
        st.session_state.filterButtonState = not bool(st.session_state.filterButtonState)

    def displayOutput(self, image: CorrectedFrame, label: str):
        st.markdown("#### Detected View (YOLO Color Labels)")
        st.image(image.original_rgb, caption="Boxes + detected color labels", use_container_width=True)

        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("FILTER"):
                st.session_state.filterButtonState = True
        with colB:
            if st.button("HIDE FILTER"):
                st.session_state.filterButtonState = False
        with colC:
            if st.button("Clear Result"):
                st.session_state.last_frame = None
                st.session_state.filterButtonState = False
                st.rerun()

        if st.session_state.filterButtonState:
            st.markdown("#### Color Corrected Image (LMS / Daltonized)")
            st.image(
                image.corrected_rgb,
                caption=f"Corrected view for {st.session_state.cvd_type}",
                use_container_width=True,
            )

        st.write("**Summary:**", label)

        if st.button("🔊 Play Audio Feedback"):
            status = self.feedback_module.generateAudio(label)
            st.info(status)

    # --------------------- UI Pages ---------------------
    def page_welcome(self):
        st.markdown("<h1 style='text-align:center;'>VisionAssist</h1>", unsafe_allow_html=True)
        st.write("Detect color regions using YOLO and apply LMS-based correction for selected CVD type.")

        name = st.text_input("Write your name:")
        if st.button("Enter"):
            st.session_state.user_name = name.strip() if name.strip() else "User"
            st.session_state.page = "onboarding"

    def page_onboarding(self):
        st.markdown(f"### Welcome {st.session_state.user_name} 👋")
        st.write(
            "- Step 1: Select your CVD type.\n"
            "- Step 2: Capture an image.\n"
            "- Step 3: YOLO detects color regions (RED/GREEN/BLUE/YELLOW).\n"
            "- Step 4: Tap **FILTER** to view corrected image.\n"
            "- Step 5: Optional audio feedback."
        )
        if st.button("Get Started"):
            st.session_state.page = "cvd_select"

    def page_cvd_select(self):
        st.markdown("### Select CVD Type")

        pref = self.data_store.loadPreferences()
        if pref.get("cvd_type") in ["Protanopia", "Deuteranopia", "Tritanopia"]:
            st.caption(f"Loaded saved preference: **{pref['cvd_type']}**")

        cvd = st.selectbox(
            "Color Vision Deficiency:",
            ["Protanopia", "Deuteranopia", "Tritanopia"],
            index=["Protanopia", "Deuteranopia", "Tritanopia"].index(st.session_state.cvd_type),
        )
        self.selectCVDType(cvd)

        st.info(f"Current selection: **{cvd}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Save Preference"):
                self.data_store.savePreferences(json.dumps({"cvd_type": cvd}, indent=2))
                st.session_state.saved_pref = True
                st.success(f"Preference saved: {cvd}")
        with col2:
            if st.button("Open Camera"):
                st.session_state.page = "camera"
        with col3:
            if st.button("⬅ Back"):
                st.session_state.page = "onboarding"

    def _process_frame(
        self,
        image_pil: Image.Image,
        cvd_type: str,
        conf: float,
        imgsz: int,
        intensity: float,
        include_conf_in_label: bool,
    ) -> CorrectedFrame:
        frame_rgb = np.array(image_pil.convert("RGB"), dtype=np.uint8)

        # 3.0 ML module: detect color regions
        detections = self.ml_module.classifyColor(
            frame_rgb,
            conf=conf,
            imgsz=imgsz,
            include_conf_in_label=include_conf_in_label,
        )

        # 4.0 Feedback module: overlay on original
        orig_with_text = self.feedback_module.generateTextLabel(frame_rgb, detections)

        # 2.0 Correction engine: apply correction to full frame
        corrected_rgb = self.correction_engine.applyCorrection(frame_rgb, type=cvd_type, intensity=float(intensity))

        # 4.0 Feedback module: overlay on corrected
        corrected_with_text = self.feedback_module.generateTextLabel(corrected_rgb, detections)

        # Summary: list detected colors
        if detections:
            # speak only class names (clean audio)
            class_names = [self.ml_module.model.names.get(d.cls_id, d.label) for d in detections]
            summary = "Detected: " + ", ".join(class_names)
        else:
            summary = "No colors detected."

        return CorrectedFrame(
            original_rgb=orig_with_text,
            corrected_rgb=corrected_with_text,
            summary_text=summary,
            detections=detections,
        )

    def page_camera(self):
        st.markdown(f"### Camera – Mode: {st.session_state.cvd_type}")
        st.write("Capture an image, run YOLO color detection, then optionally apply FILTER.")

        with st.expander("Settings", expanded=False):
            conf = st.slider("Confidence", 0.05, 0.95, 0.5, 0.05)
            imgsz = st.selectbox("Image size (imgsz)", [320, 480, 640, 800, 960], index=2)
            intensity = st.slider("Correction intensity", 0.0, 2.0, 1.0, 0.1)
            include_conf = st.checkbox("Show confidence in labels", value=False)

        img_data = st.camera_input("Tap **Take Photo** to capture")

        if img_data is not None:
            image = Image.open(img_data).convert("RGB")
            st.image(image, caption="Captured image", use_container_width=True)

            if st.button("Run Detection"):
                with st.spinner("Processing..."):
                    frame = self._process_frame(
                        image_pil=image,
                        cvd_type=st.session_state.cvd_type,
                        conf=conf,
                        imgsz=imgsz,
                        intensity=intensity,
                        include_conf_in_label=include_conf,
                    )
                    st.session_state.last_frame = frame
                    st.session_state.filterButtonState = False

        if st.session_state.last_frame is not None:
            frame: CorrectedFrame = st.session_state.last_frame
            self.displayOutput(frame, frame.summary_text)

        if st.button("⬅ Back to CVD Selection"):
            st.session_state.page = "cvd_select"
            st.session_state.last_frame = None
            st.session_state.filterButtonState = False

    # --------------------- Router ---------------------
    def run(self):
        st.sidebar.markdown("## VisionAssist")
        st.sidebar.write("Navigation")

        if st.sidebar.button("Welcome"):
            st.session_state.page = "welcome"
            st.session_state.last_frame = None
            st.session_state.filterButtonState = False

        if st.sidebar.button("CVD Selection"):
            st.session_state.page = "cvd_select"
            st.session_state.last_frame = None
            st.session_state.filterButtonState = False

        if st.sidebar.button("Camera"):
            st.session_state.page = "camera"
            st.session_state.last_frame = None
            st.session_state.filterButtonState = False

        if st.session_state.page == "welcome":
            self.page_welcome()
        elif st.session_state.page == "onboarding":
            self.page_onboarding()
        elif st.session_state.page == "cvd_select":
            self.page_cvd_select()
        elif st.session_state.page == "camera":
            self.page_camera()


# ----------------------------- APP BOOTSTRAP -----------------------------
def main():
    st.set_page_config(page_title="VisionAssist", page_icon="👁️", layout="centered")

    # 5.0 DataStore
    data_store = DataStore(
        model_path_candidates=[os.path.join("models", "best.pt"), "best.pt"],
        mapping_dataset_path=None,
        preferences_path="user_preferences.json",
    )

    # 3.0 ML module
    ml_module = MachineLearningModule(mlModelPath=data_store.get_model_path())
    try:
        ml_module.loadModel(ml_module.mlModelPath)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Debug: confirm classes (RED/GREEN/BLUE/YELLOW)
    with st.expander("Debug: Model Classes", expanded=False):
        try:
            st.write("Model path:", ml_module.mlModelPath)
            st.write("Model classes:", ml_module.model.names)
        except Exception:
            pass

    # 2.0 Color correction
    correction_engine = ColorCorrectionEngine()

    # 4.0 Feedback
    feedback_module = FeedbackModule()

    # 1.0 UI
    ui = UserInterface(
        data_store=data_store,
        ml_module=ml_module,
        correction_engine=correction_engine,
        feedback_module=feedback_module,
    )
    ui.run()


if __name__ == "__main__":
    main()
