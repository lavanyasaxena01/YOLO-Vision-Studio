import os
import gc
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YOLO Vision Studio",
    page_icon="🎯",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_MAP = {
    "Detection": "yolo11n.pt",
    "Classification": "yolo11n-cls.pt",
    "Pose": "yolo11n-pose.pt",
    "Segmentation": "yolo11n-seg.pt",
    "OBB": "yolo11n-obb.pt",
}

# Streamlit Cloud friendly limits
MAX_VIDEO_SIZE_MB = 50
MAX_IMAGE_SIZE_MB = 10

# CPU-friendly inference
IMAGE_SIZE = 640


# ============================================================
# HEADER
# ============================================================

st.title("🎯 YOLO Vision Studio")

st.caption(
    "YOLO11 Detection • Classification • Pose • Segmentation • OBB"
)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource(max_entries=1)
def load_model(model_name):
    """
    Load only ONE YOLO model at a time.

    max_entries=1 prevents multiple YOLO/PyTorch models
    from staying in memory simultaneously.
    """
    model = YOLO(model_name)
    return model


# ============================================================
# FILE HELPERS
# ============================================================

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location."""

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def cleanup_file(file_path):
    """Safely delete a temporary file."""

    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except OSError:
            pass


# ============================================================
# RESULT STATISTICS
# ============================================================

def show_detection_stats(result, model):

    c1, c2, c3 = st.columns(3)

    if result.boxes is None:
        c1.metric("Detections", 0)
        c2.metric("Avg. Confidence", "—")
        c3.metric("Classes", 0)
        return

    count = len(result.boxes)

    c1.metric(
        "Detections",
        count
    )

    # Confidence
    if result.boxes.conf is not None and len(result.boxes.conf) > 0:

        confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        avg_conf = float(np.mean(confidences))

        c2.metric(
            "Avg. Confidence",
            f"{avg_conf:.2f}"
        )

    else:
        c2.metric(
            "Avg. Confidence",
            "—"
        )

    # Classes
    if result.boxes.cls is not None:

        class_ids = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
        )

        labels = [
            model.names[int(x)]
            for x in class_ids
        ]

        unique_labels = sorted(set(labels))

        c3.metric(
            "Classes",
            len(unique_labels)
        )

        if unique_labels:
            st.write(
                "**Detected classes:**",
                ", ".join(unique_labels)
            )

    else:
        c3.metric(
            "Classes",
            0
        )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(file_path, mode, conf):

    try:

        # Load model
        with st.spinner(
            f"Loading {mode.lower()} model..."
        ):
            model = load_model(
                MODEL_MAP[mode]
            )

        # Read image
        image = cv2.imread(file_path)

        if image is None:
            st.error(
                "Could not read the uploaded image."
            )
            return

        # Inference
        with st.spinner(
            f"Running {mode.lower()}..."
        ):

            result = model.predict(
                image,
                conf=conf,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False,
            )[0]

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if mode == "Classification":

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    rgb,
                    caption="Input image",
                    width="stretch",
                )

            with col2:

                st.subheader(
                    "Predictions"
                )

                if result.probs is not None:

                    top5 = result.probs.top5

                    for rank, idx in enumerate(
                        top5,
                        1
                    ):

                        name = model.names[
                            int(idx)
                        ]

                        probability = float(
                            result.probs.data[
                                int(idx)
                            ]
                        )

                        st.write(
                            f"**{rank}. {name}** — "
                            f"{probability:.2%}"
                        )

                        st.progress(
                            probability
                        )

                else:

                    st.info(
                        "No classification probabilities returned."
                    )

            # Free result
            del result
            gc.collect()

            return

        # ----------------------------------------------------
        # DETECTION / POSE / SEGMENTATION / OBB
        # ----------------------------------------------------

        annotated = result.plot()

        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        original_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                original_rgb,
                caption="Input image",
                width="stretch",
            )

        with col2:

            st.image(
                annotated_rgb,
                caption=f"{mode} result",
                width="stretch",
            )

        st.subheader("Results")

        show_detection_stats(
            result,
            model
        )

        # Free memory
        del result
        del annotated
        del image

        gc.collect()

    except Exception as e:

        st.error(
            f"Error processing image: {e}"
        )

        st.exception(e)


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(file_path, mode, conf):

    output_path = None
    cap = None
    writer = None

    try:

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        with st.spinner(
            f"Loading {mode.lower()} model..."
        ):

            model = load_model(
                MODEL_MAP[mode]
            )

        # ----------------------------------------------------
        # OPEN VIDEO
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            file_path
        )

        if not cap.isOpened():

            st.error(
                "Could not open the uploaded video."
            )

            return

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if not fps or fps <= 0:
            fps = 25.0

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        # ----------------------------------------------------
        # PROTECT MEMORY
        # ----------------------------------------------------

        # Very large videos can consume significant memory/time.
        # Resize very large frames while keeping aspect ratio.

        MAX_DIMENSION = 1280

        scale = min(
            1.0,
            MAX_DIMENSION / max(
                width,
                height
            )
        )

        output_width = int(
            width * scale
        )

        output_height = int(
            height * scale
        )

        # Ensure dimensions are even for MP4
        output_width -= output_width % 2
        output_height -= output_height % 2

        if output_width <= 0:
            output_width = 2

        if output_height <= 0:
            output_height = 2

        # ----------------------------------------------------
        # OUTPUT VIDEO
        # ----------------------------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name

        output_file.close()

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (
                output_width,
                output_height
            ),
        )

        if not writer.isOpened():

            st.error(
                "Could not create output video."
            )

            return

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        progress = st.progress(
            0
        )

        preview = st.empty()

        status = st.empty()

        processed = 0

        # ----------------------------------------------------
        # PROCESS VIDEO
        # ----------------------------------------------------

        with st.spinner(
            f"Processing video with {mode.lower()}..."
        ):

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # Resize large videos
                if (
                    frame.shape[1] != output_width
                    or frame.shape[0] != output_height
                ):

                    frame = cv2.resize(
                        frame,
                        (
                            output_width,
                            output_height
                        ),
                        interpolation=cv2.INTER_AREA,
                    )

                # YOLO inference
                result = model.predict(
                    frame,
                    conf=conf,
                    imgsz=IMAGE_SIZE,
                    device="cpu",
                    verbose=False,
                )[0]

                # ------------------------------------------------
                # CLASSIFICATION
                # ------------------------------------------------

                if mode == "Classification":

                    annotated = frame.copy()

                    if result.probs is not None:

                        top_idx = int(
                            result.probs.top1
                        )

                        label = model.names[
                            top_idx
                        ]

                        probability = float(
                            result.probs.top1conf
                        )

                        cv2.putText(
                            annotated,
                            f"{label}: {probability:.2f}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                # ------------------------------------------------
                # OTHER TASKS
                # ------------------------------------------------

                else:

                    annotated = result.plot()

                # Write frame
                writer.write(
                    annotated
                )

                processed += 1

                # ------------------------------------------------
                # PREVIEW
                # ------------------------------------------------

                if processed % 10 == 0:

                    preview_frame = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )

                    preview.image(
                        preview_frame,
                        channels="RGB",
                        caption="Live processing preview",
                        width="stretch",
                    )

                    del preview_frame

                # ------------------------------------------------
                # PROGRESS
                # ------------------------------------------------

                if total > 0:

                    percentage = min(
                        processed / total,
                        1.0
                    )

                    progress.progress(
                        percentage
                    )

                    status.write(
                        f"Processed "
                        f"{processed:,} / "
                        f"{total:,} frames"
                    )

                # ------------------------------------------------
                # MEMORY CLEANUP
                # ------------------------------------------------

                del result
                del annotated
                del frame

                # Periodic garbage collection
                if processed % 30 == 0:

                    gc.collect()

        # ----------------------------------------------------
        # RELEASE VIDEO RESOURCES
        # ----------------------------------------------------

        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        gc.collect()

        progress.progress(1.0)

        status.success(
            f"Video processing complete — "
            f"{processed:,} frames processed."
        )

        # ----------------------------------------------------
        # DISPLAY OUTPUT
        # ----------------------------------------------------

        if (
            output_path is None
            or not os.path.exists(output_path)
        ):

            st.error(
                "Output video was not created."
            )

            return

        st.subheader(
            "Processed Video"
        )

        # Streamlit can read the video directly
        st.video(
            output_path
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        with open(
            output_path,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()

        st.download_button(
            label="⬇️ Download processed video",
            data=video_bytes,
            file_name=(
                f"yolo_"
                f"{mode.lower()}_result.mp4"
            ),
            mime="video/mp4",
            width="stretch",
        )

        del video_bytes
        gc.collect()

    except Exception as e:

        st.error(
            f"Error processing video: {e}"
        )

        st.exception(e)

    finally:

        # Always release resources
        if cap is not None:

            try:
                cap.release()
            except Exception:
                pass

        if writer is not None:

            try:
                writer.release()
            except Exception:
                pass

        gc.collect()


# ============================================================
# MAIN APP
# ============================================================

def main():

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.header(
            "⚙️ Settings"
        )

        mode = st.selectbox(
            "Task",
            list(MODEL_MAP.keys()),
            help=(
                "Choose the YOLO vision task."
            ),
        )

        source_type = st.radio(
            "Input",
            ["Image", "Video"],
            horizontal=True,
        )

        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
        )

        st.divider()

        st.info(
            f"**Model:** `{MODEL_MAP[mode]}`\n\n"
            "The selected model downloads automatically "
            "on first use.\n\n"
            "Only one YOLO model is kept in memory."
        )

        if source_type == "Video":

            st.warning(
                f"Video uploads are limited to "
                f"{MAX_VIDEO_SIZE_MB} MB."
            )

    # --------------------------------------------------------
    # FILE UPLOADER
    # --------------------------------------------------------

    uploaded = st.file_uploader(
        f"Upload a {source_type.lower()}",
        type=(
            [
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]
            if source_type == "Image"
            else [
                "mp4",
                "avi",
                "mov",
                "mkv"
            ]
        ),
    )

    # --------------------------------------------------------
    # LANDING PAGE
    # --------------------------------------------------------

    if uploaded is None:

        st.markdown(
            """
            ### 🚀 Get started

            1. Select a task from the sidebar.
            2. Choose **Image** or **Video**.
            3. Upload your file.
            4. View the YOLO11 result directly in the browser.

            ### Supported tasks

            - 🎯 Detection
            - 🏷️ Classification
            - 🧍 Pose Estimation
            - 🎨 Segmentation
            - 📦 Oriented Bounding Box (OBB)

            ### ☁️ Cloud optimized

            This version is optimized for deployment on
            Streamlit Community Cloud using CPU inference
            and single-model caching.
            """
        )

        return

    # --------------------------------------------------------
    # FILE SIZE CHECK
    # --------------------------------------------------------

    file_size_mb = (
        uploaded.size /
        (1024 * 1024)
    )

    if source_type == "Image":

        if file_size_mb > MAX_IMAGE_SIZE_MB:

            st.error(
                f"Image is too large. "
                f"Maximum allowed size is "
                f"{MAX_IMAGE_SIZE_MB} MB."
            )

            return

    else:

        if file_size_mb > MAX_VIDEO_SIZE_MB:

            st.error(
                f"Video is too large. "
                f"Maximum allowed size is "
                f"{MAX_VIDEO_SIZE_MB} MB."
            )

            return

    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    file_path = None

    try:

        with st.spinner(
            "Preparing uploaded file..."
        ):

            file_path = save_uploaded_file(
                uploaded
            )

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        if source_type == "Image":

            process_image(
                file_path,
                mode,
                confidence
            )

        else:

            process_video(
                file_path,
                mode,
                confidence
            )

    finally:

        # Delete uploaded temporary file
        cleanup_file(
            file_path
        )

        gc.collect()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
