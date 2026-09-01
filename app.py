import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="YOLO Vision Studio",
    page_icon="🎯",
    layout="wide",
)

MODEL_MAP = {
    "Detection": "yolo11n.pt",
    "Classification": "yolo11n-cls.pt",
    "Pose": "yolo11n-pose.pt",
    "Segmentation": "yolo11n-seg.pt",
    "OBB": "yolo11n-obb.pt",
}

st.title("🎯 YOLO Vision Studio")
st.caption("YOLO11 Detection • Classification • Pose • Segmentation • OBB")

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name

def show_detection_stats(result, model):
    c1, c2, c3 = st.columns(3)
    if result.boxes is not None:
        count = len(result.boxes)
        c1.metric("Detections", count)

        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        c2.metric("Avg. Confidence", f"{np.mean(confidences):.2f}" if len(confidences) else "—")

        if result.boxes.cls is not None:
            labels = [model.names[int(x)] for x in result.boxes.cls.cpu().numpy()]
            c3.metric("Classes", len(set(labels)))
            if labels:
                st.write("**Detected classes:**", ", ".join(sorted(set(labels))))
    else:
        c1.metric("Detections", 0)

def process_image(file_path, mode, conf):
    model = load_model(MODEL_MAP[mode])
    image = cv2.imread(file_path)

    if image is None:
        st.error("Could not read the uploaded image.")
        return

    with st.spinner(f"Running {mode.lower()}..."):
        result = model.predict(image, conf=conf, verbose=False)[0]

    if mode == "Classification":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb, caption="Input image", use_container_width=True)
        with col2:
            st.subheader("Predictions")
            if result.probs is not None:
                top5 = result.probs.top5
                for rank, idx in enumerate(top5, 1):
                    name = model.names[int(idx)]
                    probability = float(result.probs.data[int(idx)])
                    st.write(f"**{rank}. {name}** — {probability:.2%}")
                    st.progress(probability)
            else:
                st.info("No classification probabilities returned.")
        return

    annotated = result.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 caption="Input image", use_container_width=True)
    with col2:
        st.image(annotated, caption=f"{mode} result", use_container_width=True)

    st.subheader("Results")
    show_detection_stats(result, model)

def process_video(file_path, mode, conf):
    model = load_model(MODEL_MAP[mode])

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # mp4v is widely available through OpenCV; browser playback may depend
    # on the local OpenCV/FFmpeg build.
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    progress = st.progress(0)
    preview = st.empty()
    processed = 0

    with st.spinner(f"Processing video with {mode.lower()}..."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model.predict(frame, conf=conf, verbose=False)[0]

            if mode == "Classification":
                annotated = frame
                if result.probs is not None:
                    top_idx = int(result.probs.top1)
                    label = model.names[top_idx]
                    probability = float(result.probs.top1conf)
                    cv2.putText(
                        annotated,
                        f"{label}: {probability:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
            else:
                annotated = result.plot()

            writer.write(annotated)
            processed += 1

            if processed % 5 == 0:
                preview.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    caption="Live processing preview",
                    use_container_width=True,
                )

            if total > 0:
                progress.progress(min(processed / total, 1.0))

    cap.release()
    writer.release()
    progress.progress(1.0)

    st.success("Video processing complete.")

    # Let Streamlit/browser attempt to render the resulting video.
    with open(output_path, "rb") as f:
        video_bytes = f.read()

    st.video(video_bytes)

    st.download_button(
        "⬇️ Download processed video",
        data=video_bytes,
        file_name=f"yolo_{mode.lower()}_result.mp4",
        mime="video/mp4",
    )

def main():
    with st.sidebar:
        st.header("⚙️ Settings")

        mode = st.selectbox(
            "Task",
            list(MODEL_MAP.keys()),
            help="Choose the YOLO vision task.",
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
            f"Model: `{MODEL_MAP[mode]}`\n\n"
            "Models download automatically on first use."
        )

    uploaded = st.file_uploader(
        f"Upload a {source_type.lower()}",
        type=["jpg", "jpeg", "png", "webp"] if source_type == "Image"
             else ["mp4", "avi", "mov", "mkv"],
    )

    if uploaded is None:
        st.markdown(
            """
            ### 🚀 Get started

            1. Select a task from the sidebar.
            2. Choose **Image** or **Video**.
            3. Upload your file.
            4. View the YOLO11 result directly in the browser.

            **Supported tasks:** Detection, Classification, Pose,
            Segmentation and OBB.
            """
        )
        return

    file_path = save_uploaded_file(uploaded)

    if source_type == "Image":
        process_image(file_path, mode, confidence)
    else:
        process_video(file_path, mode, confidence)

    try:
        os.unlink(file_path)
    except OSError:
        pass

if __name__ == "__main__":
    main()
