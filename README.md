# 🎯 YOLO Vision Studio

A Python-based computer vision application built using **YOLO11, Ultralytics, OpenCV, and Streamlit**.

YOLO Vision Studio provides a simple and interactive browser-based interface for performing multiple computer vision tasks on images and videos.

Demo : https://yolo-vision-studio-lavanyasaxena.streamlit.app/

## ✨ Features

- 🔍 Object Detection
- 🏷️ Image Classification
- 🧍 Pose Estimation
- 🎭 Instance Segmentation
- 📐 Oriented Bounding Box (OBB) Detection
- 🖼️ Image upload and processing
- 🎥 Video upload and processing
- 🎚️ Adjustable confidence threshold
- 📊 Detection statistics
- 👁️ Video processing preview
- ⬇️ Download processed videos
- 🌐 Browser-based Streamlit interface
- 🤖 Automatic YOLO model downloading

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Ultralytics YOLO | Computer vision and model inference |
| OpenCV | Image and video processing |
| Streamlit | Interactive web interface |
| NumPy | Numerical processing |
| Pillow | Image handling |

---

## 📁 Project Structure

```text
YOLO-Python/
│
├── app.py                  # Main Streamlit application
├── detect.py               # Object detection script
├── webcam.py               # Real-time webcam detection
├── classify.py             # Image classification
├── pose.py                 # Pose estimation
├── segment.py              # Image segmentation
├── obb.py                  # Oriented Bounding Box detection
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── models/                 # Custom YOLO models
├── images/                 # Input images
└── outputs/                # Output files

## 1. Installation

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

## 2. Object detection

The first run automatically downloads `yolo11n.pt`.

```powershell
python detect.py
```

## 3. Webcam

```powershell
python webcam.py
```

Press **Q** to quit.

## 4. Other modes

```powershell
python pose.py
python segment.py
python classify.py
python obb.py
```

The corresponding YOLO11 model weights are downloaded automatically.

## 5. Custom image/video

```powershell
python detect.py --model yolo11n.pt --source "images/my_image.jpg" --save
```

Video:

```powershell
python detect.py --model yolo11n.pt --source "video.mp4" --save
```

## Models

- Detection: `yolo11n.pt`
- Classification: `yolo11n-cls.pt`
- Pose: `yolo11n-pose.pt`
- Segmentation: `yolo11n-seg.pt`
- OBB: `yolo11n-obb.pt`

You can replace the model with a custom trained `.pt` file.
