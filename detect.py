from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="YOLO Python inference")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model")
    parser.add_argument("--source", default="images/bus.jpg", help="Image/video path or webcam index")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)

    source = int(args.source) if args.source.isdigit() else args.source
    results = model.predict(source=source, conf=args.conf, show=True, save=args.save)

    if not isinstance(source, int):
        print(f"Processed: {source}")
        print("Detections:")
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"  {model.names[cls]}: {conf:.2f}")

if __name__ == "__main__":
    main()
