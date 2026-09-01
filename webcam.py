from ultralytics import YOLO
import cv2

MODEL = "yolo11n.pt"

model = YOLO(MODEL)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, verbose=False)[0]
    annotated = result.plot()

    cv2.imshow("YOLO11 Webcam - Press Q to quit", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
