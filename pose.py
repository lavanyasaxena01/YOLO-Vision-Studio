from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-pose.pt")
image = cv2.imread("images/bus.jpg")

result = model(image, verbose=False)[0]
annotated = result.plot()

cv2.imshow("YOLO11 Pose", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
