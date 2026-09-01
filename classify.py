from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-cls.pt")
image = cv2.imread("images/bus.jpg")

result = model(image, verbose=False)[0]
print("Top predictions:")
for idx in result.probs.top5:
    print(model.names[int(idx)], float(result.probs.data[int(idx)]))

annotated = result.plot()
cv2.imshow("YOLO11 Classification", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
