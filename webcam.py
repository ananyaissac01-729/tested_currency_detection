import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    frame = results[0].plot()

    cv2.imshow("Currency Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
