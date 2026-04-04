from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="currency.yaml",
    epochs=70,
    imgsz=640,
    batch=8,
    lr0=0.003
)
