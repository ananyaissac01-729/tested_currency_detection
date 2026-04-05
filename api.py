from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")
CLASSES = ["₹10","₹20","₹50","₹100","₹200","₹500","₹2000","not_currency"]
MESSAGES = {
    "₹10": "Ten rupees detected",
    "₹20": "Twenty rupees detected",
    "₹50": "Fifty rupees detected",
    "₹100": "One hundred rupees detected",
    "₹200": "Two hundred rupees detected",
    "₹500": "Five hundred rupees detected",
    "₹2000": "Two thousand rupees detected",
    "not_currency": "This is not a currency note",
}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = model(image, conf=0.92)[0]

    if len(results.boxes) == 0:
        return {"detected": False, "class": "not_currency",
                "confidence": 0.0, "message": "No currency detected"}

    best_box = max(results.boxes, key=lambda b: b.conf.item())
    class_id = int(best_box.cls.item())
    confidence = float(best_box.conf.item())
    label = CLASSES[class_id]

    return {"detected": class_id != 7, "class": label,
            "confidence": confidence, "message": MESSAGES[label]}

@app.get("/")
def health():
    return {"status": "API running ✅"}