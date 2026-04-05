import numpy as np
from PIL import Image
import tensorflow as tf

CLASSES = ["₹10", "₹20", "₹50", "₹100", "₹200", "₹500", "₹2000", "not_currency"]

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape : {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# preprocess image
image = Image.open("aaa.jpeg").convert("RGB").resize((640, 640))
input_data = np.array(image, dtype=np.float32) / 255.0
input_data = np.transpose(input_data, (2, 0, 1))  # HWC → CHW
input_data = np.expand_dims(input_data, axis=0)   # (1, 3, 640, 640)

# run model
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print(f"Raw output shape: {output.shape}")

# YOLOv8 output is (1, 12, 8400) → transpose to (8400, 12)
preds = output[0].T  # (8400, 12)

# filter detections — no objectness score in YOLOv8, just 4 + num_classes
filtered = []
for pred in preds:
    class_scores = pred[4:]          # 8 class scores directly
    class_id = np.argmax(class_scores)
    conf = float(class_scores[class_id])

    if conf > 0.2:
        filtered.append((class_id, conf))

# result
if filtered:
    best = max(filtered, key=lambda x: x[1])
    class_id, conf = best
    print(f"\n✓ Detected : {CLASSES[class_id]}")
    print(f"  Confidence: {conf:.2%}")
else:
    print("\n✗ No currency detected")