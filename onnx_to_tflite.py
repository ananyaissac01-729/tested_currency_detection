import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load ONNX model
onnx_model = onnx.load("best.onnx")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("tf_model")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")

# ADD THIS 👇
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created 🚀")