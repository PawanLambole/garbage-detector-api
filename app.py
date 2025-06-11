from flask import Flask, request, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the trained MobileNetV2 model
MODEL_PATH = 'model.keras'  # or 'model.h5' if saved that way
model = load_model(MODEL_PATH)

# Class labels (update to match your training set)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@app.route("/", methods=["GET"])
def home():
    return Response("♻️ Garbage Classification API is Live!", mimetype='text/plain')

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return Response("Error: No file uploaded. Use key 'file' with an image.", status=400, mimetype='text/plain')

    try:
        file = request.files["file"]

        # Load and preprocess image
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))  # MobileNetV2 default input size
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_index]
        confidence = float(np.max(predictions[0]))

        return Response(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}", mimetype='text/plain')

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500, mimetype='text/plain')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
