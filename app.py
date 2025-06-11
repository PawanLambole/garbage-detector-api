from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
model = load_model("model.keras")  # or "model.h5"

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((96, 96))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = preprocess_image(file.read())

    predictions = model.predict(img)
    return jsonify({"prediction": predictions.tolist()})
