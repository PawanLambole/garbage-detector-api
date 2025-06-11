from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

app = Flask(__name__)
model = load_model("model.keras")

@app.route("/")
def home():
    return "✅ Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file).resize((96, 96))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return jsonify({
        "prediction": int(predicted_class),
        "confidence": float(np.max(prediction))
    })

if __name__ == "__main__":
    app.run(debug=True)
