from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io

app = Flask(__name__)

# ✅ Configuration
MODEL_PATH = "model.keras"
IMG_SIZE = 96
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# ✅ Load the model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)
    raise

@app.route('/')
def index():
    return "🎉 Garbage Detector API is running. Use POST /predict with an image file."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        print(f"📦 Received file: {file.filename}")

        # Read and preprocess image
        img_bytes = file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

        print("🧪 Image preprocessed")

        # Predict
        preds = model.predict(img_array)
        predicted_index = int(np.argmax(preds[0]))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(preds[0][predicted_index])

        print(f"✅ Prediction: {predicted_class} ({confidence:.4f})")

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
