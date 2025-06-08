from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io

app = Flask(__name__)

# Load model
MODEL_PATH = "model.keras"  # Use your correct model path
IMG_SIZE = 96  # Size used during training
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']  # Match your model's output

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    raise

@app.route('/')
def home():
    return "🌍 Garbage Classifier API running! Use POST /predict with image."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file name'}), 400

        print(f"📦 Received file: {file.filename}")

        img_bytes = file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img).astype('float32') / 255.0
        img_expanded = np.expand_dims(img_array, axis=0)

        print("🧪 Image preprocessed, running prediction...")

        predictions = model.predict(img_expanded)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        print(f"✅ Prediction: {predicted_class} (Confidence: {confidence:.4f})")

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
