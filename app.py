from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the Keras model
MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")

model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return "Keras Model API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    img_file = request.files['file']
    if img_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Load and preprocess the image
    img = image.load_img(img_file, target_size=(224, 224))  # Adjust size to your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0    # Normalize if needed

    # Predict
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]

    return jsonify({'prediction': int(pred_class)})

if __name__ == '__main__':
    app.run(debug=True)
