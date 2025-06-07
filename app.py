from flask import Flask, request, jsonify
from PIL import Image
import io
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('model.keras')
class_names = ['Plastic', 'Metal', 'Paper', 'Cardbord', 'Glass']  # Your classes

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    image = Image.open(file.stream).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]

    return jsonify({'class': predicted_class})
