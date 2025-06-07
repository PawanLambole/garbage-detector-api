from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.h5')  # Ensure this path is correct

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img_path = os.path.join('temp.jpg')
    file.save(img_path)

    img = image.load_img(img_path, target_size=(96, 96))  # 👈 Fix here
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)[0]
    classes = ['class1', 'class2', 'class3']  # Replace with your actual classes
    result = dict(zip(classes, map(float, predictions)))

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
