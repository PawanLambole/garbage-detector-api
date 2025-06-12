from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Configuration
IMG_SIZE = 96  # Adjust this to match your training size
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# Global model variable
model = None

def create_model():
    """Create and return the model architecture"""
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(len(CLASS_NAMES), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

def load_model():
    """Load the trained model"""
    global model
    try:
        # Check for model.h5 file (matching your file structure)
        if os.path.exists('model.h5'):
            model = tf.keras.models.load_model('model.h5')
            print("Loaded saved model successfully from model.h5")
        else:
            # Create new model if no saved model exists
            model = create_model()
            print("Created new model - you'll need to train it or load weights")
            print("Available files:", os.listdir('.'))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model architecture...")
        model = create_model()

def preprocess_image(image):
    """Preprocess image for prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to array and normalize
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Waste Classification API is running',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES,
        'image_size': f"{IMG_SIZE}x{IMG_SIZE}",
        'version': '1.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict waste type from uploaded image"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please check if model.h5 file exists in the deployment'
            }), 500
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'message': 'Please upload an image file with key "image"'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'error': 'No image file selected',
                'message': 'Please select a valid image file'
            }), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            return jsonify({
                'error': 'Invalid file type',
                'message': f'Allowed types: {", ".join(allowed_extensions)}'
            }), 400
        
        # Read and preprocess image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all class probabilities
        class_probabilities = {
            CLASS_NAMES[i]: round(float(predictions[0][i]), 4)
            for i in range(len(CLASS_NAMES))
        }
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 4),
            'all_probabilities': class_probabilities,
            'image_processed': True
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing image'
        }), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Predict waste type from base64 encoded image"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please check if model.h5 file exists in the deployment'
            }), 500
        
        data = request.json
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No base64 image data provided',
                'message': 'Please provide base64 image data in JSON format: {"image": "base64_string"}'
            }), 400
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({
                'error': 'Invalid base64 image data',
                'message': str(e)
            }), 400
        
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all class probabilities
        class_probabilities = {
            CLASS_NAMES[i]: round(float(predictions[0][i]), 4)
            for i in range(len(CLASS_NAMES))
        }
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 4),
            'all_probabilities': class_probabilities,
            'image_processed': True
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing base64 image'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_status': 'loaded' if model is not None else 'not loaded',
        'image_size': f"{IMG_SIZE}x{IMG_SIZE}",
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'endpoints': [
            {'path': '/', 'method': 'GET', 'description': 'Basic status'},
            {'path': '/predict', 'method': 'POST', 'description': 'Upload image file for prediction'},
            {'path': '/predict_base64', 'method': 'POST', 'description': 'Send base64 image for prediction'},
            {'path': '/health', 'method': 'GET', 'description': 'Detailed health check'}
        ],
        'supported_formats': ['png', 'jpg', 'jpeg', 'gif', 'bmp']
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
