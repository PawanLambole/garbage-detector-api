from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMG_SIZE = 224

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model (will be loaded when the app starts)
model = None

# Class labels - Update these based on your dataset
CLASS_LABELS = [
    'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'
]  # Replace with your actual class labels

def load_model():
    """Load the trained model"""
    global model
    try:
        # Try loading different model formats
        if os.path.exists('model.keras'):
            model = tf.keras.models.load_model('model.keras')
            print("✅ Model loaded from model.keras")
        elif os.path.exists('model.h5'):
            model = tf.keras.models.load_model('model.h5')
            print("✅ Model loaded from model.h5")
        else:
            print("❌ No model file found!")
            return False
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'message': '🚮 Garbage Classification API is running!',
        'status': 'healthy',
        'model_loaded': model is not None,
        'endpoints': {
            'predict_file': '/predict (POST with file)',
            'predict_base64': '/predict/base64 (POST with base64 image)',
            'health': '/ (GET)'
        }
    })

@app.route('/predict', methods=['POST'])
def predict_file():
    """Predict garbage type from uploaded file"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400
        
        # Read and process image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class label
        predicted_class = CLASS_LABELS[predicted_class_idx] if predicted_class_idx < len(CLASS_LABELS) else f"Class_{predicted_class_idx}"
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            class_name = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else f"Class_{idx}"
            top_3_predictions.append({
                'class': class_name,
                'confidence': float(predictions[0][idx])
            })
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_3_predictions': top_3_predictions,
            'filename': secure_filename(file.filename)
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    """Predict garbage type from base64 encoded image"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No base64 image provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process and predict
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class label
        predicted_class = CLASS_LABELS[predicted_class_idx] if predicted_class_idx < len(CLASS_LABELS) else f"Class_{predicted_class_idx}"
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            class_name = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else f"Class_{idx}"
            top_3_predictions.append({
                'class': class_name,
                'confidence': float(predictions[0][idx])
            })
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_3_predictions': top_3_predictions
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tensorflow_version': tf.__version__,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'image_size': IMG_SIZE,
        'classes': CLASS_LABELS
    })

if __name__ == '__main__':
    print("🚀 Starting Garbage Classification API...")
    
    # Load model on startup
    if load_model():
        print("✅ Model loaded successfully!")
    else:
        print("❌ Failed to load model!")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
