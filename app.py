from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import gc
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
IMG_SIZE = 224
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels - UPDATE THESE WITH YOUR ACTUAL CLASSES
CLASS_LABELS = [
    'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'
]  # Replace with your actual class labels from training

class ModelLoader:
    """Memory-efficient model loader that supports both Keras and TFLite models"""
    
    def __init__(self):
        self.model = None
        self.interpreter = None
        self.model_type = None
        self.input_details = None
        self.output_details = None
        self.model_loaded = False
    
    def load_model(self):
        """Try to load model in order of preference: TFLite -> Keras -> H5"""
        model_files = [
            ('model_optimized.tflite', 'tflite'),
            ('model_quantized.tflite', 'tflite'),
            ('model.tflite', 'tflite'),
            ('model.keras', 'keras'),
            ('model.h5', 'h5')
        ]
        
        for model_file, model_format in model_files:
            if os.path.exists(model_file):
                try:
                    logger.info(f"Attempting to load {model_file} ({model_format})")
                    
                    if model_format == 'tflite':
                        self._load_tflite_model(model_file)
                    else:
                        self._load_keras_model(model_file)
                    
                    self.model_type = model_format
                    self.model_loaded = True
                    logger.info(f"✅ Successfully loaded {model_file}")
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_file}: {str(e)}")
                    continue
        
        logger.error("❌ No compatible model file found!")
        return False
    
    def _load_tflite_model(self, model_path):
        """Load TFLite model"""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Log model info
        input_shape = self.input_details[0]['shape']
        logger.info(f"TFLite model input shape: {input_shape}")
    
    def _load_keras_model(self, model_path):
        """Load Keras/H5 model"""
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Keras model input shape: {self.model.input_shape}")
    
    def predict(self, input_data):
        """Make prediction using loaded model"""
        if not self.model_loaded:
            raise Exception("No model loaded")
        
        try:
            if self.model_type == 'tflite':
                return self._predict_tflite(input_data)
            else:
                return self._predict_keras(input_data)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def _predict_tflite(self, input_data):
        """Predict using TFLite model"""
        # Ensure input data is float32
        input_data = input_data.astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
    
    def _predict_keras(self, input_data):
        """Predict using Keras model"""
        return self.model.predict(input_data, verbose=0)
    
    def get_model_info(self):
        """Get model information"""
        if not self.model_loaded:
            return {"loaded": False}
        
        info = {
            "loaded": True,
            "type": self.model_type,
            "num_classes": len(CLASS_LABELS)
        }
        
        if self.model_type == 'tflite':
            info["input_shape"] = self.input_details[0]['shape'].tolist()
            info["input_dtype"] = str(self.input_details[0]['dtype'])
        else:
            info["input_shape"] = self.model.input_shape
        
        return info

# Initialize model loader
model_loader = ModelLoader()

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
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def get_predictions_info(predictions):
    """Extract prediction information"""
    try:
        # Handle both 2D and 1D prediction arrays
        if len(predictions.shape) > 1:
            pred_array = predictions[0]
        else:
            pred_array = predictions
        
        predicted_class_idx = np.argmax(pred_array)
        confidence = float(pred_array[predicted_class_idx])
        
        # Get class label
        if predicted_class_idx < len(CLASS_LABELS):
            predicted_class = CLASS_LABELS[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        
        # Get top 3 predictions
        top_3_indices = np.argsort(pred_array)[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            class_name = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else f"Class_{idx}"
            top_3_predictions.append({
                'class': class_name,
                'confidence': float(pred_array[idx]),
                'class_index': int(idx)
            })
        
        return predicted_class, confidence, top_3_predictions
        
    except Exception as e:
        logger.error(f"Error extracting predictions: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    model_info = model_loader.get_model_info()
    
    return jsonify({
        'message': '🚮 Garbage Classification API is running!',
        'status': 'healthy',
        'model_info': model_info,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),
        'endpoints': {
            'predict_file': '/predict (POST with file)',
            'predict_base64': '/predict/base64 (POST with base64 image)',
            'health': '/health (GET)',
            'model_info': '/model-info (GET)'
        },
        'classes': CLASS_LABELS
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    model_info = model_loader.get_model_info()
    
    return jsonify({
        'status': 'healthy',
        'model_info': model_info,
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'image_size': IMG_SIZE,
        'classes': CLASS_LABELS,
        'environment': {
            'python_version': os.sys.version.split()[0],
            'platform': os.name
        }
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    return jsonify(model_loader.get_model_info())

@app.route('/predict', methods=['POST'])
def predict_file():
    """Predict garbage type from uploaded file"""
    if not model_loader.model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided. Please upload a file with key "file"'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'error': f'File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)} MB'
            }), 400
        
        # Read and process image
        try:
            image = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Make prediction
        predictions = model_loader.predict(processed_image)
        predicted_class, confidence, top_3_predictions = get_predictions_info(predictions)
        
        # Clean up memory
        del processed_image, image
        gc.collect()
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_3_predictions': top_3_predictions,
            'filename': secure_filename(file.filename),
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'model_type': model_loader.model_type
        })
        
    except Exception as e:
        # Clean up memory on error
        gc.collect()
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    """Predict garbage type from base64 encoded image"""
    if not model_loader.model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No base64 image provided. Expected JSON: {"image": "base64_string"}'
            }), 400
        
        # Decode base64 image
        image_data = data['image']
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        try:
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image: {str(e)}'}), 400
        
        # Process and predict
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Make prediction
        predictions = model_loader.predict(processed_image)
        predicted_class, confidence, top_3_predictions = get_predictions_info(predictions)
        
        # Clean up memory
        del processed_image, image, image_bytes
        gc.collect()
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_3_predictions': top_3_predictions,
            'model_type': model_loader.model_type
        })
        
    except Exception as e:
        # Clean up memory on error
        gc.collect()
        logger.error(f"Base64 prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize model on startup
def initialize_app():
    """Initialize the application"""
    logger.info("🚀 Starting Garbage Classification API...")
    
    # Try to load model
    if model_loader.load_model():
        logger.info("✅ Model loaded successfully!")
        model_info = model_loader.get_model_info()
        logger.info(f"Model type: {model_info.get('type', 'unknown')}")
        logger.info(f"Number of classes: {len(CLASS_LABELS)}")
    else:
        logger.error("❌ Failed to load model!")
        logger.error("Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith(('.keras', '.h5', '.tflite')):
                size_mb = os.path.getsize(file) / (1024 * 1024)
                logger.error(f"  - {file} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    initialize_app()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode,
        threaded=True
    )
else:
    # For production servers like Gunicorn
    initialize_app()
