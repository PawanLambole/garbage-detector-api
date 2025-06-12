# 🔧 Model Optimization Solutions for Render Deployment

import tensorflow as tf
import os

# Solution 1: Check your current model size
def check_model_size(model_path):
    """Check model file size"""
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        return size_mb
    else:
        print(f"Model file not found: {model_path}")
        return 0

# Solution 2: Create a quantized TFLite model (Recommended)
def create_optimized_model(keras_model_path):
    """Create an optimized TFLite model"""
    print("🔄 Loading original model...")
    model = tf.keras.models.load_model(keras_model_path)
    
    print("🔄 Converting to TFLite with quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable quantization for smaller size and faster inference
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Use integer quantization for even smaller size
    # converter.representative_dataset = representative_data_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    # Save the optimized model
    with open('model_optimized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    original_size = check_model_size(keras_model_path)
    optimized_size = len(tflite_model) / (1024 * 1024)
    
    print(f"✅ Original model: {original_size:.2f} MB")
    print(f"✅ Optimized model: {optimized_size:.2f} MB")
    print(f"🎯 Size reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")
    
    return tflite_model

# Solution 3: Model pruning (Advanced)
def create_pruned_model(keras_model_path):
    """Create a pruned model (requires tensorflow-model-optimization)"""
    try:
        import tensorflow_model_optimization as tfmot
        
        model = tf.keras.models.load_model(keras_model_path)
        
        # Define pruning parameters
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.30,
                final_sparsity=0.70,
                begin_step=0,
                end_step=1000
            )
        }
        
        # Apply pruning
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            model, **pruning_params
        )
        
        # Compile the pruned model
        model_for_pruning.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Pruned model created (requires fine-tuning)")
        return model_for_pruning
        
    except ImportError:
        print("❌ tensorflow-model-optimization not installed")
        print("💡 Install with: pip install tensorflow-model-optimization")
        return None

# Solution 4: Lazy loading approach for Flask
class LazyModelLoader:
    """Load model only when needed to reduce startup memory"""
    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None
        self._interpreter = None
        self.is_tflite = model_path.endswith('.tflite')
    
    @property
    def model(self):
        if self._model is None and not self.is_tflite:
            print("🔄 Loading Keras model...")
            self._model = tf.keras.models.load_model(self.model_path)
            print("✅ Keras model loaded")
        return self._model
    
    @property
    def interpreter(self):
        if self._interpreter is None and self.is_tflite:
            print("🔄 Loading TFLite interpreter...")
            self._interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self._interpreter.allocate_tensors()
            print("✅ TFLite interpreter loaded")
        return self._interpreter
    
    def predict(self, input_data):
        if self.is_tflite:
            return self._predict_tflite(input_data)
        else:
            return self.model.predict(input_data)
    
    def _predict_tflite(self, input_data):
        interpreter = self.interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data

# Solution 5: Memory-efficient Flask app
def create_memory_efficient_app():
    """Modified Flask app for better memory management"""
    from flask import Flask, request, jsonify
    import numpy as np
    from PIL import Image
    import io
    import gc  # Garbage collection
    
    app = Flask(__name__)
    
    # Use lazy loading
    model_loader = LazyModelLoader('model_optimized.tflite')  # or your model
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Process image
            file = request.files['file']
            image = Image.open(io.BytesIO(file.read()))
            
            # Preprocess (memory efficient)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image.resize((224, 224))
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model_loader.predict(img_array)
            
            # Get results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Clean up memory
            del img_array, image
            gc.collect()
            
            return jsonify({
                'success': True,
                'predicted_class': f'Class_{predicted_class_idx}',
                'confidence': confidence
            })
            
        except Exception as e:
            gc.collect()  # Clean up on error too
            return jsonify({'error': str(e)}), 500
    
    return app

# Run the optimization
if __name__ == "__main__":
    # Check if you have the original model
    model_paths = ['model.keras', 'model.h5']
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"📁 Found model: {path}")
            size = check_model_size(path)
            
            if size > 100:  # If larger than 100MB
                print("⚠️  Model is large, creating optimized version...")
                create_optimized_model(path)
            else:
                print("✅ Model size is acceptable")
            break
    else:
        print("❌ No model file found! Please ensure model.keras or model.h5 exists.")
