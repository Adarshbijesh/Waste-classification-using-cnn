"""
TinyML Model Prediction Script
Supports both Keras (.h5) and TensorFlow Lite (.tflite) models
"""

import tensorflow as tf
import numpy as np
import json
import os
import argparse
from PIL import Image

# Configuration
IMG_SIZE = (96, 96)  # Match training size
MODEL_PATH = 'waste_tinyml_model.h5'
TFLITE_MODEL_PATH = 'waste_tinyml_model.tflite'
QUANTIZED_TFLITE_PATH = 'waste_tinyml_quantized.tflite'
CLASS_INDICES_PATH = 'tinyml_class_indices.json'

def load_class_indices():
    """Load class indices mapping"""
    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"Warning: {CLASS_INDICES_PATH} not found. Using default class names.")
        return {i: f"class_{i}" for i in range(8)}
    
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # Reverse the dictionary to get index -> class name mapping
    index_to_class = {v: k for k, v in class_indices.items()}
    return index_to_class

def preprocess_image(img_path):
    """Preprocess image for prediction"""
    img = Image.open(img_path)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Ensure RGB format
    if img_array.shape[2] != 3:
        img_array = img_array[:, :, :3]
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_keras(model, img_array, index_to_class, top_k=3):
    """Predict using Keras model"""
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top-k predictions
    top_k_indices = np.argsort(predictions)[-top_k:][::-1]
    top_k_probs = predictions[top_k_indices]
    
    results = []
    for idx, prob in zip(top_k_indices, top_k_probs):
        class_name = index_to_class.get(idx, f"class_{idx}")
        results.append({
            'class': class_name,
            'probability': float(prob)
        })
    
    return results

def predict_tflite(tflite_path, img_array, index_to_class, top_k=3, quantized=False):
    """Predict using TensorFlow Lite model"""
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input
    if quantized:
        # Convert to uint8 for quantized models
        input_scale, input_zero_point = input_details[0]['quantization']
        img_array_uint8 = (img_array / input_scale + input_zero_point).astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], img_array_uint8)
    else:
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if quantized:
        # Check if output is quantized or already float32
        output_quantization = output_details[0]['quantization']
        if output_quantization[0] != 0.0:  # Has quantization parameters
            # Dequantize output
            output_scale, output_zero_point = output_quantization
            predictions = (output_data.astype(np.float32) - output_zero_point) * output_scale
        else:
            # Output is already float32 (INT8 weights + float output strategy)
            predictions = output_data[0] if len(output_data.shape) > 1 else output_data
    else:
        predictions = output_data[0] if len(output_data.shape) > 1 else output_data
    
    # Get top-k predictions
    top_k_indices = np.argsort(predictions)[-top_k:][::-1]
    top_k_probs = predictions[top_k_indices]
    
    results = []
    for idx, prob in zip(top_k_indices, top_k_probs):
        class_name = index_to_class.get(idx, f"class_{idx}")
        results.append({
            'class': class_name,
            'probability': float(prob)
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='TinyML Waste Classification Predictor')
    parser.add_argument('image_path', type=str, help='Path to the image file to classify')
    parser.add_argument('--model-type', type=str, choices=['keras', 'tflite', 'quantized'], 
                       default='keras', help='Model type to use (default: keras)')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions to show (default: 3)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    # Load class indices
    index_to_class = load_class_indices()
    
    # Preprocess image
    img_array = preprocess_image(args.image_path)
    
    # Make prediction based on model type
    print(f"\nClassifying image: {args.image_path}")
    print(f"Using model type: {args.model_type}")
    print("-" * 50)
    
    if args.model_type == 'keras':
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Keras model not found: {MODEL_PATH}")
            return
        model = tf.keras.models.load_model(MODEL_PATH)
        results = predict_keras(model, img_array, index_to_class, args.top_k)
    
    elif args.model_type == 'tflite':
        if not os.path.exists(TFLITE_MODEL_PATH):
            print(f"Error: TFLite model not found: {TFLITE_MODEL_PATH}")
            return
        results = predict_tflite(TFLITE_MODEL_PATH, img_array, index_to_class, args.top_k, quantized=False)
    
    elif args.model_type == 'quantized':
        if not os.path.exists(QUANTIZED_TFLITE_PATH):
            print(f"Error: Quantized TFLite model not found: {QUANTIZED_TFLITE_PATH}")
            return
        results = predict_tflite(QUANTIZED_TFLITE_PATH, img_array, index_to_class, args.top_k, quantized=True)
    
    # Display results
    print(f"\nTop {args.top_k} Predictions:")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['class']}: {result['probability']*100:.2f}%")
    
    print("\n" + "-" * 50)
    print(f"Predicted Class: {results[0]['class']}")
    print(f"Confidence: {results[0]['probability']*100:.2f}%")

if __name__ == '__main__':
    main()

