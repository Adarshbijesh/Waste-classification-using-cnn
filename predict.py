"""
Waste Classification Prediction Script
Use the trained model to classify waste images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import json
import os
import argparse
from PIL import Image

# Configuration
MODEL_PATH = 'waste_classification_model.h5'
CLASS_INDICES_PATH = 'class_indices.json'

def load_model_and_classes():
    """Load the trained model and class indices"""
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    
    # Get model input shape
    if hasattr(model, 'input_shape') and model.input_shape:
        input_shape = model.input_shape[1:3]  # Get (height, width)
    else:
        # Try to get from first layer
        try:
            input_shape = model.layers[0].input_shape[0][1:3]
        except:
            input_shape = (224, 224)  # Default fallback
    
    print(f"Model input size: {input_shape[0]}x{input_shape[1]}")
    
    print(f"Loading class indices from {CLASS_INDICES_PATH}...")
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # Reverse the dictionary to get index -> class name mapping
    index_to_class = {v: k for k, v in class_indices.items()}
    
    return model, index_to_class, input_shape

def preprocess_image(img_path, img_size):
    """Preprocess image for prediction using EfficientNet preprocessing"""
    img = Image.open(img_path)
    img = img.resize(img_size)
    img_array = np.array(img, dtype=np.float32)
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Ensure RGB format
    if img_array.shape[2] != 3:
        img_array = img_array[:, :, :3]
    
    img_array = np.expand_dims(img_array, axis=0)
    # Use EfficientNet preprocessing (normalizes to [-1, 1] range)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, img_path, index_to_class, img_size, top_k=3):
    """Predict the class of an image"""
    # Preprocess image with correct size
    img_array = preprocess_image(img_path, img_size)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top-k predictions
    top_k_indices = np.argsort(predictions)[-top_k:][::-1]
    top_k_probs = predictions[top_k_indices]
    
    results = []
    for idx, prob in zip(top_k_indices, top_k_probs):
        class_name = index_to_class[idx]
        results.append({
            'class': class_name,
            'probability': float(prob)
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Waste Classification Predictor')
    parser.add_argument('image_path', type=str, help='Path to the image file to classify')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions to show (default: 3)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    # Load model and classes
    model, index_to_class, img_size = load_model_and_classes()
    
    # Make prediction
    print(f"\nClassifying image: {args.image_path}")
    print("-" * 50)
    results = predict_image(model, args.image_path, index_to_class, img_size, args.top_k)
    
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

