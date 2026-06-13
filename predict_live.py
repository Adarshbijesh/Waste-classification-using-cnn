"""
Live Waste Classification from Webcam
Uses the trained model to classify waste in real time from your laptop camera.
"""

import argparse
import json
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = "waste_classification_model.h5"
CLASS_INDICES_PATH = "class_indices.json"


def load_model_and_classes():
    """Load the trained model and class indices."""
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    if hasattr(model, "input_shape") and model.input_shape:
        input_shape = model.input_shape[1:3]
    else:
        try:
            input_shape = model.layers[0].input_shape[0][1:3]
        except Exception:
            input_shape = (224, 224)

    print(f"Model input size: {input_shape[0]}x{input_shape[1]}")

    print(f"Loading class indices from {CLASS_INDICES_PATH}...")
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)

    index_to_class = {v: k for k, v in class_indices.items()}
    return model, index_to_class, input_shape


def preprocess_frame(frame_bgr, img_size):
    """Convert a webcam frame (BGR) into model input."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, img_size)
    img_array = np.array(resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def predict_frame(model, frame_bgr, index_to_class, img_size, top_k=3):
    """Run prediction on a single webcam frame."""
    img_array = preprocess_frame(frame_bgr, img_size)
    predictions = model.predict(img_array, verbose=0)[0]

    top_k_indices = np.argsort(predictions)[-top_k:][::-1]
    top_k_probs = predictions[top_k_indices]

    results = []
    for idx, prob in zip(top_k_indices, top_k_probs):
        results.append(
            {
                "class": index_to_class[idx],
                "probability": float(prob),
            }
        )
    return results


def draw_predictions(frame, results):
    """Overlay prediction text on the webcam frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, 30 + 28 * len(results)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    for i, result in enumerate(results):
        text = f"{result['class']}: {result['probability'] * 100:.1f}%"
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        cv2.putText(
            frame,
            text,
            (20, 40 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "Press Q to quit",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def main():
    parser = argparse.ArgumentParser(description="Live waste classification from webcam")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show (default: 3)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between predictions (default: 0.5)",
    )
    args = parser.parse_args()

    model, index_to_class, img_size = load_model_and_classes()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    print("\nWebcam started. Point the camera at waste items.")
    print("Press Q in the video window to quit.\n")

    last_prediction_time = 0.0
    latest_results = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            now = time.time()
            if now - last_prediction_time >= args.interval:
                latest_results = predict_frame(
                    model, frame, index_to_class, img_size, args.top_k
                )
                last_prediction_time = now

                top = latest_results[0]
                print(
                    f"Predicted: {top['class']} ({top['probability'] * 100:.1f}%)",
                    end="\r",
                )

            display_frame = draw_predictions(frame, latest_results)
            cv2.imshow("Waste Classification - Live", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera closed.")


if __name__ == "__main__":
    main()
