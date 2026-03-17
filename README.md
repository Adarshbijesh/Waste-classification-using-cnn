# ♻️ Waste Classification AI

An end-to-end deep learning pipeline for classifying waste images into **8 categories** (biodegradable vs non-biodegradable), built on **EfficientNetB4** transfer learning with support for full-size inference and TinyML deployment on edge devices.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Classes](#classes)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset Structure](#dataset-structure)
- [Training the Model](#training-the-model)
- [Running Predictions](#running-predictions)
- [TinyML / Edge Inference](#tinyml--edge-inference)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Output Files](#output-files)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project trains an image classification model to identify waste types from photos. It is designed for two deployment targets:

| Script | Model | Input Size | Use Case |
|---|---|---|---|
| `train_model.py` | EfficientNetB4 (`.h5`) | 300×300 | Server / Cloud |
| `predict.py` | EfficientNetB4 (`.h5`) | 300×300 | Server / Cloud inference |
| `predict_tinyml.py` | MobileNet-style (`.tflite`) | 96×96 | Embedded / Edge devices |

The full model targets **high accuracy**, while the TinyML variant targets **low memory / low latency** for microcontrollers and Raspberry Pi–class devices.

---

## Classes

The model is trained to classify images into **8 waste categories**, structured in a two-level nested directory (category → class):

```
Biodegradable/
    food_waste/
    garden_waste/
    paper/
    wood/

Non-Biodegradable/
    glass/
    metal/
    plastic/
    textile/
```

> The exact class names are determined automatically from the `train/` directory at runtime. After training, they are saved to `class_indices.json`.

---

## Project Structure

```
waste-classification/
│
├── train/                          # Raw dataset (nested structure)
│   ├── Biodegradable/
│   │   ├── food_waste/
│   │   └── ...
│   └── Non-Biodegradable/
│       ├── plastic/
│       └── ...
│
├── train_model.py                  # Full model training (EfficientNetB4)
├── predict.py                      # Inference script for full model
├── predict_tinyml.py               # Inference script for TinyML / TFLite models
│
├── waste_classification_model.h5   # Saved full Keras model (generated)
├── waste_tinyml_model.h5           # Saved TinyML Keras model (generated)
├── waste_tinyml_model.tflite       # TFLite model (generated)
├── waste_tinyml_quantized.tflite   # INT8 quantized TFLite model (generated)
│
├── class_indices.json              # Class name → index mapping (generated)
├── tinyml_class_indices.json       # TinyML class name → index mapping (generated)
│
├── training_history.png            # Accuracy/loss plots (generated)
└── confusion_matrix.png            # Confusion matrix heatmap (generated)
```

---

## Requirements

### Python & Core Libraries

```bash
pip install tensorflow>=2.10
pip install numpy pillow matplotlib seaborn scikit-learn
```

### Full Requirements

```
tensorflow>=2.10
numpy
Pillow
matplotlib
seaborn
scikit-learn
```

### Hardware Recommendations

| Setup | Minimum | Recommended |
|---|---|---|
| Training | 8 GB RAM, CPU | 16 GB RAM, NVIDIA GPU (CUDA) |
| Full model inference | 4 GB RAM | 8 GB RAM |
| TinyML inference | 256 MB RAM | Raspberry Pi 4 or better |

> The training script auto-detects GPU availability and will use it if present.

---

## Dataset Structure

Place your dataset in a `train/` folder with the following **nested** structure:

```
train/
├── CategoryA/
│   ├── class_1/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── class_2/
│       └── ...
└── CategoryB/
    ├── class_3/
    │   └── ...
    └── ...
```

- Supported image formats: `.jpg`, `.jpeg`, `.png` (case-insensitive)
- Images do **not** need to be pre-resized — the pipeline handles resizing automatically.
- Corrupted/truncated images are skipped gracefully (`LOAD_TRUNCATED_IMAGES = True`).
- The training script flattens this nested structure into a temporary `temp_train/` directory internally, then cleans it up after training.

**80/20 split:** 80% of images are used for training and 20% for validation, automatically via `validation_split=0.2`.

---

## Training the Model

### 1. Prepare your dataset

Ensure your `train/` directory follows the nested structure described above.

### 2. Run training

```bash
python train_model.py
```

### What happens during training

Training proceeds in **two phases**:

**Phase 1 — Feature Extraction (frozen base)**
- The EfficientNetB4 backbone is frozen.
- Only the custom classification head is trained.
- Runs for up to 50 epochs with early stopping (patience = 10).
- Uses warmup + cosine learning rate decay.

**Phase 2 — Fine-tuning (unfrozen last 60 layers)**
- The last 60 layers of EfficientNetB4 are unfrozen.
- Training continues for up to 20 more epochs.
- Learning rate is reduced 10× for careful fine-tuning.
- The best checkpoint (by `val_accuracy`) is preserved throughout.

### Training outputs

After training completes, the following files are saved:

| File | Description |
|---|---|
| `waste_classification_model.h5` | Best model checkpoint |
| `class_indices.json` | Class name ↔ index mapping |
| `training_history.png` | Accuracy, loss, top-3 accuracy, and LR plots |
| `confusion_matrix.png` | Confusion matrix on validation set |

---

## Running Predictions

### Basic usage

```bash
python predict.py path/to/image.jpg
```

### Show top-5 predictions

```bash
python predict.py path/to/image.jpg --top-k 5
```

### Example output

```
Loading model from waste_classification_model.h5...
Model input size: 300x300

Classifying image: test_bottle.jpg
--------------------------------------------------

Top 3 Predictions:
--------------------------------------------------
1. plastic: 94.31%
2. glass: 3.82%
3. metal: 1.12%

--------------------------------------------------
Predicted Class: plastic
Confidence: 94.31%
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `image_path` | positional | — | Path to the image to classify |
| `--top-k` | int | `3` | Number of top predictions to display |

> **Preprocessing note:** Images are automatically resized to the model's input size and preprocessed using EfficientNet's normalization (scales pixel values to `[-1, 1]`).

---

## TinyML / Edge Inference

The `predict_tinyml.py` script supports three model formats optimized for edge deployment:

### Model types

| Flag | File Used | Description |
|---|---|---|
| `keras` | `waste_tinyml_model.h5` | Keras model, standard float32 |
| `tflite` | `waste_tinyml_model.tflite` | TFLite converted model |
| `quantized` | `waste_tinyml_quantized.tflite` | INT8 post-training quantized model |

### Usage

```bash
# Using Keras model (default)
python predict_tinyml.py path/to/image.jpg

# Using TFLite model
python predict_tinyml.py path/to/image.jpg --model-type tflite

# Using INT8 quantized model (smallest, fastest)
python predict_tinyml.py path/to/image.jpg --model-type quantized

# Show top-5 predictions
python predict_tinyml.py path/to/image.jpg --model-type tflite --top-k 5
```

### Key differences from full model

| Property | Full Model | TinyML Model |
|---|---|---|
| Input size | 300×300 | 96×96 |
| Normalization | EfficientNet (`[-1, 1]`) | Standard (`[0, 1]`) |
| Quantization | None | Optional INT8 |
| Class index file | `class_indices.json` | `tinyml_class_indices.json` |

### Quantized model notes

The quantized TFLite model uses **INT8 weights** with automatic dequantization. The script handles both output types:
- If the model output has quantization parameters → it dequantizes to float automatically.
- If the output is already float32 → it is used directly.

---

## Model Architecture

### Full Model (EfficientNetB4)

```
Input (300, 300, 3)
    ↓
EfficientNetB4 backbone (ImageNet pretrained, frozen initially)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.5)
    ↓
Dense (512, ReLU)
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense (NUM_CLASSES, Softmax)
```

**Loss function:** `CategoricalFocalCrossentropy` (α=0.25, γ=2.0) — helps with class imbalance by down-weighting easy examples.

**Metrics:**
- Top-1 Accuracy
- Top-3 Accuracy

---

## Training Strategy

### Data Augmentation

The following augmentations are applied **only to the training set** at runtime (validation data is not augmented):

| Augmentation | Value |
|---|---|
| Rotation | ±20° |
| Width / Height shift | 20% |
| Shear | 20% |
| Zoom | 20% |
| Horizontal flip | Yes |
| Vertical flip | No |
| Brightness | [0.8, 1.2] |
| Channel shift | ±20.0 |
| Fill mode | `nearest` |

### Class Imbalance Handling

Class weights are calculated automatically based on sample counts:

```
weight[class] = total_samples / (num_classes × samples_in_class)
```

Underrepresented classes receive higher weights during loss computation.

### Learning Rate Schedule

| Phase | Schedule |
|---|---|
| Epochs 1–5 | Linear warmup: `LR × (epoch+1) / 5` |
| Epochs 6+ | Cosine decay back to near zero |
| Fine-tuning | Fixed at `LR / 10` |

Additionally, `ReduceLROnPlateau` halves the LR if validation loss does not improve for 5 epochs (min LR: `1e-8`).

### Callbacks Summary

| Callback | Monitors | Action |
|---|---|---|
| `EarlyStopping` | `val_accuracy` | Stops if no improvement for 10 epochs; restores best weights |
| `ReduceLROnPlateau` | `val_loss` | Reduces LR by 5× after 5 stalled epochs |
| `LearningRateScheduler` | — | Warmup + cosine decay |
| `ModelCheckpoint` | `val_accuracy` | Saves only when validation accuracy improves |

---

## Output Files

| File | Generated By | Description |
|---|---|---|
| `waste_classification_model.h5` | `train_model.py` | Best full model (Keras HDF5) |
| `class_indices.json` | `train_model.py` | `{"class_name": index, ...}` |
| `training_history.png` | `train_model.py` | 4-panel plot: accuracy, loss, top-3, LR |
| `confusion_matrix.png` | `train_model.py` | Heatmap of predictions vs. ground truth |
| `waste_tinyml_model.h5` | *(TinyML training script)* | TinyML Keras model |
| `waste_tinyml_model.tflite` | *(TinyML training script)* | Standard TFLite model |
| `waste_tinyml_quantized.tflite` | *(TinyML training script)* | INT8 quantized TFLite model |
| `tinyml_class_indices.json` | *(TinyML training script)* | TinyML class index mapping |

---

## Configuration Reference

Key constants at the top of `train_model.py`:

| Variable | Default | Description |
|---|---|---|
| `IMG_SIZE` | `(300, 300)` | Input image dimensions |
| `BATCH_SIZE` | `16` | Training batch size |
| `EPOCHS` | `50` | Max training epochs (Phase 1) |
| `LEARNING_RATE` | `0.0003` | Initial learning rate |
| `NUM_CLASSES` | `8` | Auto-detected from dataset |
| `TRAIN_DIR` | `'train'` | Root training data directory |
| `MODEL_SAVE_PATH` | `'waste_classification_model.h5'` | Model output path |

For TinyML inference (`predict_tinyml.py`):

| Variable | Default | Description |
|---|---|---|
| `IMG_SIZE` | `(96, 96)` | TinyML input size |
| `MODEL_PATH` | `'waste_tinyml_model.h5'` | Keras TinyML model |
| `TFLITE_MODEL_PATH` | `'waste_tinyml_model.tflite'` | Standard TFLite model |
| `QUANTIZED_TFLITE_PATH` | `'waste_tinyml_quantized.tflite'` | INT8 quantized TFLite |
| `CLASS_INDICES_PATH` | `'tinyml_class_indices.json'` | TinyML class mapping |

---

## Troubleshooting

**`Error: Image file not found`**
Make sure the path to your image is correct and the file exists.

**`OSError: Unable to open file` on model load**
Ensure `waste_classification_model.h5` (or the TinyML variants) exist in the working directory. They are created after running `train_model.py`.

**Training runs out of memory**
Reduce `BATCH_SIZE` from `16` to `8` or `4` in `train_model.py`. On CPU-only machines, also consider reducing `IMG_SIZE` to `(224, 224)` and switching to `EfficientNetB0`.

**Validation accuracy is low**
- Check that your dataset has sufficient images per class (at least 100 recommended).
- Look at `confusion_matrix.png` to identify which classes are being confused.
- Consider increasing `EPOCHS` or reducing augmentation if the model is underfitting.

**`temp_train/` directory not cleaned up**
If training is interrupted, the temporary directory may persist. Delete it manually:
```bash
rm -rf temp_train/
```

**Quantized model gives unexpected results**
Confirm the model was quantized using representative data from the same domain. The `predict_tinyml.py` script auto-handles both quantized and float outputs, but quantization accuracy may degrade if the representative dataset was too small or unrepresentative.

---

## License

This project is provided as-is for educational and research purposes.
