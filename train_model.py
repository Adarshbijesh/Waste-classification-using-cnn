"""
Waste Classification Model Training Script
Image Classification Model for Biodegradable and Non-Biodegradable Waste
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import glob
from pathlib import Path
import shutil
from PIL import ImageFile

# Allow loading of truncated/corrupted images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration - Optimized for high accuracy
IMG_SIZE = (300, 300)  # Increased back to 300 for B4
BATCH_SIZE = 16  
EPOCHS = 50  
LEARNING_RATE = 0.0003  # Lower initial LR for B4
NUM_CLASSES = 8

# Paths
TRAIN_DIR = 'train'
# We don't have a separate validation directory, so we will use validation_split
MODEL_SAVE_PATH = 'waste_classification_model.h5'

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Get class names from nested directory structure
def get_class_names_from_nested(train_dir):
    """Extract class names from nested directory structure"""
    class_names = []
    class_map = {}
    idx = 0
    
    for category in sorted(os.listdir(train_dir)):
        category_path = os.path.join(train_dir, category)
        if os.path.isdir(category_path):
            for class_name in sorted(os.listdir(category_path)):
                class_path = os.path.join(category_path, class_name)
                if os.path.isdir(class_path):
                    # Use just the class name (e.g., 'food_waste') as the class identifier
                    class_names.append(class_name)
                    class_map[class_name] = idx
                    idx += 1
    
    return class_names, class_map

def create_temp_flat_structure(source_dir, temp_dir):
    """Create temporary flat directory structure for ImageDataGenerator"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    class_names, class_map = get_class_names_from_nested(source_dir)
    
    # Create class directories in temp folder
    for class_name in class_names:
        os.makedirs(os.path.join(temp_dir, class_name), exist_ok=True)
    
    # Copy or symlink files
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if os.path.isdir(category_path):
            for class_name in os.listdir(category_path):
                class_path = os.path.join(category_path, class_name)
                if os.path.isdir(class_path):
                    dest_class_dir = os.path.join(temp_dir, class_name)
                    # Copy files
                    for file in os.listdir(class_path):
                        src_file = os.path.join(class_path, file)
                        if os.path.isfile(src_file) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                            dest_file = os.path.join(dest_class_dir, f"{category}_{file}")
                            try:
                                shutil.copy2(src_file, dest_file)
                            except Exception as e:
                                print(f"Warning: Could not copy {src_file}: {e}")
    
    return class_names, class_map

# Create temporary flat structures
print("\nCreating temporary flat directory structure for data loading...")
TEMP_TRAIN_DIR = 'temp_train'

class_names, class_map = create_temp_flat_structure(TRAIN_DIR, TEMP_TRAIN_DIR)

print(f"\nFound {len(class_names)} classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Enhanced Data Augmentation for Training - Tuned down for clarity
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # EfficientNet preprocessing
    rotation_range=20,  # Reduced from 40
    width_shift_range=0.2, # Reduced from 0.3
    height_shift_range=0.2, # Reduced from 0.3
    shear_range=0.2, # Reduced from 0.3
    zoom_range=0.2, # Reduced from 0.3
    horizontal_flip=True,
    vertical_flip=False,  # Waste is rarely upside down outside context
    fill_mode='nearest',
    brightness_range=[0.8, 1.2], # Tuned from 0.7-1.3
    channel_shift_range=20.0,  # Add back slight color augmentation
    samplewise_center=False,
    samplewise_std_normalization=False,
    validation_split=0.2 # Use 20% of data for validation
)

# Create data generators using flat structure
print("\nLoading training data...")
train_generator = train_datagen.flow_from_directory(
    TEMP_TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    classes=class_names,
    subset='training'
)

print("\nLoading validation data...")
val_generator = train_datagen.flow_from_directory(
    TEMP_TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    classes=class_names,
    subset='validation'
)

NUM_CLASSES = train_generator.num_classes
print(f"\nNumber of classes: {NUM_CLASSES}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Calculate class weights for imbalanced data
class_counts = {}
for class_name in class_names:
    class_path = os.path.join(TEMP_TRAIN_DIR, class_name)
    if os.path.exists(class_path):
        class_counts[class_name] = len([f for f in os.listdir(class_path) 
                                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])

total_samples = sum(class_counts.values())
class_weights = {}
# Use generator's class_indices mapping to ensure correct index mapping
for class_name, idx in train_generator.class_indices.items():
    if class_name in class_counts and class_counts[class_name] > 0:
        class_weights[idx] = total_samples / (NUM_CLASSES * class_counts[class_name])
    else:
        class_weights[idx] = 1.0

print(f"\nClass distribution:")
for class_name, idx in train_generator.class_indices.items():
    count = class_counts.get(class_name, 0)
    weight = class_weights.get(idx, 1.0)
    print(f"  {class_name}: {count} samples, weight: {weight:.3f}")

# Build Model using Transfer Learning (EfficientNetB4)
print("\nBuilding optimized model...")
base_model = EfficientNetB4(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze base model initially
base_model.trainable = False

# Build optimized model with simplified head architecture
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)

# Use GlobalAveragePooling2D
x = layers.GlobalAveragePooling2D()(x)

# Simplified dense layer for less overfitting
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compile the model with optimized settings and Focal Loss
loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    loss=loss_fn,
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
    ],
)

print("\nModel Summary:")
model.summary()

# Enhanced Callbacks for better training
early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',  # Monitor accuracy instead of loss
    patience=10,  # More patience
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,  # More aggressive reduction
    patience=5,  # Increased patience
    min_lr=1e-8,
    verbose=1,
    mode='min'
)

# Learning rate warmup with cosine decay
def warmup_cosine_schedule(epoch, lr):
    """Warmup for first 5 epochs, then cosine decay"""
    if epoch < 5:
        # Warmup phase
        return LEARNING_RATE * (epoch + 1) / 5
    else:
        # Cosine decay
        import math
        total_epochs = EPOCHS
        decay_epochs = total_epochs - 5
        current_decay_epoch = epoch - 5
        cosine_decay = 0.5 * (1 + math.cos(math.pi * current_decay_epoch / decay_epochs))
        return LEARNING_RATE * cosine_decay

lr_scheduler = callbacks.LearningRateScheduler(warmup_cosine_schedule, verbose=1)

model_checkpoint = callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    mode='max'
)

# Training with class weights
print("\nStarting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr, lr_scheduler, model_checkpoint],
    class_weight=class_weights,  # Handle class imbalance
    verbose=1
)

print(f"\nTraining completed! Best model saved to {MODEL_SAVE_PATH}")

# Unfreeze some layers for fine-tuning
print("\nFine-tuning with unfrozen layers...")
base_model.trainable = True

# Freeze early layers, fine-tune later ones
for layer in base_model.layers[:-60]:  # Unfreeze the last 60 layers for deeper fine-tuning
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
fine_tune_lr = LEARNING_RATE / 10  # Standard conservative decay
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    loss=loss_fn,
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
    ],
)

# Continue training with fine-tuning
fine_tune_epochs = 20  # More fine-tuning epochs
history_fine = model.fit(
    train_generator,
    initial_epoch=len(history.epoch),
    epochs=len(history.epoch) + fine_tune_epochs,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    class_weight=class_weights,
    verbose=1
)

# Evaluate the model
print("\nEvaluating model on validation set...")
val_loss, val_accuracy, val_top3 = model.evaluate(val_generator, verbose=1)
print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
print(f"Validation Top-3 Accuracy: {val_top3*100:.2f}%")

# Generate predictions for classification report
print("\nGenerating classification report...")
val_generator.reset()
predictions = model.predict(val_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Classification Report
print("\nClassification Report:")
print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
))

# Plot training history
def plot_history(history, history_fine=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    if history_fine:
        axes[0, 0].plot(history_fine.history['accuracy'], label='Fine-tune Train')
        axes[0, 0].plot(history_fine.history['val_accuracy'], label='Fine-tune Val')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    if history_fine:
        axes[0, 1].plot(history_fine.history['loss'], label='Fine-tune Train')
        axes[0, 1].plot(history_fine.history['val_loss'], label='Fine-tune Val')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-3 Accuracy
    axes[1, 0].plot(history.history['top_3_accuracy'], label='Train Top-3')
    axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Val Top-3')
    if history_fine:
        axes[1, 0].plot(history_fine.history['top_3_accuracy'], label='Fine-tune Train')
        axes[1, 0].plot(history_fine.history['val_top_3_accuracy'], label='Fine-tune Val')
    axes[1, 0].set_title('Top-3 Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-3 Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate')
        if history_fine and 'lr' in history_fine.history:
            axes[1, 1].plot(history_fine.history['lr'], label='Fine-tune LR')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history saved to training_history.png")
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(true_classes, predicted_classes, class_labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to confusion_matrix.png")
    plt.close()

# Generate plots
plot_history(history, history_fine)
plot_confusion_matrix(true_classes, predicted_classes, class_labels)

# Save class indices mapping
import json
class_indices = val_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f, indent=2)
print("\nClass indices saved to class_indices.json")

# Clean up temporary directories
print("\nCleaning up temporary directories...")
if os.path.exists(TEMP_TRAIN_DIR):
    shutil.rmtree(TEMP_TRAIN_DIR)
print("Temporary directories removed.")

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
print(f"Best model saved: {MODEL_SAVE_PATH}")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Validation Top-3 Accuracy: {val_top3*100:.2f}%")

