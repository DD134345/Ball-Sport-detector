import tensorflow as tf
import keras
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tf.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tf.keras.optimizers import Adam
from tf.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tf.keras.applications import MobileNetV2
from datetime import datetime

# ============ CONFIGURATION ============
TRAIN_IMAGE = 'C:/Users/huyph/Downloads/Dataset/train'
TEST_IMAGE = 'C:/Users/huyph/Downloads/Dataset/test'
IMAGE_SIZE = (224, 224)  # Increased for better feature extraction
BATCH_SIZE = 16  # Reduced batch size for better generalization
NUM_CLASSES = 6

# ============ DATA AUGMENTATION ============
print("🔄 Setting up data augmentation...")

# Enhanced data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,           
    width_shift_range=0.15,      
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    brightness_range=[0.7, 1.3], 
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    # Additional augmentation
    channel_shift_range=20
)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_IMAGE,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Validation data (minimal augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)
test_generator = val_datagen.flow_from_directory(
    directory=TEST_IMAGE,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ============ MODEL ARCHITECTURE - TRANSFER LEARNING ============
print("🏗️  Building optimized model with transfer learning...")

# Use pre-trained MobileNetV2 as base
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Freeze initial layers for faster training
base_model.trainable = False

# Build custom top layers
model = Sequential([
    Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

# ============ COMPILE MODEL ============
print("⚙️  Compiling model...")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

model.summary()

# ============ CALLBACKS ============
print("📋 Setting up training callbacks...")

# Create timestamped log directory for TensorBoard
log_dir = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=1,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'Ball_sport_classifier.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# ============ TRAINING ============
print("\n" + "="*60)
print("🚀 BEGIN TRAINING")
print("="*60)

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=test_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    validation_steps=math.ceil(test_generator.samples / BATCH_SIZE),
    callbacks=callbacks,
    verbose=1
)

# ============ FINE-TUNING ============
print("\n" + "="*60)
print("🔧 FINE-TUNING - Unfreezing base model layers...")
print("="*60)

# Unfreeze the last layers of base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Keep early layers frozen
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

# Continue training
history_finetune = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    validation_steps=math.ceil(test_generator.samples / BATCH_SIZE),
    callbacks=callbacks,
    verbose=1
)

# ============ EVALUATION ============
print("\n" + "="*60)
print("📊 EVALUATION ON TEST SET")
print("="*60)

test_loss, test_acc, test_top2 = model.evaluate(
    test_generator,
    steps=math.ceil(test_generator.samples / BATCH_SIZE)
)
print(f'\n✓ Test Accuracy: {test_acc*100:.2f}%')
print(f'✓ Top-2 Accuracy: {test_top2*100:.2f}%')
print(f'✓ Test Loss: {test_loss:.4f}')

class_names = list(train_generator.class_indices.keys())

# ============ VISUALIZATION ============
print("\n📈 Generating visualization plots...")

# Plot 1: Accuracy and Loss
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Combine history from both training phases
all_accuracy = history.history['accuracy'] + history_finetune.history['accuracy']
all_val_accuracy = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
all_loss = history.history['loss'] + history_finetune.history['loss']
all_val_loss = history.history['val_loss'] + history_finetune.history['val_loss']

axes[0].plot(all_accuracy, label='Train Accuracy', linewidth=2)
axes[0].plot(all_val_accuracy, label='Validation Accuracy', linewidth=2)
axes[0].axvline(x=len(history.history['accuracy']), color='red', linestyle='--', label='Fine-tuning Start')
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(all_loss, label='Train Loss', linewidth=2)
axes[1].plot(all_val_loss, label='Validation Loss', linewidth=2)
axes[1].axvline(x=len(history.history['loss']), color='red', linestyle='--', label='Fine-tuning Start')
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Confusion Matrix
print("Generating confusion matrix...")
test_generator.reset()
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE), verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

confusionMatrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============ CLASSIFICATION REPORT ============
print("\n" + "="*60)
print("📋 CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Save classification report
with open('classification_report.txt', 'w') as f:
    f.write("BALL SPORT DETECTOR - CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Top-2 Accuracy: {test_top2*100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write("="*60 + "\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=False) + "\n")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"✓ Best Model saved as: 'Ball_sport_classifier.h5'")
print(f"✓ Training history saved as: 'training_history.png'")
print(f"✓ Confusion matrix saved as: 'confusion_matrix.png'")
print(f"✓ Classification report saved as: 'classification_report.txt'")
print("="*60)
