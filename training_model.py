import tensorflow as tf
import keras
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout,BatchNormalization,Input,SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
#Preprocessing and Data Loading
TRAIN_IMAGE = 'C:/Users/huyph/Downloads/Dataset/train' #Train image directory
TEST_IMAGE = 'C:/Users/huyph/Downloads/Dataset/test' #Test image directory
IMAGE_SIZE = (192,192) #image size 224x224
BATCH_SIZE = 32#number of image for each literation
EPOCHS = 100
train_val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True
)

train_generator = train_val_datagen.flow_from_directory(
    TRAIN_IMAGE,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
class_names = list(train_generator.class_indices.keys())
print("Classes:", class_names)
val_generator = train_val_datagen.flow_from_directory(
    TRAIN_IMAGE,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_IMAGE,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
#Model Architecture(Convolutional Neural Network)

# Input & Feature Learning
model = Sequential()
model.add(Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))) 
model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    padding='same',
))
model.add(BatchNormalization())# Stabilizes learning process
model.add(MaxPooling2D((2, 2)))# Reduces spatial dimensions
model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu',
    padding='same',
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(
    filters=128,
    kernel_size=(3, 3),
    activation='relu',
    padding='same',
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    activation='relu',
    padding='same',
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(SpatialDropout2D(0.3))
#Classification Head
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(
    units=6,
    activation='softmax'
))
#Compile the Model
model.compile(optimizer=Adam(learning_rate=1e-4),# Set a small learning rate for stable training
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),# Standard loss function for multi-class classification with one-hot labels
              metrics=['accuracy'])
model.summary()
print("\nBegin training")
# Define Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',# Stop training when validation loss stops improving        
    patience=10,# Number of epochs wait when validation loss stops improving           
    mode='min',#Sort the lowest validation loss                
    verbose=1
)
checkpoint = ModelCheckpoint(
    'Ball_sport_classifier.h5',#Path to save model
    monitor='val_loss',# Save the model based on the best validation loss
    save_best_only=True,# Only the best version of the model is saved          
    mode='min',
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,# Reduce learning rate by half        
    patience=5,        
    min_lr=1e-6,#lowest learning rate      
    verbose=1
)
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    validation_steps=math.ceil(val_generator.samples / BATCH_SIZE),
    callbacks=[early_stopping, checkpoint, reduce_lr]
)
test_loss, test_acc = model.evaluate(
    test_generator,
    steps=math.ceil(test_generator.samples / BATCH_SIZE)
)
print(f"\nTest Accuracy: {test_acc:.4f}")
#Evaluation and Visualization
print("\nEvaluation on the Test Set")
# Plot Accuracy and Loss over epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
#Confusion Matrix and Classification Report
test_generator.reset() 
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE))
y_pred = np.argmax(y_pred_probs, axis=1)
#Confusion Matrix
confusionMatrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues',
xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show() 
#Classification Report
print("\nClassification report(Precision, Recall, F1-Score)")
print(classification_report(y_true, y_pred, target_names=class_names))
plt.figure(figsize=(6, 4))
plt.plot(history.history['learning_rate'])
plt.title('Learning Rate by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
print(f"\nTraining Complete. Best Model is saved in 'Ball_sport_classifier.h5'")
