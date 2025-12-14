import tensorflow as tf
import keras
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization,Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#Preprocessing and Data Loading
TRAIN_IMAGE = 'C:/Users/huyph/Downloads/Archi/train'
TEST_IMAGE = 'C:/Users/huyph/Downloads/Archi/test'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32 #number of image for each literation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,          
    width_shift_range=0.1,       
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.8,1.2],
    channel_shift_range=10.0,    
    horizontal_flip=True,        
    fill_mode='nearest'
)
train_generator = datagen.flow_from_directory(
    directory=TRAIN_IMAGE,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=TEST_IMAGE,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
# Input & Feature Learning
model = Sequential()
model.add(Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    padding='same',
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu',
    padding='same'
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(
    filters=128,
    kernel_size=(3, 3),
    activation='relu',
    padding='same'
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    activation='relu',
    padding='same'
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(
    filters=512,
    kernel_size=(3, 3),
    activation='relu',
    padding='same'
))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
#Classification Head
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(
    units=6,
    activation='softmax'
))
#Compile the Model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

print("\nBegin training")
early_stopping = EarlyStopping( 
    monitor='val_loss',         
    patience=10,               
    mode='min',                 
    verbose=1
)
checkpoint = ModelCheckpoint(
    'best_ball_classifier.h5', 
    monitor='val_loss',
    save_best_only=True,          
    mode='min',
    verbose=1
)
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=test_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    validation_steps=math.ceil(test_generator.samples / BATCH_SIZE),
    callbacks=[early_stopping, checkpoint]
)

print("\n--- ĐÁNH GIÁ TRÊN TẬP KIỂM TRA ---")
test_loss, test_acc = model.evaluate(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE))
print(f'\nĐộ chính xác trên tập kiểm tra: {test_acc:.4f}')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Độ chính xác Mô hình (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Hàm Mất mát (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
print(f"\nTraining Complete.")
