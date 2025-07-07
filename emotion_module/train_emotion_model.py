import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# STEP 1: Dataset Path (update this if needed)
dataset_path = "C:/Users/hans/source/repos/RealTimeBehaviorSystem/RealTimeBehaviorSystem/face_module/facial_expressions-master"  # Make sure this folder exists and has 7 emotion subfolders

# STEP 2: Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    subset="training"
)

valid_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    subset="validation"
)

# STEP 3: Build the Model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotions
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# STEP 4: Train the Model
model = build_model()
model.fit(train_gen, validation_data=valid_gen, epochs=30)

# STEP 5: Save the Model
output_path = os.path.join("emotion_module", "custom_emotion_model.h5")
model.save(output_path)
print(f"✅ Model saved to: {output_path}")
