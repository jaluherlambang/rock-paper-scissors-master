import tensorflow as tf
import numpy as np
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.models import Sequential
import os

IMG_SAVE_PATH = 'image_data'

CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}

NUM_CLASSES = len(CLASS_MAP)

datagen = ImageDataGenerator(
    rescale=1.0/255,  # normalize pixel values to [0, 1]
    rotation_range=20,  # randomly rotate images by 20 degrees
    width_shift_range=0.1,  # randomly shift images horizontally by 10% of the width
    height_shift_range=0.1,  # randomly shift images vertically by 10% of the height
    horizontal_flip=True  # randomly flip images horizontally
)

train_generator = datagen.flow_from_directory(
    IMG_SAVE_PATH,
    target_size=(150,150),
    batch_size=50,
    class_mode='categorical' #sparse for encode an
)

model = Sequential([
    Convolution2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPool2D(2,2),
    Convolution2D(32, (3,3), activation='relu'),
    MaxPool2D(2,2),
    Convolution2D(64, (3,3), activation='relu'),
    MaxPool2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics='accuracy')
model.fit(train_generator,
          steps_per_epoch=8,
          epochs=15,
          verbose=1)
model.save("rock-paper-scissors-model-2.h5")