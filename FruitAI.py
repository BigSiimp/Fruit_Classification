import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import matplotlib.pyplot as plt

# Library os und cv2 sind für bounding boxes und live image recognition, wofür man z.b. yolo erweitert
#import os
#import cv2

# Define paths
train_dir = r'Dataset\Fruits Classification\train'
validation_dir = r'Dataset\Fruits Classification\valid'
test_dir = r'Dataset\Fruits Classification\test'

# Data augmentation for training data and normalization for validation and test data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc:.2f}')

# Get class labels
class_labels = list(validation_generator.class_indices.keys())

# Select a random image from validation set
random_idx = random.randint(0, validation_generator.samples - 1)
validation_generator.reset()
for i in range(random_idx + 1):
    img, label = next(validation_generator)

# Predict the selected image
img_for_prediction = np.expand_dims(img[0], axis=0)
predictions = model.predict(img_for_prediction)
predicted_class = np.argmax(predictions, axis=1)[0]
confidence = predictions[0]

# Display the image and prediction
plt.imshow(img[0])
plt.title(f'Predicted: {class_labels[predicted_class]} ({confidence[predicted_class]:.2f})\n'
          f'True: {class_labels[np.argmax(label[0])]}\n'
          f'All Predictions: {confidence}')
plt.show()

