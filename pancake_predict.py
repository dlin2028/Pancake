import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the best model
best_model = tf.keras.models.load_model('best_model.h5')

# Predictions functionality (on new unseen images)
def predict_image(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    pred = predictions[0]
    if pred >= 0.5:
        print(f"Prediction: Cooked with confidence {pred}")
    else:
        print(f"Prediction: Uncooked with confidence {1 - pred}")

#predict_image(best_model, 'path/to/your/image.jpg')