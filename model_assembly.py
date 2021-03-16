from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

def get_base_model():
    input_img = layers.Input(shape=(128, 128, 3))
    x = layers.Conv2D(64,(3, 3), activation='relu')(input_img)
    return input_img,x

def get_additional_layer(filters,x):
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu')(x)
    return x

def get_final_layers(neurons,x):
    x = layers.Flatten()(x)
    x = layers.Dense(neurons)(x)
    x = layers.Dense(3)(x)
    return x

    





