import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
from PIL import Image
from skimage.transform import rescale
import tensorflow as tf
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

img_width = 1296
img_height = 972

class_names = ['cirrus', 'not cirrus']

script_dir = Path(__file__).parent.resolve()
data_dir = script_dir/'data'
model_dir = script_dir/'model'


def intensity(r, g, b):

    return np.float32((r + b + g) / 3)


def rbd(r, b):

    return np.float32(r - b)


def egd(r, g, b):

    ans = np.sqrt(r**2 + g**2 + b**2 - (r + g + b)**2 / 3)

    # linearly rescale from [0, 208] to [0, 255]
    return np.float32(ans / 208 * 255)


model = tf.keras.models.load_model(model_dir)

for filename in os.listdir(data_dir):

    photo = Image.open(data_dir/filename)
    img = tf.keras.utils.load_img(
        data_dir/filename, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)

    for row in img_array:
        for pixel in row:

            r = float(pixel[0])
            g = float(pixel[1])
            b = float(pixel[2])

            pixel[0] = rbd(r, b)
            pixel[1] = intensity(r, g, b)
            pixel[2] = egd(r, g, b)

    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    photo.save(script_dir/class_names[np.argmax(score)]/filename)
