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

model = tf.keras.models.load_model(model_dir)

# run the model on every photo and save it in the corresponding folder

for filename in os.listdir(data_dir):

    photo = Image.open(data_dir/filename)
    img = tf.keras.utils.load_img(
        data_dir/filename, target_size=(img_height, img_width)
    )
    
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    photo.save(script_dir/class_names[np.argmax(score)]/filename)
