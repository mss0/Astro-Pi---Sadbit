import os
from skimage.transform import rescale
import matplotlib.image as img
from pathlib import Path
import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

script_dir = Path(__file__).parent.resolve()
data_dir = script_dir/'data'
cirrus_dir = data_dir/'cirrus'
not_cirrus_dir = data_dir/'not cirrus'
img_dir = script_dir/'images'

combinations = [img_dir/'Combination 1 -- rbd, intensity, egd',
                img_dir/'Combination 2 -- nrbr, saturation, egd',
                img_dir/'Combination 3 -- hue, satruation, egd',
                img_dir/'Combination 4 -- hue, intensity, egd']

batch_size = 32
img_width = 1296
img_height = 972

# for every image, turn it's pixels to arrays containing more useful data

# different metrics one can use to better identify cirrus clouds


def hue(r, g, b):

    d = np.sqrt((r - g)**2 + (r - b) * (g - b))
    if d == 0:
        d = 0.0001

    ans = np.degrees(np.arccos(1/2 * ((r - g) + (r - b)) / d))

    if b > g:
        ans = 360 - ans

    # linearly rescale from [0, 360] to [0, 255]
    return np.uint8(ans / 360 * 255)


def intensity(r, g, b):

    return np.uint8((r + b + g) / 3)


def brr(r, b):

    if r == 0:
        r = 0.0001

    return np.uint8(b / r)


def rbd(r, b):

    return np.uint8(r - b)


def nrbr(r, b):

    d = b + r
    if d == 0:
        d = 0.0001

    ans = (b - r) / d

    # linearly rescale from [-1, 1] to [0, 255]
    return np.uint8((ans + 1) / 2 * 255)


def saturation(r, g, b):

    d = r + g + b
    if d == 0:
        d = 0.0001

    ans = min(r, g, b) / d

    # linearly rescale from [0, 1] to [0, 255]
    return np.uint8(ans * 255)


def egd(r, g, b):

    ans = np.sqrt(r**2 + g**2 + b**2 - (r + g + b)**2 / 3)

    # linearly rescale from [0, 208] to [0, 255]
    return np.uint8(ans / 208 * 255)


def preprocess(img_path, c, name):

    for index in range(0, len(combinations)):

        # get the image
        image = Image.open(img_path)

        # rezise
        image = image.resize((img_width, img_height))

        # transform to numpy array
        array = asarray(image)

        # iterate through every pixel
        for row in array:
            for pixel in row:

                # get rgb values
                r = float(pixel[0])
                g = float(pixel[1])
                b = float(pixel[2])

                # construct new image

                if index == 0:
                    pixel[0] = rbd(r, b)
                    pixel[1] = intensity(r, g, b)
                    pixel[2] = egd(r, g, b)
                elif index == 1:
                    pixel[0] = nrbr(r, b)
                    pixel[1] = saturation(r, g, b)
                    pixel[2] = egd(r, g, b)
                elif index == 2:
                    pixel[0] = hue(r, g, b)
                    pixel[1] = saturation(r, g, b)
                    pixel[2] = egd(r, g, b)
                elif index == 3:
                    pixel[0] = hue(r, g, b)
                    pixel[1] = intensity(r, g, b)
                    pixel[2] = egd(r, g, b)

        # save the new image
        new_img = Image.fromarray(array)

        if c == 1:
            new_img.save(combinations[index]/'cirrus'/name)
        else:
            new_img.save(combinations[index]/'not cirrus'/name)


# preprocess the images

'''
for filename in os.listdir(cirrus_dir):

    preprocess(cirrus_dir/filename, 1, filename)

for filename in os.listdir(not_cirrus_dir):

    preprocess(not_cirrus_dir/filename, 0, filename)'''
'''
what = script_dir/'dumb mf'
for filename in os.listdir(what):

    preprocess(what/filename, 1, filename)'''

training_dataset = tf.keras.utils.image_dataset_from_directory(

    combinations[0],
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(

    combinations[0],
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = training_dataset.class_names
classes_no = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                   img_width,
                                   3)),
    layers.RandomRotation(0.1),
  ]
)

model = Sequential([

    data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(classes_no)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 20

history = model.fit(

    training_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('model')
