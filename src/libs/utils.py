import os
import cv2
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

KEYBOARD_MAPPING = {
    'ArrowDown': 0,
    'ArrowUp': 1,
    'ArrowLeft': 2,
    'ArrowRight': 3
}

def getImagesForTrain(path: str):
    imagesPathnames = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for imagePath in imagesPathnames:
        imageFace = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(imagePath)[-1].split('-')[1])
        ids.append(id)
        faces.append(imageFace)

    return np.array(ids), faces

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)

    return tf.image.resize(img, [300, 300])

def get_label(file_path):
    return tf.one_hot(KEYBOARD_MAPPING)

def getKeyboardForTrain(path: str):
    keyboardPathnames = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    keyboards = []

    np.random.shuffle(keyboardPathnames)

    for keyboardPath in keyboardPathnames:
        imageKeyboard = tf.io.read_file(keyboardPath)
        keyboard = os.path.split(keyboardPath)[-1].split('-')[0]
        keyboards.append(KEYBOARD_MAPPING[keyboard])
        images.append(decode_img(imageKeyboard))

    return to_categorical(keyboards), np.array(images) 

