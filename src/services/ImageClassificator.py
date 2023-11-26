import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from src.libs.utils import decode_img
import logging

STEPS_PER_EPOCH = 20
EPOCHS = 10
VALIDATION_STEPS = 50

KEYBOARD_MAPPING = [
    'down',
    'top',
    'left',
    'right',
]

class ImageClassificator:
    __slots__ = ("model")

    def __init__(self) -> None:
        self.model =  tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (720,1280,3)) ,
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32,(3,3),activation = "relu"), 
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64,(3,3),activation = "relu") ,  
            tf.keras.layers.MaxPooling2D(2,2),
            # tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),  
            # tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(), 
            # tf.keras.layers.Flatten(), 
            # tf.keras.layers.Dense(550,activation="relu"),
            # tf.keras.layers.Dropout(0.1,seed = 2019),
            # tf.keras.layers.Dense(400,activation ="relu"),
            # tf.keras.layers.Dropout(0.3,seed = 2019),
            # tf.keras.layers.Dense(300,activation="relu"),
            # tf.keras.layers.Dropout(0.4,seed = 2019),
            # tf.keras.layers.Dense(200,activation ="relu"),
            # tf.keras.layers.Dropout(0.2,seed = 2019),
            tf.keras.layers.Dense(4,activation = "softmax")
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])

    
    
    def train(self, X, classes, output):        
        # train = tf.keras.utils.split_dataset(
        #     dataset, left_size=None, right_size=None, shuffle=True, seed=None
        # )
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'{output}/cp.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)
        
        self.model.fit(
                    x=X,
                    y=classes,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    # validation_steps=VALIDATION_STEPS
                    callbacks=[cp_callback]
                    )
    
    def predict(self, fileName, output):
        self.model.load_weights(f'{output}/cp.ckpt').expect_partial()
        X = []
        image = decode_img(tf.io.read_file(fileName))
        X.append(image)
        predict = self.model.predict(x=np.array(X))
        logging.warning(predict)
        return KEYBOARD_MAPPING[int(np.argmax(predict, axis=1))]

            