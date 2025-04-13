import os
import random
import textwrap
import time
import GPUtil

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


class classifierModel():
    def __init__(self, loadFileDir: str):
        """
        creates a new LSTM trading model class
        :param loadFileDir: directory to load from, if not found it will create a new one with this directory in mind
        """
        tf.random.set_seed(random.randint(0, 6000))
        self.filePath = loadFileDir
        self.path = loadFileDir
        self.opt = keras.optimizers.Adam(learning_rate=0.00025)
        if os.path.exists(loadFileDir):
            self.model = load_model(loadFileDir)
            self.model.compile(optimizer=self.opt, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            self.model = self._createNewModel()
            self.model.compile(optimizer=self.opt, loss='binary_crossentropy', metrics=['accuracy'])

    def _createNewModel(self):
        ###
        # After I wrote this around 6 months ago, there were some things in terms of architecture that I would do
        # differently as this model architecture is kind of bad for the job... However, it got the job done.
        #
        # also I didn't write this in a very good way, but it's fine for a sequential model. However, if I wanted
        # to use any other type of model like a MOE or anything with multiple inputs and paths and what not I would
        # write it with the x = layer(x) format because it is much neater and allows for more complex operations
        ###
        newModel = keras.models.Sequential()
        newModel.add(layers.Input(shape=(6,6)))
        newModel.add(layers.Flatten())
        newModel.add(layers.Normalization())
        newModel.add(layers.Dense(100))
        newModel.add(layers.Dense(120))
        newModel.add(layers.Dense(800))
        newModel.add(layers.Dense(800))
        newModel.add(layers.Dense(800))
        newModel.add(layers.Dropout(0.3))
        newModel.add(layers.Dense(1, activation='sigmoid'))
        return newModel

    def fitModelLoop(self, x, y, epochs, batch_size=32):
        """fit the model to the dataset"""
        self.model.fit(x, y, epochs=epochs, shuffle=True, batch_size=batch_size)
        self.model.save(self.path)

    def saveModel(self):
        """saves the model to the path"""
        self.model.save(self.path)

    def __call__(self, values):
        """input a sequence to get a prediction"""
        return self.model.predict(values)