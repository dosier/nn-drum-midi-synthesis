# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:45:44 2021

@author: Sietse
"""
import random
import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import glob
import math
from typing import List
import natsort
import numpy
from numpy import ndarray

directory = 'C:/Users/Sietse/Documents/Biomolecular Sciences RUG/NN&AI/data'

def load_samples(path: str = "C:/Users/Sietse/Documents/Biomolecular Sciences RUG/NN&AI/data") -> List[ndarray]:
    samples = []
    for file_path in natsort.natsorted(glob.glob(path + "/*.npy", recursive=True)):
        samples.append(numpy.load(file_path))
    return samples

numpy.random.seed(69)

INSTRUMENTS_COUNT = 9


INPUT_LENGTH = 16
OUTPUT_LENGTH = 1

# If false than many to one
MANY_TO_MANY = False
X = []
Y = []

for sample in load_samples():
    xy_pair_count = int(len(sample) / (INPUT_LENGTH + OUTPUT_LENGTH)) # 16 models + 1
    i = 0
    for _ in range(xy_pair_count):
        x = []
        y = []
        for _ in range(INPUT_LENGTH):
            x.append(sample[i])
            i += 1
        for _ in range(OUTPUT_LENGTH):
            if not MANY_TO_MANY:
                Y.append(sample[i])
            else:
                y.append(sample[i])
            i += 1
        X.append(x)
        if MANY_TO_MANY:
            Y.append(y)

X = numpy.array(X)
Y = numpy.array(Y)

print(X.shape)




Optimizer = RMSprop(learning_rate = 0.01, rho = 0.9, momentum = 0.0, epsilon = 10^-7) #add arguments to 

model = Sequential()
model.add(Dropout(0.2, input_shape=(X.shape[1], INSTRUMENTS_COUNT)))
model.add(LSTM(64, input_shape = (X.shape[1], INSTRUMENTS_COUNT),
               return_sequences = True, recurrent_regularizer = 'l2')) #The input shape will be a problem, I think, as it does not allow for different lengths of the examples like this
model.add(Dropout(0.5))
# models.add(LSTM(64, return_sequences = True))
# models.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(32, activation = 'relu', kernel_regularizer = 'l2'))
model.add(Dropout(0.5))
# models.add(Dense(16, activation = 'relu', kernel_regularizer = 'l2'))
# models.add(Dropout(0.5))
model.add(Dense(9, activation = 'sigmoid'))

# models.compile(loss = BinaryCrossentropy(from_logits = False), optimizer = SGD())
model.compile(
    loss=BinaryCrossentropy(from_logits=False),
    optimizer=RMSprop(learning_rate=0.001, momentum=0.9),
    metrics = ['accuracy'])
model.summary()

model.fit(X, Y, batch_size = 100, epochs = 200, verbose = 1, validation_split = 0.2)

