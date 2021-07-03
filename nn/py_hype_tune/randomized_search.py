# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:45:44 2021

@author: Sietse
"""
import random
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
import glob
import math
from typing import List
import natsort
import numpy
from numpy import ndarray
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras.optimizer_v2.nadam import Nadam

from nn.preprocessing import load_samples, load_X_Y, HIGH_TOM, LOW_MID_TOM, HIGH_FLOOR_TOM, CRASH, RIDE

numpy.random.seed(69)

REMOVE_INSTRUMENT_INDICES = [
    HIGH_TOM[1],
    LOW_MID_TOM[1],
    HIGH_FLOOR_TOM[1],
    CRASH[1],
    RIDE[1]
]
INSTRUMENTS_COUNT = 9-len(REMOVE_INSTRUMENT_INDICES)

INPUT_LENGTH = 16

# If false than many to one
MANY_TO_MANY = False

if MANY_TO_MANY:
    OUTPUT_LENGTH = INPUT_LENGTH
else:
    OUTPUT_LENGTH = 1

X, Y = load_X_Y(many_to_many=MANY_TO_MANY,
                input_length=INPUT_LENGTH,
                output_length=OUTPUT_LENGTH,
                remove_instrument_indices=REMOVE_INSTRUMENT_INDICES,
                min_non_zero_entries=2,
                generate_shifted_samples=False,
                path="../data/numpy")

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=69)

tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1)


def create_model(optimizer=RMSprop(), dropout=0.5,
                 n_layers=2, r_reg='l2', k_reg='l2', layer_width=64):
    model = Sequential()
    model.add(LSTM(layer_width,
                   input_shape=(INPUT_LENGTH, INSTRUMENTS_COUNT),
                   return_sequences=True,
                   recurrent_regularizer=r_reg,
                   kernel_regularizer=k_reg))
    model.add(Dropout(dropout))

    last_layer_width = layer_width
    for layer in range(n_layers - 2):
        if layer == 0:
            last_layer_width = min(32, int(layer_width/1.5))
        if layer == 1:
            last_layer_width = min(32, int(layer_width/2))
        elif layer == 2:
            last_layer_width = min(32, int(layer_width/2.5))
        elif layer == 3:
            last_layer_width = min(32, int(layer_width/3))
        model.add(LSTM(last_layer_width,
                       return_sequences=True,
                       recurrent_regularizer=r_reg,
                       kernel_regularizer=k_reg))
        model.add(Dropout(dropout))

    model.add(LSTM(min(32, last_layer_width),
                   return_sequences=MANY_TO_MANY,
                   recurrent_regularizer=r_reg,
                   kernel_regularizer=k_reg))
    model.add(Dropout(dropout))

    dense = Dense(INSTRUMENTS_COUNT, activation='sigmoid')
    if MANY_TO_MANY:
        model.add(TimeDistributed(dense))
    else:
        model.add(dense)
    model.compile(
        loss=BinaryCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics=['accuracy', 'binary_accuracy'])

    return model


SKL_model = KerasClassifier(build_fn=create_model, verbose=2)

param_grid = {
    'optimizer': [RMSprop(), Nadam()],
    'dropout': [0.2, 0.3, 0.4],
    'n_layers': [2, 3, 4],
    'r_reg': ['l1', 'l2'],
    'k_reg': ['l1', 'l2'],
    'layer_width': [64, 128, 256]
}

k_folds = 10

grid_search = RandomizedSearchCV(SKL_model, param_grid, n_iter=50, n_jobs=1,
                                 verbose=10, return_train_score=True, cv=k_folds)

grid_results = grid_search.fit(X_train, Y_train)
print('Best score: ' + grid_results.best_score_ + '. Params:' + grid_results.best_params_)
print(grid_results.best_score_)
print(grid_results.best_params_)