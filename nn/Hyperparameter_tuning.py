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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
import glob
import math
from typing import List
import natsort
import numpy
from numpy import ndarray
from sklearn.model_selection import RandomizedSearchCV


directory = 'C:/Users/Sietse/Documents/Biomolecular_Sciences_RUG/NN_AI/data'

def load_samples(path: str = "C:/Users/Sietse/Documents/Biomolecular_Sciences_RUG/NN_AI/data") -> List[ndarray]:
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
    xy_pair_count = int(len(sample) / (INPUT_LENGTH + OUTPUT_LENGTH)) # 16 predict + 1
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 69)

tensorboard_callback = TensorBoard(log_dir='C:/Users/Sietse/Documents/Biomolecular_Sciences_RUG/NN_AI/logs', histogram_freq=1)

Optimizer = RMSprop(learning_rate = 0.001, rho = 0.9, momentum = 0.0, epsilon = 10^-7) #add arguments to 

def create_model(optimizer = RMSprop(), initial_dropout = 0.2, dropout = 0.5,
                 n_layers = 2, r_reg = 'l2', k_reg = 'l2', layer_width = 64):
    model = Sequential()    
    model.add(Dropout(initial_dropout, input_shape=(X.shape[1], INSTRUMENTS_COUNT)))
    model.add(LSTM(layer_width, input_shape = (X.shape[1], INSTRUMENTS_COUNT),
                    return_sequences = True, recurrent_regularizer = r_reg,
                    kernel_regularizer = k_reg))
    model.add(Dropout(dropout))
    for layer in range(n_layers - 2):
        model.add(LSTM(layer_width, return_sequences = True,
                       recurrent_regularizer = r_reg, kernel_regularizer = k_reg))
        model.add(Dropout(dropout))
    model.add(LSTM(layer_width, recurrent_regularizer = r_reg, kernel_regularizer = k_reg))
    model.add(Dropout(dropout))
    # model.add(Dense(32, activation = 'relu', kernel_regularizer = 'l2'))
    # model.add(Dropout(0.5))
    # model.add(Dense(16, activation = 'relu', kernel_regularizer = 'l2'))
    # model.add(Dropout(0.5))
    model.add(Dense(9, activation = 'sigmoid'))
    
    model.compile(
        loss=BinaryCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics = ['accuracy'])
    
    return model

SKL_model = KerasClassifier(build_fn = create_model, verbose = 1)

param_grid = {
    'optimizer':[RMSprop(), RMSprop(learning_rate = 0.01), SGD()],
    'initial_dropout':[0.0, 0.1, 0.2],
    'dropout':[0.0, 0.25, 0.5],
    'n_layers':[2, 3],
    'r_reg':[None, 'l1', 'l2'],
    'k_reg':[None, 'l1', 'l2'],
    'layer_width':[32, 64, 128, 256, 512]    
    }

k_folds = 10

grid_search = RandomizedSearchCV(SKL_model, param_grid, n_iter = 100, n_jobs = -1, 
                                 verbose = 1, return_train_score = True, cv = k_folds)

grid_results = grid_search.fit(X_train, Y_train)
print('Best score: ' + grid_results.best_score_ + '. Params:' + grid_results.best_params_)
print(grid_results.best_score_)
print(grid_results.best_params_)
# model.fit(X, Y, batch_size = 100, epochs = 200, callbacks=[tensorboard_callback], verbose = 1, validation_split = 0.2)

