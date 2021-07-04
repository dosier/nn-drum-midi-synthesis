import os
import random
import shutil
from typing import List, Optional

import numpy
from IPython.core.display import SVG
from numpy import ndarray
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Dense, TimeDistributed, Bidirectional
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.utils.vis_utils import plot_model, model_to_dot

from nn.preprocessing import load_X_Y, shuffle_X_Y


class ModelController:

    def __init__(self,
                 batch_size: int,
                 train_epochs: int,
                 seed: int,
                 input_length: int,
                 many_to_many: bool,
                 remove_instrument_indices: List[int],
                 optimizer: OptimizerV2,
                 first_lstm_bidirectional: bool,
                 n_lstm_layers: int,
                 lstm_dropout: float,
                 lstm_units: List[int],
                 data_path: str = "../data/numpy",
                 logs_path: str = "logs"):
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.seed = seed
        self.instruments_count = 9 - len(remove_instrument_indices)
        self.input_length = input_length
        if many_to_many:
            self.output_length = input_length
            # min amount of non-zero values in (OUTPUT_LENGTH, INSTRUMENTS_COUNT)
            self.min_non_zero_entries = input_length
            self.max_consecutive_duplicates = 5
        else:  # many_to_one
            self.output_length = 1
            # min amount of non-zero values in (OUTPUT_LENGTH, )
            self.min_non_zero_entries = int(self.instruments_count / 2)
            self.max_consecutive_duplicates = input_length * 5
        self.many_to_many = many_to_many
        self.remove_instrument_indices = remove_instrument_indices
        self.optimizer = optimizer
        self.n_lstm_layers = n_lstm_layers
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.first_lstm_bidirectional = first_lstm_bidirectional
        self.data_path = data_path
        self.logs_path = logs_path
        if many_to_many:
            name = "MM"
        else:
            name = "MO"
        name += "_" + str(self.instruments_count) + ""
        self.name = name
        self.model: Optional[Sequential] = None

    def build(self):
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        try:
            shutil.rmtree(self.logs_path + "/{}".format(self.name))
        except:
            print("Did not remove logs folder (doesn't exist)")

        model = Sequential()
        model.add(InputLayer(input_shape=(self.input_length, self.instruments_count)))
        for idx in range(self.n_lstm_layers):
            if idx < self.n_lstm_layers - 1:
                return_sequence = True
            else:
                return_sequence = self.many_to_many
            lstm_layer = LSTM(
                units=self.lstm_units[idx],
                dropout=self.lstm_dropout,
                return_sequences=return_sequence)
            if idx == 0 and self.first_lstm_bidirectional:
                model.add(Bidirectional(lstm_layer))
            else:
                model.add(lstm_layer)
        if self.many_to_many:
            model.add(TimeDistributed(Dense(self.instruments_count, activation='sigmoid')))
        else:
            model.add(Dense(self.instruments_count, activation='sigmoid'))
        model.compile(
            loss=BinaryCrossentropy(),
            metrics=['accuracy', 'binary_accuracy'],
            optimizer=self.optimizer
        )
        model.summary()
        self.model = model

    def train(self):
        X: ndarray
        Y: ndarray
        X, Y = load_X_Y(self.many_to_many,
                        self.input_length,
                        self.output_length,
                        self.remove_instrument_indices,
                        self.min_non_zero_entries,
                        max_consecutive_duplicates=self.max_consecutive_duplicates,
                        path=self.data_path)
        shuffle_X_Y(X, Y)
        print("Size of X {}".format(X.shape))
        print("Size of Y {}".format(Y.shape))
        self.model.fit(
            X,
            Y,
            batch_size=self.batch_size,
            epochs=self.train_epochs,
            validation_split=0.2,
            # validation_data=(X_test, Y_test),
            callbacks=[
                TensorBoard(log_dir=self.logs_path + "/{}".format(self.name), histogram_freq=1),
                EarlyStopping(
                    monitor='val_binary_accuracy', min_delta=0.001, patience=100, verbose=1,
                    mode='auto', baseline=0.3, restore_best_weights=True
                )
            ]
        )

    def load_weights(self, folder_path: str = ""):
        self.model.load_weights(folder_path + "{}.h5".format(self.name))

    def save(self, folder_path: str = ""):
        name = self.name
        name += "_N" + str(self.min_non_zero_entries) + "]"
        name += "_Seed[" + str(self.seed) + "]"
        model_configuration = self.model.to_json()
        with open(folder_path + "{}.json".format(self.name), "w") as json_file:
            json_file.write(model_configuration)
        self.model.save_weights(folder_path + "{}.h5".format(self.name))
        print("Saved model {} to disk".format(self.name))

    def plot(self, folder_path: str = "plots/"):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plot_model(self.model,
                   to_file=folder_path + '{}.svg'.format(self.name),
                   dpi=None,
                   show_shapes=True,
                   show_layer_names=False)
        SVG(model_to_dot(self.model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))