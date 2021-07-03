import random

import keras_tuner
import numpy
from keras_tuner import HyperParameters
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional
from tensorflow.python.keras.losses import BinaryCrossentropy

from nn.preprocessing import load_X_Y, HIGH_TOM, LOW_MID_TOM, HIGH_FLOOR_TOM, CRASH, RIDE, shuffle_X_Y

REMOVE_INSTRUMENT_INDICES = [
    # HIGH_TOM[1],
    # LOW_MID_TOM[1],
    # HIGH_FLOOR_TOM[1],
    # CRASH[1],
    # RIDE[1]
]
INSTRUMENTS_COUNT = 9 - len(REMOVE_INSTRUMENT_INDICES)

INPUT_LENGTH = 16

# If false than many to one
MANY_TO_MANY = True

if MANY_TO_MANY:
    OUTPUT_LENGTH = INPUT_LENGTH
    MIN_NON_ZERO = OUTPUT_LENGTH  # min amount of non-zero values in (OUTPUT_LENGTH, INSTRUMENTS_COUNT)
else:
    OUTPUT_LENGTH = 1
    MIN_NON_ZERO = int(INSTRUMENTS_COUNT / 2)  # min amount of non-zero values in (OUTPUT_LENGTH, )

units_scaling = [1.0, 0.75, 0.5, 0.25, 0.25, 0.25]


def build_model(hp: HyperParameters):
    model = keras.Sequential()
    model.add(InputLayer(input_shape=(INPUT_LENGTH, INSTRUMENTS_COUNT)))

    lstm_bidirectional = hp.Choice('lstm_bidirectional', [True, False])
    lstm_units = hp.Choice('lstm_units', [128, 256, 512])
    lstm_dropout = hp.Choice('lstm_dropout', [0.0, 0.1])
    n_lstm_layers = hp.Choice('n_lstm_layers', [2, 3, 4, 5])
    for i in range(n_lstm_layers):
        if i < n_lstm_layers - 1:
            return_sequence = True
        else:
            return_sequence = MANY_TO_MANY
        lstm_layer = LSTM(
            units=int(lstm_units / units_scaling[i]),
            dropout=lstm_dropout,
            return_sequences=return_sequence)
        if i == 0 and lstm_bidirectional:
            model.add(Bidirectional(lstm_layer))
        else:
            model.add(lstm_layer)
    if MANY_TO_MANY:
        model.add(TimeDistributed(Dense(INSTRUMENTS_COUNT, activation='sigmoid')))
    else:
        model.add(Dense(INSTRUMENTS_COUNT, activation='sigmoid'))
    model.compile(
        loss=BinaryCrossentropy(),
        metrics=['accuracy', 'binary_accuracy'],
        optimizer=hp.Choice('optimizer', ["RMSprop", "Nadam"])
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective='val_binary_accuracy',
    max_trials=100
)

X, Y = load_X_Y(many_to_many=MANY_TO_MANY,
                input_length=INPUT_LENGTH,
                output_length=OUTPUT_LENGTH,
                remove_instrument_indices=REMOVE_INSTRUMENT_INDICES,
                min_non_zero_entries=MIN_NON_ZERO,
                max_consecutive_duplicates=5,
                generate_shifted_samples=False,
                path="../data/numpy")

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

shuffle_X_Y(X, Y)

tuner.search(X, Y, epochs=120, validation_split=0.2)
best_model = tuner.get_best_models()[0]

if MANY_TO_MANY:
    name = "mtm_"
else:
    name = "mto_"
name += "RS_"
name += str(INSTRUMENTS_COUNT) + "_"
name += str(MIN_NON_ZERO)


model_json = best_model.to_json()
with open("{}.json".format(name), "w") as json_file:
    json_file.write(model_json)
best_model.save_weights("{}.h5".format(name))
print("Saved model {} to disk".format(name))
