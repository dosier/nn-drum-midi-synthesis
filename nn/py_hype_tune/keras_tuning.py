import random

import keras_tuner
import numpy
import tensorflow
from keras_tuner import HyperParameters
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional
from tensorflow.python.keras.losses import BinaryCrossentropy

from nn.preprocessing import load_X_Y, HIGH_TOM, LOW_MID_TOM, HIGH_FLOOR_TOM, CRASH, RIDE, shuffle_X_Y, to_data_set

REMOVE_INSTRUMENT_INDICES = [
    HIGH_TOM[1],
    LOW_MID_TOM[1],
    HIGH_FLOOR_TOM[1],
    CRASH[1],
    RIDE[1]
]
INSTRUMENTS_COUNT = 9 - len(REMOVE_INSTRUMENT_INDICES)

INPUT_LENGTH = 16

# If false than many to one
MANY_TO_MANY = False

if MANY_TO_MANY:
    OUTPUT_LENGTH = INPUT_LENGTH
    MIN_NON_ZERO = OUTPUT_LENGTH  # min amount of non-zero values in (OUTPUT_LENGTH, INSTRUMENTS_COUNT)
    MAX_CONSECUTIVE_DUPLICATES = 5
else:
    OUTPUT_LENGTH = 1
    MIN_NON_ZERO = int(INSTRUMENTS_COUNT / 2)  # min amount of non-zero values in (OUTPUT_LENGTH, )
    MAX_CONSECUTIVE_DUPLICATES = INPUT_LENGTH * 5

if MANY_TO_MANY:
    name = "mtm_"
else:
    name = "mto_"
name += str(INSTRUMENTS_COUNT) + "_"
name += str(MIN_NON_ZERO)
units_scaling = [1.0, 0.75, 0.5, 0.25, 0.25, 0.25]


def build_model(hp: HyperParameters):
    model = keras.Sequential()
    model.add(InputLayer(input_shape=(INPUT_LENGTH, INSTRUMENTS_COUNT)))

    lstm_bidirectional = hp.Choice('lstm_bidirectional', [True, False])
    lstm_units = hp.Choice('lstm_units', [128, 256, 512])
    lstm_dropout = hp.Choice('lstm_dropout', [0.0, 0.1, 0.2, 0.3])
    n_lstm_layers = hp.Choice('n_lstm_layers', [2, 3, 4, 5])
    for i in range(n_lstm_layers):
        if i < n_lstm_layers - 1:
            return_sequence = True
        else:
            return_sequence = MANY_TO_MANY
        lstm_layer = LSTM(
            units=int(lstm_units * units_scaling[i]),
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


project_name = name
directory = "results"
tuners = [
    keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective='val_binary_accuracy',
        max_trials=100,
        project_name=project_name,
        directory=directory+"/randomized_search"
    ),
    keras_tuner.Hyperband(
        hypermodel=build_model,
        objective='val_binary_accuracy',
        max_epochs=100,
        project_name=project_name,
        directory=directory+"/hyperband"
    ),
    keras_tuner.BayesianOptimization(
        hypermodel=build_model,
        objective='val_binary_accuracy',
        max_trials=100,
        project_name=project_name,
        directory=directory + "/bayes"
    )
]
tuner = tuners[2]

tuner.search_space_summary()

X, Y = load_X_Y(many_to_many=MANY_TO_MANY,
                input_length=INPUT_LENGTH,
                output_length=OUTPUT_LENGTH,
                remove_instrument_indices=REMOVE_INSTRUMENT_INDICES,
                min_non_zero_entries=MIN_NON_ZERO,
                max_consecutive_duplicates=MAX_CONSECUTIVE_DUPLICATES,
                generate_shifted_samples=False,
                path="../data/numpy")

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

shuffle_X_Y(X, Y)

train, test = to_data_set(X, Y)

tuner.search(
    train,
    validation_data=test,
    batch_size=100,
    epochs=100,
    use_multiprocessing=True,
    workers=3,
    callbacks=[
        TensorBoard(log_dir="logs/"+project_name),
        EarlyStopping(
            monitor='val_binary_accuracy', min_delta=0.001, patience=50, verbose=1,
            mode='auto', baseline=0.3, restore_best_weights=True
        )
    ]
)
best_model = tuner.get_best_models()[0]

if isinstance(tuner, keras_tuner.RandomSearch):
    name += "RS_"
elif isinstance(tuner, keras_tuner.Hyperband):
    name += "HB_"
elif isinstance(tuner, keras_tuner.BayesianOptimization):
    name += "BO_"
else:
    name += "unknown_"
model_json = best_model.to_json()
with open("{}.json".format(name), "w") as json_file:
    json_file.write(model_json)
best_model.save_weights("{}.h5".format(name))
print("Saved model {} to disk".format(name))
