import keras_tuner
from keras_tuner import HyperParameters
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras.losses import BinaryCrossentropy

from nn.preprocessing import load_X_Y, HIGH_TOM, LOW_MID_TOM, HIGH_FLOOR_TOM, CRASH, RIDE

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
MANY_TO_MANY = True

if MANY_TO_MANY:
    OUTPUT_LENGTH = INPUT_LENGTH
    MIN_NON_ZERO = OUTPUT_LENGTH  # min amount of non-zero values in (OUTPUT_LENGTH, INSTRUMENTS_COUNT)
else:
    OUTPUT_LENGTH = 1
    MIN_NON_ZERO = int(INSTRUMENTS_COUNT / 2)  # min amount of non-zero values in (OUTPUT_LENGTH, )


def build_model(hp: HyperParameters):
    model = keras.Sequential()
    model.add(InputLayer(input_shape=(INPUT_LENGTH, INSTRUMENTS_COUNT)))
    model.add(LSTM(
        units=hp.Choice('lstm_units', [48, 64, 128, 256]),
        dropout=hp.Choice('lstm_dropout', [0.1, 0.2, 0.3, 0.4]),
        return_sequences=MANY_TO_MANY))
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
    max_trials=20
)

X, Y = load_X_Y(many_to_many=MANY_TO_MANY,
                input_length=INPUT_LENGTH,
                output_length=OUTPUT_LENGTH,
                remove_instrument_indices=REMOVE_INSTRUMENT_INDICES,
                min_non_zero_entries=MIN_NON_ZERO,
                generate_shifted_samples=False,
                path="../data/numpy")

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=69)

tuner.search(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test))
best_model = tuner.get_best_models()[0]

name = "keras_tuning_best_model"
# serialize models to JSON
model_json = best_model.to_json()
with open("{}.json".format(name), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
best_model.save_weights("{}.h5".format(name))
print("Saved models {} to disk".format(name))
