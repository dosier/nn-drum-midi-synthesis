import random
import shutil

import numpy
from numpy import ndarray
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Dense, TimeDistributed
from tensorflow.python.keras.losses import BinaryCrossentropy

from nn.preprocessing import load_X_Y, HIGH_TOM, LOW_MID_TOM, HIGH_FLOOR_TOM, CRASH, RIDE, shuffle_X_Y

random.seed(69)
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
MANY_TO_MANY = True

if MANY_TO_MANY:
    OUTPUT_LENGTH = INPUT_LENGTH
    MIN_NON_ZERO = int(OUTPUT_LENGTH / 2)  # min amount of non-zero values in (OUTPUT_LENGTH, INSTRUMENTS_COUNT)
else:
    OUTPUT_LENGTH = 1
    MIN_NON_ZERO = int(INSTRUMENTS_COUNT / 2)  # min amount of non-zero values in (OUTPUT_LENGTH, )

try:
    shutil.rmtree("logs")
except:
    print("Did not remove logs folder (doesn't exist)")

X: ndarray
Y: ndarray
X, Y = load_X_Y(many_to_many=MANY_TO_MANY, input_length=INPUT_LENGTH, output_length=OUTPUT_LENGTH,
                remove_instrument_indices=REMOVE_INSTRUMENT_INDICES, min_non_zero_entries=MIN_NON_ZERO,
                max_consecutive_duplicates=5,
                generate_shifted_samples=False)

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

model = Sequential()
model.add(InputLayer(input_shape=(INPUT_LENGTH, INSTRUMENTS_COUNT)))
model.add(LSTM(
    units=256,
    dropout=0.2,
    return_sequences=MANY_TO_MANY))
if MANY_TO_MANY:
    model.add(TimeDistributed(Dense(INSTRUMENTS_COUNT, activation='sigmoid')))
else:
    model.add(Dense(INSTRUMENTS_COUNT, activation='sigmoid'))
model.compile(
    loss=BinaryCrossentropy(),
    metrics=['accuracy', 'binary_accuracy'],
    optimizer="Nadam"
)
model.summary()

shuffle_X_Y(X, Y)

model.fit(
    X,
    Y,
    batch_size=40,
    epochs=500,
    validation_split=0.2,
    # validation_data=(X_test, Y_test),
    callbacks=[
        TensorBoard(log_dir='logs/nn-drum-synthesis', histogram_freq=1),
        EarlyStopping(
            monitor='val_binary_accuracy', min_delta=0.001, patience=100, verbose=1,
            mode='auto', baseline=0.3, restore_best_weights=True
        )
    ]
)

# serialize models to JSON
model_json = model.to_json()
with open("models/server/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models.h5")
print("Saved models to disk")

print(numpy.around(model.predict(numpy.array([X[0]])), 3))
print(Y[0])
