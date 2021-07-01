import shutil

import numpy
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, Dropout, TimeDistributed, Bidirectional
from tensorflow.python.keras.losses import BinaryCrossentropy

from nn.preprocessing import load_X_Y, HIGH_TOM, LOW_MID_TOM, HIGH_FLOOR_TOM, CRASH, RIDE

numpy.random.seed(69)

REMOVE_INSTRUMENT_INDICES = [
    # HIGH_TOM[1],
    # LOW_MID_TOM[1],
    # HIGH_FLOOR_TOM[1],
    # CRASH[1],
    # RIDE[1]
]
INSTRUMENTS_COUNT = 9-len(REMOVE_INSTRUMENT_INDICES)

INPUT_LENGTH = 16

# If false than many to one
MANY_TO_MANY = False

if MANY_TO_MANY:
    OUTPUT_LENGTH = INPUT_LENGTH
else:
    OUTPUT_LENGTH = 1

try:
    shutil.rmtree("logs")
except:
    print("Did not remove logs folder (doesn't exist)")

X, Y = load_X_Y(many_to_many=MANY_TO_MANY,
                input_length=INPUT_LENGTH,
                output_length=OUTPUT_LENGTH,
                remove_instrument_indices=REMOVE_INSTRUMENT_INDICES,
                generate_shifted_samples=False,
                min_non_zero_entries=2)

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

model = Sequential(name="drum_prediction")
model.add(Dropout(0.5, input_shape=(INPUT_LENGTH, INSTRUMENTS_COUNT)))
model.add(Bidirectional(LSTM(
    units=64,
    return_sequences=True,
    recurrent_regularizer='l2')))
model.add(Dropout(0.5))

model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(64, return_sequences=MANY_TO_MANY))
model.add(Dropout(0.5))

dense1 = Dense(32, activation='relu', kernel_regularizer='l2')
if MANY_TO_MANY:
    model.add(TimeDistributed(dense1))
else:
    model.add(dense1)
model.add(Dropout(0.5))

dense2 = Dense(INSTRUMENTS_COUNT, activation='sigmoid')
if MANY_TO_MANY:
    model.add(TimeDistributed(dense2))
else:
    model.add(dense2)

# model.compile(loss = BinaryCrossentropy(from_logits = False), optimizer = SGD())
model.compile(
    loss=BinaryCrossentropy(from_logits=False),
    optimizer=RMSprop(learning_rate=0.001, momentum=0.9),
    metrics=['binary_accuracy'])
model.summary()
model.fit(
    X,
    Y,
    batch_size=200,
    epochs=1500,
    validation_split=0.2,
    callbacks=[TensorBoard(log_dir='logs/nn-drum-synthesis', histogram_freq=1)]
)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print(numpy.around(model.predict(numpy.array([X[0]])), 3))
print(Y[0])
