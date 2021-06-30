import shutil

import numpy

from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, Dropout, TimeDistributed
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import RMSprop

from nn.preprocessing import load_samples, load_X_Y

numpy.random.seed(69)

INSTRUMENTS_COUNT = 9

INPUT_LENGTH = 16
OUTPUT_LENGTH = 16

# If false than many to one
MANY_TO_MANY = True


try:
    shutil.rmtree("logs")
except:
    print("Did not remove logs folder (doesn't exist)")

X, Y = load_X_Y(MANY_TO_MANY, INPUT_LENGTH, OUTPUT_LENGTH)

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

model = Sequential(name="drum_prediction")
model.add(Dropout(0.2, input_shape=(X.shape[1], INSTRUMENTS_COUNT)))
model.add(LSTM(64, input_shape=(X.shape[1], INSTRUMENTS_COUNT),
               return_sequences=True,
               recurrent_regularizer='l2'))  # The input shape will be a problem, I think, as it does not allow for different lengths of the examples like this
model.add(Dropout(0.5))

model.add(LSTM(64, return_sequences=MANY_TO_MANY))
model.add(Dropout(0.5))

dense1 = Dense(32, activation='relu', kernel_regularizer='l2')
if MANY_TO_MANY:
    model.add(TimeDistributed(dense1))
else:
    model.add(dense1)
model.add(Dropout(0.5))

dense2 = Dense(9, activation='sigmoid')
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
