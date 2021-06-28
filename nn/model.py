# The number of instruments used throughout all the samples
import numpy
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp

INSTRUMENTS_COUNT = 9

# The number of slices we cut a beat into
RESOLUTION = 16

# The number of beats we want to make predictions from
INPUT_BEATS = 1
# The length (n.o. time steps) to makes predictions from
INPUT_LENGTH = RESOLUTION * INPUT_BEATS

# If false than many to one
MANY_TO_MANY = False

model = Sequential(name="drum_prediction")
model.add(LSTM(units=512, input_shape=(INPUT_LENGTH, INSTRUMENTS_COUNT)))
model.add(Dropout(rate=0.5))
model.add(Dense(INSTRUMENTS_COUNT))
model.add(Activation("sigmoid"))
model.compile(
    loss=BinaryCrossentropy(from_logits=False),
    optimizer=RMSProp(learning_rate=0.001, momentum=0.9)
)
model.summary()

X = [
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
]

Y = [[1, 0, 0, 0, 0, 0, 1, 0, 0]]

X = numpy.array(X)
Y = numpy.array(Y)

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

model.fit(X, Y, epochs=20)

print(model.predict(X))