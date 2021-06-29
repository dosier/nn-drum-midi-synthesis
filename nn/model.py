# The number of instruments used throughout all the samples
import numpy
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp

from nn.preprocessing import load_samples

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

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

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
model.fit(X, Y, batch_size=100, epochs=200, validation_split=0.2)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print(model.predict(numpy.array([X[0]])))
print(Y[0])