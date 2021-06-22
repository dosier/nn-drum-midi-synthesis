import numpy
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, LSTM, Dropout, Activation

from constants import time_steps, INSTRUMENTS_COUNT, DATA_PATH, SAVE_PATH
from data_load import load

if __name__ == '__main__':
    batches = {}

    (X, Y) = load(DATA_PATH)
    X = numpy.array(X)
    Y = numpy.array(Y)

    print("Size of X {}".format(X.shape))
    print("Size of Y {}".format(Y.shape))

    model = Sequential()
    model.add(LSTM(512, input_shape=(time_steps, INSTRUMENTS_COUNT)))
    model.add(Dropout(0.75))  # 2nd layer of Dropout
    model.add(Dense(INSTRUMENTS_COUNT))  # a dense layer of 3 tensors
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    model.summary()

    tensorboard = TensorBoard(log_dir='./logs/nn-drum-synthesis', histogram_freq=1)

    model.fit(X, Y, batch_size=200, epochs=200, callbacks=[tensorboard], validation_split=0.2)

    model.save(SAVE_PATH)
