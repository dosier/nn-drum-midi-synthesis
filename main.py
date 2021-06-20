from collections import OrderedDict
from os import listdir
from os.path import isfile, join

import keras as keras
import numpy
from keras import Input, Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, LSTM, Embedding, Dropout, Activation

from data_input_stream import DataInputStream

KICK = 0
SNARE = 1
HIGH_TOM = 2
LOW_MID_TOM = 3
HIGH_FLOOR_TOM = 4
OPEN_HI_HA = 5
CLOSED_HI_HAT = 6
CRASH = 7
RIDE = 8
INSTRUMENTS_COUNT = 9

NOTE_OFF = 0
NOTE_ON = 1

time_steps = 50

DATA_PATH = "data"
SAVE_PATH = "model"

def read_data(file):
    last_tick = 0
    with open(DATA_PATH + "/" + file, 'rb') as f:
        dis = DataInputStream(f)
        division_type = dis.read_float()
        resolution = dis.read_int()
        events = {}
        while True:
            try:
                # the tick at which the note states for each of the instruments are read
                tick = dis.read_int()
                if tick >= last_tick:
                    last_tick = tick
                # the number of instruments which have a note on or off event here,
                # note: if a note for an instrument is on, it may span multiple ticks,
                #       only when a note off event is read, it should turn off (release) the note!
                notes_on_or_off_count = dis.read_byte()
                list = [NOTE_OFF] * 9
                for i in range(notes_on_or_off_count):
                    instrument_index = dis.read_byte()
                    on_or_off = dis.read_boolean()
                    if on_or_off:
                        list[instrument_index] = NOTE_ON
                    else:
                        list[instrument_index] = NOTE_OFF
                events[tick] = list
            # No more events to read now (struct doesn't support variable length read, bleh)
            except:
                break
    return last_tick, OrderedDict(sorted(events.items()))


def load():
    X = []
    Y = []
    for file in [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]:
        (last_tick, events) = read_data(file)
        last_states = None
        try:
            x = []
            if last_tick > time_steps:
                states = None
                for time_step in range(time_steps):
                    if events.__contains__(time_step):
                        states = events[time_step]
                    else:
                        states = [NOTE_OFF] * INSTRUMENTS_COUNT
                        if last_states is not None:
                            for i in range(INSTRUMENTS_COUNT):
                                states[i] = last_states[i]
                    last_states = states
                    x.append(states)
                if all(v == 0 for v in x):  # TODO: not sure why some file are `empty`, error in midi conversion part?
                    continue
                # batches[file] = dict((k, v) for k, v in events.items() if k <= time_steps)
                X.append(x)
                # for now only predict the next time step based on the previous time steps
                Y.append(states)
                print("Successfully parsed file " + file)
        except:
            print("Failed to parse file " + file)

    return X, Y


if __name__ == '__main__':
    batches = {}

    (X, Y) = load()
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
