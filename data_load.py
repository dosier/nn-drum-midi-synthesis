from collections import OrderedDict
from os import listdir
from os.path import isfile, join

from constants import NOTE_OFF, NOTE_ON, input_length, INSTRUMENTS_COUNT, DATA_PATH, predict_length
from data_input_stream import DataInputStream


def read_data(path):
    last_tick = 0
    with open(path, 'rb') as f:
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
    return division_type, resolution, last_tick, OrderedDict(sorted(events.items()))


def read_file(path):
    (division_type, resolution, last_tick, events) = read_data(path)
    last_states = None
    try:
        x = []
        y = []
        if last_tick > input_length:
            for time_step in range(input_length + predict_length):
                if events.__contains__(time_step):
                    states = events[time_step]
                else:
                    states = [NOTE_OFF] * INSTRUMENTS_COUNT
                    if last_states is not None:
                        for i in range(INSTRUMENTS_COUNT):
                            states[i] = last_states[i]
                last_states = states
                if time_step < input_length:
                    x.append(states)
                else:
                    y.append(states)
            # TODO: not sure why some file are `empty`, error in midi conversion part?
            if all(v == 0 for v in x) or all(v == 0 for v in y):
                return division_type, resolution, None, None
            return division_type, resolution, x, y
    except:
        print("Failed to parse file " + path)
        return division_type, resolution, None, None


def load(path_to_dir):
    X = []
    Y = []
    for file in [f for f in listdir(path_to_dir) if isfile(join(path_to_dir, f))]:
        (_, _, x, y) = read_file(path_to_dir + "/" + file)
        if x is not None and y is not None:
            X.append(x)
            Y.append(y)
    return X, Y