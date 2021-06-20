from os import listdir
from os.path import isfile, join

import keras as keras

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

PATH = "/Users/stanvanderbend/IdeaProjects/nn-project-data/data/dat"

def read_data(file):
    last_tick = 0
    with open(PATH+"/"+file, 'rb') as f:
        dis = DataInputStream(f)
        division_type = dis.read_float()
        resolution = dis.read_int()
        events = {}
        while True:
            try:
                # the tick at which the note states for each of the instruments are read
                tick = dis.read_int()
                last_tick += tick
                # the number of instruments which have a note on or off event here,
                # note: if a note for an instrument is on, it may span multiple ticks,
                #       only when a note off event is read, it should turn off (release) the note!
                notes_on_or_off_count = dis.read_byte()
                array = [INSTRUMENTS_COUNT]
                for i in range(notes_on_or_off_count):
                    instrument_index = dis.read_byte()
                    on_or_off = dis.read_boolean()
                    array[instrument_index] = on_or_off
                events[tick] = array
            # No more events to read now (struct doesn't support variable length read, bleh)
            except:
                break
    return last_tick, events


if __name__ == '__main__':
    for file in [f for f in listdir(PATH) if isfile(join(PATH, f))]:
        (last_tick, events) = read_data(file)
        print(last_tick)
