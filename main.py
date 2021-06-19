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


def read_data(path):
    with open(path, 'rb') as f:
        dis = DataInputStream(f)
        division_type = dis.read_float()
        resolution = dis.read_int()
        while True:
            try:
                # the tick at which the note states for each of the instruments are read
                tick = dis.read_int()
                # the number of instruments which have a note on or off event here,
                # note: if a note for an instrument is on, it may span multiple ticks,
                #       only when a note off event is read, it should turn off (release) the note!
                notes_on_or_off_count = dis.read_byte()
                for i in range(notes_on_or_off_count):
                    instrument_index = dis.read_byte()
                    on_or_off = dis.read_boolean()
            # No more events to read now (struct doesn't support variable length read, bleh)
            except:
                break


if __name__ == '__main__':
    read_data("/Users/stanvanderbend/IdeaProjects/nn-project-data/data/dat/generated/12_funk_81_beat_4-4.dat")
