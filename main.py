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
        divisionType = dis.read_float()
        resolution = dis.read_int()
        readFully = False
        while not readFully:
            # the tick at which the note states for each of the instruments are read
            tick = dis.read_int()
            # the number of instruments which have a note on or off event here,
            # note: if a note for an instrument is on, it may span multiple ticks,
            #       only when a note off event is read, it should turn off (release) the note!
            notesOnOrOffCount = dis.read_byte()
            for i in range(notesOnOrOffCount):
                instrumentIndex = dis.read_byte()
                onOrOff = dis.read_boolean()


if __name__ == '__main__':
    read_data("/Users/stanvanderbend/IdeaProjects/nn-project-data/data/dat/generated/12_funk_81_beat_4-4.dat")
