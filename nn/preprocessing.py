import glob
import math
from typing import List

import numpy
from music21 import converter
from music21.chord import Chord
from music21.note import Note
from music21.stream import Stream
from natsort import natsort
from numpy import ndarray

INSTRUMENTS_COUNT = 9

# The number of slices we cut a beat into
RESOLUTION = 4

KICK = ("KICK", 0)
SNARE = ("SNARE", 1)
HIGH_TOM = ("HIGH_TOM", 2)
LOW_MID_TOM = ("LOW_MID_TOM", 3)
HIGH_FLOOR_TOM = ("HIGH_FLOOR_TOM", 4)
OPEN_HI_HAT = ("OPEN_HI_HAT", 5)
CLOSED_HI_HAT = ("CLOSED_HI_HAT", 6)
CRASH = ("CRASH", 7)
RIDE = ("RIDE", 8)

instruments_map = {
    36: KICK,
    38: SNARE,
    40: SNARE,
    37: SNARE,
    48: HIGH_TOM,
    50: HIGH_TOM,
    45: LOW_MID_TOM,
    47: LOW_MID_TOM,
    43: HIGH_FLOOR_TOM,
    58: HIGH_FLOOR_TOM,
    46: OPEN_HI_HAT,
    26: OPEN_HI_HAT,
    42: CLOSED_HI_HAT,
    22: CLOSED_HI_HAT,
    44: CLOSED_HI_HAT,
    49: CRASH,
    55: CRASH,
    57: CRASH,
    52: CRASH,
    51: RIDE,
    59: RIDE,
    53: RIDE
}


def quantize_midi_files(in_files_path: str = "data/midi/original", out_files_path="data/midi/quantized"):
    for file_path in natsort.natsorted(glob.glob(in_files_path + "/*.mid", recursive=True)):
        print("Quantize-ing midi file {}".format(file_path))
        score: Stream = converter.parse(file_path)
        score: Stream = score.quantize(quarterLengthDivisors=[RESOLUTION, ], processDurations=False)
        if out_files_path is not None:
            out_file_path = file_path.replace(in_files_path, out_files_path)
            score.write(fmt="midi", fp=out_file_path)

def print_meta(in_files_path: str = "data/midi/quantized"):
    for file_path in natsort.natsorted(glob.glob(in_files_path + "/*.mid", recursive=True)):
        print("Parsing file {}".format(file_path))
        midi: Stream = converter.parse(file_path)
        print("\t quarterLength = {}".format(midi.quarterLength))
        print("\t measureNumber = {}".format(midi.measureNumber))


def process_midi_files(in_files_path: str = "data/midi/quantized") -> List[ndarray]:
    min_time_steps = math.inf
    max_time_steps = -math.inf
    samples: List[ndarray] = []
    for file_path in natsort.natsorted(glob.glob(in_files_path + "/*.mid", recursive=True)):
        print("Parsing file {}".format(file_path))
        bars: List[Bar] = []
        current_bar: Bar = Bar(1)
        chord: Chord
        for chord in converter.parse(file_path).chordify().flat.notes:
            onset = int(chord.offset * RESOLUTION)
            if onset >= current_bar.count * 16:
                bars.append(current_bar)
                current_bar = Bar(current_bar.count + 1)
            note: Note
            for note in chord:
                (instrument_name, instrument_idx) = instruments_map[note.pitch.ps]
                current_bar.add(onset % 16, instrument_idx)
                # print("\tBar {}\tOffset {}\tInstrument {}".format(current_bar.count, onset % 16, instrument_name))
        time_steps = len(bars) * 16
        min_time_steps = min(min_time_steps, time_steps)
        max_time_steps = max(max_time_steps, time_steps)
        slices = numpy.zeros(shape=(time_steps, INSTRUMENTS_COUNT))
        i = 0
        for bar in bars:
            for j in range(16):
                slices[i] = bar.slices[j]
                i += 1
        samples.append(slices)
    print("min_time_steps = {}".format(min_time_steps))
    print("max_time_steps = {}".format(max_time_steps))
    return samples


class Bar:
    slices: List[List[int]]

    def __init__(self, count: int):
        self.count = count
        # create a 2D array of shape (16, 9)
        self.slices = []
        for _ in range(16):
            notes = [0] * INSTRUMENTS_COUNT
            self.slices.append(notes)

    def add(self, offset: int, instrument_idx: int):
        self.slices[offset][instrument_idx] = 1


def save_as_numpy(samples: List[ndarray], path: str = "data/numpy"):
    i = 0
    for sample in samples:
        file_path = path + "/{}".format(i)
        numpy.save(file_path, sample)
        i += 1


def load_samples(path: str = "data/numpy") -> List[ndarray]:
    samples = []
    for file_path in natsort.natsorted(glob.glob(path + "/*.npy", recursive=True)):
        samples.append(numpy.load(file_path))
    return samples


def load_X_Y(
        many_to_many: bool,
        input_length: int,
        output_length: int,
        remove_instrument_indices: List[int],
        min_non_zero_entries: int,
        generate_shifted_samples=False
) -> (numpy.ndarray, numpy.ndarray):
    """
    Loads X and Y.

    :param many_to_many: if true predict sequences, else predict single time step
    :param input_length: the number of time_steps to input to the model
    :param output_length: the number of time_steps the model should output
    :param remove_instrument_indices: the indices of the instruments to be omitted from the samples
    :param min_non_zero_entries: the minimum number of non-zero entries in x and y (otherwise omitted)
    :param generate_shifted_samples: whether each file should create shifted examples (highly increases size of X)
    :return: (X, Y)
    """
    X = []
    Y = []
    skipped = 0
    for sample in load_samples():
        offset = 0
        # Only loops when generate_shifted_samples == True
        while True:
            xy_pair_count = int((len(sample) - offset) / (input_length + output_length))
            if xy_pair_count == 0:
                break
            if generate_shifted_samples:
                i = offset
                offset += 1
            else:
                i = 0

            for _ in range(xy_pair_count):
                x = []
                y = []
                for _ in range(input_length):
                    x.append(sample[i])
                    i += 1
                for _ in range(output_length):
                    y.append(sample[i])
                    i += 1
                if len(remove_instrument_indices) > 0:
                    for i in range(len(x)):
                        x.__setitem__(i, numpy.delete(x[i], remove_instrument_indices))
                    for i in range(len(y)):
                        y.__setitem__(i, numpy.delete(y[i], remove_instrument_indices))

                x_note_on_count = numpy.count_nonzero(x)
                if not many_to_many:
                    y_note_on_count = numpy.count_nonzero(y[0])
                else:
                    y_note_on_count = numpy.count_nonzero(y)
                if x_note_on_count < min_non_zero_entries or y_note_on_count < min_non_zero_entries:
                    skipped += 1
                    continue

                X.append(x)
                if not many_to_many:
                    Y.append(y[0])
                else:
                    Y.append(y)

            if not generate_shifted_samples:
                break
    print("Skipped {} entries because too many zeros in x or y".format(skipped))
    return numpy.array(X), numpy.array(Y)


# quantize_midi_files()
# save_as_numpy(process_midi_files())
# print_meta()