import glob
import math
import random
from typing import List

import numpy
import tensorflow
from music21 import converter
from music21.chord import Chord
from music21.note import Note
from music21.stream import Stream
from natsort import natsort
from numpy import ndarray

from nn.py_util.progress_bar import print_progress_bar

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
        stream: Stream = converter.parse(file_path)
        print("\tquarterLength {}".format(stream.quarterLength))
        bars: List[Bar] = []
        current_bar: Bar = Bar(1)
        chord: Chord
        for chord in stream.chordify().flat.notes:
            onset = int(chord.offset * RESOLUTION)
            if onset >= current_bar.count * 16:
                bars.append(current_bar)
                current_bar = Bar(current_bar.count + 1)
            note: Note
            for note in chord:
                (instrument_name, instrument_idx) = instruments_map[note.pitch.ps]
                current_bar.add(onset % 16, instrument_idx)
                print("\tBar {}\tOffset {}\tInstrument {}".format(current_bar.count, onset % 16, instrument_name))
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
        generate_shifted_samples: object = False,
        max_consecutive_duplicates: int = math.inf,
        path: str = "data/numpy"
) -> (numpy.ndarray, numpy.ndarray):
    """
    Loads X and Y.

    :param many_to_many: if true models sequences, else models single time step
    :param input_length: the number of time_steps to input to the models
    :param output_length: the number of time_steps the models should output
    :param remove_instrument_indices: the indices of the instruments to be omitted from the samples
    :param min_non_zero_entries: the minimum number of non-zero entries in x and y (otherwise omitted)
    :param generate_shifted_samples: whether each file should create shifted examples (highly increases size of X)
    :param max_consecutive_duplicates:
    :param path:
    :return: (X, Y)
    """
    X = []
    Y = []
    skipped = 0
    duplicated_removed = 0
    remove_duplicates = max_consecutive_duplicates != math.inf
    shifted_count = 0
    samples = load_samples(path)
    instruments_count = 9 - len(remove_instrument_indices)
    number_of_samples = len(samples)
    print("Loading X and Y data...")
    for idx in range(number_of_samples):
        print_progress_bar(idx, number_of_samples, prefix='Progress:', suffix='Complete', length=50)
        sample = samples[idx]
        offset = 0
        # Only loops when generate_shifted_samples == True
        while True:
            xy_pair_count = int((len(sample) - offset) / (input_length + output_length))
            if xy_pair_count == 0:
                break
            if generate_shifted_samples:
                i = offset
                offset += 8
            else:
                i = 0

            duplicate_count = 0
            last_x = None
            last_y = None
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

                if not many_to_many:
                    current_y = y[0]
                else:
                    current_y = y

                if remove_duplicates:
                    if last_x is not None and last_y is not None:
                        if is_duplicate(current_y, last_x, last_y, x, instruments_count):
                            duplicate_count += 1
                        else:
                            duplicate_count = 0
                        if duplicate_count >= max_consecutive_duplicates:
                            duplicated_removed += 1
                            continue
                    last_x = x.copy()
                    last_y = current_y.copy()

                X.append(x)
                if not many_to_many:
                    Y.append(current_y)
                else:
                    Y.append(current_y)

            if not generate_shifted_samples:
                break
        shifted_count += offset
    print("Finished X and Y data!")
    print("\tSkipped {} entries because too many zeros in x or y".format(skipped))
    if remove_duplicates:
        print(
            "\tRemoved {} duplicates because chain exceeded {}".format(duplicated_removed, max_consecutive_duplicates))
    if generate_shifted_samples:
        print("\tGenerated {} shifted samples from the input data".format(shifted_count))
    return numpy.array(X), numpy.array(Y)


def is_duplicate(y1, x2, y2, x1, instrument_count: int) -> bool:
    if len(x1) != len(x2) or len(y1) != len(y2):
        return False

    for i in range(len(x1)):
        for j in range(instrument_count):
            if x1[i][j] != x2[i][j]:
                return False
    for i in range(len(y1)):
        for j in range(instrument_count):
            if y1[i][j] != y2[i][j]:
                return False

    return True


def shuffle_X_Y(X, Y):
    indices = []
    for i in range(len(X)):
        indices.append(i)
    random.shuffle(indices)
    copy_X = numpy.copy(X)
    copy_Y = numpy.copy(Y)
    j = 0
    for i in indices:
        X[i] = copy_X[j]
        Y[i] = copy_Y[j]
        j += 1


def to_data_set(X, Y, test_split: float = 0.2, batch_size: int = 50):
    n_samples = len(X)
    length_validation = int(n_samples * test_split)
    test_examples = X[0:length_validation]
    test_labels = Y[0:length_validation]
    train_examples = X[length_validation:n_samples]
    train_labels = Y[length_validation:n_samples]
    train_dataset = tensorflow.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tensorflow.data.Dataset.from_tensor_slices((test_examples, test_labels))
    options = tensorflow.data.Options()
    # options.experimental_distribute.auto_shard_policy = tensorflow.data.experimental.AutoShardPolicy.OFF
    return train_dataset.with_options(options).cache().batch(batch_size), \
           test_dataset.with_options(options).cache().batch(batch_size)

# quantize_midi_files()
# process_midi_files()
# save_as_numpy(process_midi_files())
# print_meta()
