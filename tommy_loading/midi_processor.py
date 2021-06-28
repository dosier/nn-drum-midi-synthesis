import os
from typing import List

from mido import MidiFile
from music21 import converter


class MidiProcessor:

    def __init__(self, default, file_path, lower_bound, upper_bound, quantize_resolution):
        """
        Constructs a new MidiProcessor

        :param bool default: if `default==True`, use default values, else use given
        :param str file_path: the path to read MIDI files from (recursively)
        :param int lower_bound: discard files with less than `lower_bound` whitelisted messages
        :param int upper_bound: discard files with more than `upper_bound` whitelisted messages
        :param int quantize_resolution: the number of time steps a beat has to be split up into
        """
        if default:
            self.path = './files'
            self.lower_bound = 0
            self.upper_bound = 9999999
            self.quantize_resolution = 16
        else:
            self.path = file_path
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            self.quantize_resolution = quantize_resolution

        self.instruments_map = []           # variable containing the instrument library used in the midi set
        self.white_list = ['note_on']       # type of messages accepted
        self.rolling_total = False          # switch between absolute and relative time (false=relative)
        self.note_time_pairs_list = []
        self.timed_states_list = []
        self.state_vector_timeline = []
        return

    def load_midi_files(self):
        """
        Load MIDI files from `path`
        :return: a list of `MidiFile` entries
        """
        midi_files = []
        directory = os.fsencode(self.path)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".mid"):
                # print("found file " + filename)
                midi_files.append(MidiFile(self.path + '/' + filename))
        print('loaded ' + str(len(midi_files)) + ' MIDI files')
        return midi_files

    def quantize_folder(self):

        directory = os.fsencode(self.path)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".mid"):
                # print("found file " + filename)
                m = converter.parse(self.path + '/' + filename)
                m = m.quantize([self.quantize_resolution])
                m.write('midi', self.path + '/q' + filename)
                continue
            else:
                continue

        return

    def parse_note_time_pairs(self, messages):
        structure = []
        total = 0
        for message in messages:
            # print(msg.type)
            if self.is_whitelisted(message.type):
                if self.rolling_total:
                    total += message.time
                    structure.append([message.note, total])
                else:
                    structure.append([message.note, message.time])
            # print(msg.type + ' ' + str(msg.note) + ' ' + str(msg.time))

        # print('n of messages: ' + str(len(structure)))
        return structure

    def is_whitelisted(self, message_type_to_check):
        for message_type in self.white_list:
            if message_type_to_check == message_type:
                return True
        return False

    def generate_instrument_library(self, track_list):

        x: List[int] = [0 for _ in range(127)]

        for track in track_list:
            for h, tup in enumerate(track):
                # print(str(tup))
                x[tup[0]] += 1
        for i in range(127):
            if x[i] != 0:
                self.instruments_map.append(i)
        return

    def find_instrument_idx(self, time):
        for i, x in enumerate(self.instruments_map):
            if self.instruments_map[i] == time:
                return i
        print('library error')
        return 1

    def merge(self, note_time_pairs):
        """
        Generates a vectorized input from a list of (note, time) pairs
        where at index 0 is encoded the offset between this message and the next,
        and index 1..onwards are binary states representing whether a instrument
        at the respective index is playing.
        """

        result = []

        for (note, time) in note_time_pairs:
            x = [0 for _ in range(len(self.instruments_map) + 1)]
            x[0] = time  # time in slot 0
            x[self.find_instrument_idx(note)] = 1  # instrument played at 1, rest at 0
            result.append(x)

        return result

    def timeline(self, timed_states):  # extends a wrapped array into a timeseries. bigger list index is time

        result = []

        instruments_count = len(self.instruments_map)
        x = [0 for _ in range(instruments_count)]

        for timed_state in timed_states:
            if timed_state[0] != 0:
                result.append(x)
                x = [0 for _ in range(instruments_count)]
            for instrument_idx in range(instruments_count):
                x[instrument_idx] += timed_state[instrument_idx + 1]
            for _ in range(timed_state[0]):
                result.append([0 for x in range(instruments_count)])

        return result

    def generate_note_time_pairs(self):

        files = self.load_midi_files()

        min_length = 10000
        max_length = 0

        for h, sample in enumerate(files):
            for i, track in enumerate(sample.tracks):
                print('Track {}: {}'.format(i, track.name))
                self.note_time_pairs_list.append(self.parse_note_time_pairs(track))
                length = len(self.note_time_pairs_list[len(self.note_time_pairs_list) - 1])
                if length > self.upper_bound or length < self.lower_bound:
                    self.note_time_pairs_list.pop()
                else:
                    min_length = min(min_length, length)
                    max_length = max(max_length, length)
        # print(str(self.v[0]))
        print('selected tracks: ' + str(len(self.note_time_pairs_list)))
        print('shortest file: ' + str(min_length))
        # print('longest file: ' + str(max_len))
        self.generate_instrument_library(self.note_time_pairs_list)  # necessary to call wrap() and timeline()
        print('instrument dictionary' + str(self.instruments_map))

        return

    def generate_timed_state_vectors(self):

        for note_time_pairs in self.note_time_pairs_list:
            self.timed_states_list.append(self.merge(note_time_pairs))

        print(str(self.timed_states_list[0]))

        return

    def generate_state_vector_timeline(self):

        for timed_states in self.timed_states_list:
            self.state_vector_timeline.append(self.timeline(timed_states))

        print(str(self.state_vector_timeline[0]))

        return


if __name__ == "__main__":
    processor = MidiProcessor(True, "./files", 0, 0, 0)
    # q.quantize_folder()
    processor.generate_note_time_pairs()
    processor.generate_timed_state_vectors()
    processor.generate_state_vector_timeline()
