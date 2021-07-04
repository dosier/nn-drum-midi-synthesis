import os
import numpy as np
from mido import MidiFile
from mido import MidiTrack
from mido import MetaMessage
from mido import bpm2tempo
from mido import Message


class MidiConverter:

    def __init__(self, path):

        self.path = path

        self.width = 9

        self.inst_map = [36, 38, 48, 45, 43, 46, 42, 49, 51]  # length = width

        try:
            os.mkdir(path, 0o755)
        except OSError:
            print('directory creation failed(maybe it exists already)')
        return

    def make_midi(self, timed_notes, bpm: int, filename: str) -> str:
        """
        :param timed_notes: a 2d numpy array of shape (time,instruments)
        :param bpm:
        :param filename:
        """
        if timed_notes.ndim != 2:
            raise ValueError('Invalid dimensions for `timed_notes` got {} expected 2!'.format(timed_notes.ndim))

        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm), time=0))
        track.append(Message('program_change', program=12, time=0))

        duration = 30
        base = 120  # change this if notes are too close/far apart

        c = 0
        for i in range(0, timed_notes.shape[0]):
            first_on = True
            first_off = True
            for instrument_idx in range(0, timed_notes.shape[1]):
                if timed_notes[i, instrument_idx] == 1:
                    if first_on:
                        first_on = False
                        track.append(Message('note_on', note=self.inst_map[instrument_idx], velocity=120, time=c))
                        c = 0
                    else:
                        track.append(Message('note_on', note=self.inst_map[instrument_idx], velocity=120, time=0))
            for instrument_idx in range(0, timed_notes.shape[1]):
                if first_off:
                    first_off = False
                    track.append(Message('note_off', note=self.inst_map[instrument_idx], velocity=120, time=duration))
                else:
                    track.append(Message('note_off', note=self.inst_map[instrument_idx], velocity=120, time=0))
            c += base - duration * 2
            if not first_off:
                c += duration

        filepath = self.path + filename
        mid.save(filepath)

        return filepath


if __name__ == "__main__":
    a = MidiConverter('/src/output')
    test_input = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 1, 0],
                           [1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
    a.make_midi(test_input, 60, 'test')