import numpy

from nn.model_load import load_model_and_weights
from nn.preprocessing import load_X_Y
from nn.py_util.midi_converter import MidiConverter

model_path = "models/mm_4_03_52/MM_4"
model = load_model_and_weights(model_path)

X = [
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
timed_notes = []
for time_step in X:
    timed_notes.append(time_step)
repeat = 4
X = numpy.array([X])
for i in range(repeat):
    X = numpy.around(model.predict(X), 0)
    for time_step in X[0]:
        timed_notes.append(time_step)

midi_converter = MidiConverter(model_path+"/")
midi_converter.make_midi(timed_notes=numpy.array(timed_notes), bpm=100, filename="output.mid")