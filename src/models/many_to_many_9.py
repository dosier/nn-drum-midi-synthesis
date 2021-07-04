import numpy

from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from model_controller import ModelController
from io.preprocessing import load_input_and_labels
from io.midi_converter import MidiConverter

model = ModelController(
    batch_size=200,
    train_epochs=200,
    seed=69,
    input_length=16,
    many_to_many=True,
    remove_instrument_indices=[],
    optimizer=RMSprop(),
    first_lstm_bidirectional=True,
    n_lstm_layers=5,
    lstm_dropout=0.3,
    lstm_units=[512, 384, 256, 128, 128]
)
model.build()

# model.train()
# model.save()

model_path = '../../results/many_to_many_9/'
model.load_weights(model_path)


X = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]

timed_notes = []
for time_step in X:
    timed_notes.append(time_step)

X = numpy.array([X])
for i in range(4):
    X = numpy.around(model.predict(X), 0)
    for time_step in X[0]:
        timed_notes.append(time_step)

midi_converter = MidiConverter(model_path)
midi_converter.make_midi(timed_notes=numpy.array(timed_notes), bpm=100, filename="MM_9.mid")

X, Y = load_input_and_labels(
    model.many_to_many,
    model.input_length,
    model.output_length,
    model.remove_instrument_indices,
    model.min_non_zero_entries,
    path="../../data/numpy")

last_x = X[-1]
last_Y = Y[-1]

timed_notes = []
for x in last_x:
    timed_notes.append(x.tolist())


X = numpy.array([last_x])
for i in range(1):
    X = numpy.around(model.predict(X), 0)
    for time_step in X[0]:
        timed_notes.append(time_step)

for x in last_x:
    timed_notes.append(x.tolist())
for y in last_Y:
    timed_notes.append(y.tolist())

midi_converter.make_midi(timed_notes=numpy.array(timed_notes), bpm=100, filename="MM_9_2.mid")


# model.plot(model_path)