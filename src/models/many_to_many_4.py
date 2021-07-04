import numpy
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from model_controller import ModelController
from io.preprocessing import HIGH_TOM, LOW_MID_TOM, HIGH_FLOOR_TOM, CRASH, RIDE
from io.midi_converter import MidiConverter

model = ModelController(
    batch_size=200,
    train_epochs=200,
    seed=69,
    input_length=16,
    many_to_many=True,
    remove_instrument_indices=[
        HIGH_TOM[1],
        LOW_MID_TOM[1],
        HIGH_FLOOR_TOM[1],
        CRASH[1],
        RIDE[1]
    ],
    optimizer=RMSprop(),
    first_lstm_bidirectional=True,
    n_lstm_layers=3,
    lstm_dropout=0,
    lstm_units=[512, 384, 256]
)
model.build()

model_path = '../../results/many_to_many_4/'
model.load_weights(model_path)

X = [
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
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
X = numpy.array([X])
for i in range(4):
    X = numpy.around(model.model.predict(X), 0)
    for time_step in X[0]:
        timed_notes.append(time_step)

midi_converter = MidiConverter(model_path)
midi_converter.make_midi(timed_notes=numpy.array(timed_notes), bpm=100, filename="MM_4.mid")

model.plot(folder_path=model_path)
