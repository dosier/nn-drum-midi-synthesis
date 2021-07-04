import numpy
from tensorflow.python.keras.optimizer_v2.nadam import Nadam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from nn.model_controller import ModelController
from nn.preprocessing import HIGH_TOM, LOW_MID_TOM, HIGH_FLOOR_TOM, CRASH, RIDE

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

model.load_weights()


X = [[
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]]
values = []
x = numpy.array(X)
print(numpy.around(model.model.predict(x)))

model.plot()