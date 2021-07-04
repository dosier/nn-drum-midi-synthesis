from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from nn.model_controller import ModelController
from nn.preprocessing import HIGH_TOM, LOW_MID_TOM, HIGH_FLOOR_TOM, CRASH, RIDE

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

# model.train()
# model.save()

model.load_weights()
model.plot()
