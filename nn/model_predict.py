import numpy
from tensorflow.python.keras.models import model_from_json

from nn.preprocessing import load_X_Y

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

X, Y = load_X_Y(True, 16, 16)

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

print(numpy.around(model.predict(numpy.array([X[0]])), 3)[0])
print(numpy.around(model.predict(numpy.array([X[0]])), 0)[0])
print(Y[0])