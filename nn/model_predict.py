import numpy

from nn.model_load import load_model_and_weights
from nn.preprocessing import load_X_Y

X, Y = load_X_Y(True, 16, 16, [], 1)

print("Size of X {}".format(X.shape))
print("Size of Y {}".format(Y.shape))

model = load_model_and_weights("models/")
print(numpy.around(model.predict(numpy.array([X[0]])), 3)[0])
print(numpy.around(model.predict(numpy.array([X[0]])), 0)[0])
print(Y[0])