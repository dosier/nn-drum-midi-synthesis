from tensorflow.python.keras.models import model_from_json


def load_model_and_weights(model_type: str, path: str = ""):
    json_file = open(path+"{}.json".format(model_type), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(path+"{}.h5".format(model_type))
    return model
