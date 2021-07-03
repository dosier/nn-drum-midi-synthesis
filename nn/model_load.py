from tensorflow.python.keras.models import model_from_json


def load_model_and_weights(path: str = ""):
    json_file = open(path+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(path+"model.h5")
    return model
