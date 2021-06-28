import socket

from keras.models import load_model

from constants import SAVE_PATH, PREDICTIONS_PATH, input_length, INSTRUMENTS_COUNT, NOTE_ON
from data_load import read_file
from data_output_stream import DataOutputStream


class Server:

    def __init__(self, host: str, port: int):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.s.bind((host, port))
            print("Created socket")
        except socket.error as err:
            print('Bind failed. Error Code : '.format(err))
        self.s.listen(10)
        print("Socket Listening")
        self.conn, self.addr = self.s.accept()

    def run(self):
        while True:
            self.conn.send(bytes("Message" + "\r\n", 'UTF-8'))
            print("Message sent")
            data = self.conn.recv(1024)
            file_path = data.decode(encoding='UTF-8')
            (division_type, resolution, x, _) = read_file(file_path)
            model = load_model(SAVE_PATH)
            y = model.predict(x)
            out_file_path = PREDICTIONS_PATH + "/out.dat"
            with open(out_file_path, 'wr') as out:
                write_to_file(out, x, y, division_type, resolution)
                self.conn.send(bytes(out_file_path))


def write_to_file(out, x, y, division_type, resolution):
    dos = DataOutputStream(out)
    dos.write_float(division_type)
    dos.write_int(resolution)
    dos.write_int(input_length)
    last_states = x[input_length - 1]
    count = 0
    for i in range(INSTRUMENTS_COUNT):
        if last_states[i] != y[i]:
            count += 1
    dos.write_byte(count)
    for i in range(INSTRUMENTS_COUNT):
        if last_states[i] != y[i]:
            dos.write_byte(i)
            dos.write_boolean(y[i] == NOTE_ON)
    out.flush()
    out.close()
