import socket
from io import BytesIO

import numpy

from nn.model_load import load_model_and_weights
from nn.py_util.data_input_stream import DataInputStream
from nn.py_util.midi_converter import MidiConverter


class Server:

    def __init__(self, host: str, port: int, model=load_model_and_weights(path="models/server/")):
        self.midi_converter = MidiConverter('output/')
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.model = model
        self.instruments_count: int = self.model.lstm_layers[-1].output_shape[-1]
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
            self.conn.send(bytes("ALIVE!" + "\r\n", 'UTF-8'))
            print("Message sent")
            x = self.conn.recv(1 + 1 + (self.instruments_count * 16 * 8))
            y, midi_path = self.make_prediction(x)
            self.conn.send(y)
            self.conn.recv(1)
            self.conn.send(bytes(midi_path + "\r\n", 'UTF-8'))

    def make_prediction(self, data: bytes) -> (bytearray, str):
        dis = DataInputStream(BytesIO(data))
        bpm = dis.read_unsigned_byte()
        repeat = dis.read_unsigned_byte()
        X = []
        for time_step in range(16):
            x = []
            for instrument_idx in range(self.instruments_count):
                x.append(int(dis.read_byte()))
            X.append(numpy.array(x))
        timed_notes = []
        for time_step in X:
            timed_notes.append(time_step)
        X = numpy.array([X])
        for i in range(repeat):
            X = numpy.around(self.model.predict(X), 0)
            for time_step in X[0]:
                timed_notes.append(time_step)

        midi_path = self.midi_converter.make_midi(timed_notes=numpy.array(timed_notes), bpm=bpm, filename="sexy.mid")
        print(X)
        out = bytearray()
        for i in range(16):
            for j in range(self.instruments_count):
                if X[0][i][j] == 1.:
                    out.append(1)
                else:
                    out.append(0)
        return out, midi_path


Server("localhost", 6969).run()
