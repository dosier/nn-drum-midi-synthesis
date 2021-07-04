import socket
from io import BytesIO

import numpy

from io.load_model import load_model_and_weights
from io.data_input_stream import DataInputStream
from io.midi_converter import MidiConverter

MODEL_TYPES = ["MM_4", "MM_9", "MO_4", "MO_9"]
MODEL_TYPE_INSTRUMENT_COUNTS = [4, 9, 4, 9]

class Server:

    def __init__(self, host: str, port: int):
        self.midi_converter = MidiConverter('output/')
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.s.bind((host, port))
            print("Created socket")
        except socket.error as err:
            print('Bind failed. Error Code : '.format(err))
        self.s.listen(10)
        print("Socket Listening")
        self.conn, self.addr = self.s.accept()
        self.model = None
        self.last_type = -1

    def run(self):
        while True:
            self.conn.send(bytes("ALIVE!" + "\r\n", 'UTF-8'))
            print("Message sent")

            header = DataInputStream(BytesIO(self.conn.recv(1 + 1 + 1)))
            type_ordinal = header.read_unsigned_byte()
            bpm = header.read_unsigned_byte()
            repeat = header.read_unsigned_byte()
            instruments_count = MODEL_TYPE_INSTRUMENT_COUNTS[type_ordinal]
            x_data = self.conn.recv(instruments_count * 16 * 8)
            y, midi_path = self.make_prediction(type_ordinal, bpm, repeat, instruments_count, x_data)
            self.conn.send(y)
            self.conn.recv(1)
            self.conn.send(bytes(midi_path + "\r\n", 'UTF-8'))

    def make_prediction(self, type_ordinal: int, bpm: int, repeat: int, instruments_count: int, data: bytes) -> (bytearray, str):
        dis = DataInputStream(BytesIO(data))
        X = []
        if self.last_type == -1 or type_ordinal != self.last_type:
            self.model = load_model_and_weights(model_type=MODEL_TYPES[type_ordinal], path="models/")
        for time_step in range(16):
            x = []
            for instrument_idx in range(instruments_count):
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
            for j in range(instruments_count):
                if X[0][i][j] == 1.:
                    out.append(1)
                else:
                    out.append(0)
        return out, midi_path


Server("localhost", 6969).run()
