import socket
from io import BytesIO

import numpy

from nn.model_load import load_model_and_weights
from nn.predict.data_input_stream import DataInputStream


class Server:

    def __init__(self, host: str, port: int):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.model = load_model_and_weights()
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
            data: bytes = self.conn.recv(1024)
            dis = DataInputStream(BytesIO(data))
            X = []
            for time_step in range(16):
                x = []
                for instrument_idx in range(9):
                    x.append(dis.read_boolean())
                X.append(x)
            X = numpy.array(X)
            y = self.model.predict(X)
            print(numpy.around(y, 0))
            out_file_path = "temp/out.dat"
            with open(out_file_path, 'wr') as out:
                self.conn.send(bytes(out_file_path))
