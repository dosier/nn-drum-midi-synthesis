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
            self.conn.send(bytes("ALIVE!" + "\r\n", 'UTF-8'))
            print("Message sent")
            data: bytes = self.conn.recv(9*16*8)
            dis = DataInputStream(BytesIO(data))
            X = []
            for time_step in range(16):
                x = []
                for instrument_idx in range(9):
                    x.append(int(dis.read_byte()))
                X.append(x)
            X = numpy.array([X])
            y = numpy.around(self.model.predict(X)[0], 0)
            print(y)
            out = bytearray()
            for i in range(16):
                for j in range(9):
                    if y[i][j] == 1.:
                        out.append(1)
                    else:
                        out.append(0)
            self.conn.send(out)


Server("localhost", 6969).run()
