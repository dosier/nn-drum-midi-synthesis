import socket

from keras.models import load_model

from constants import SAVE_PATH
from data_load import read_file

HOST = "localhost"
PORT = 9999
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

try:
    s.bind((HOST, PORT))
except socket.error as err:
    print('Bind failed. Error Code : '.format(err))
s.listen(10)
print("Socket Listening")
conn, addr = s.accept()
while True:
    conn.send(bytes("Message" + "\r\n", 'UTF-8'))
    print("Message sent")
    data = conn.recv(1024)
    file_path = data.decode(encoding='UTF-8')
    x, _ = read_file(file_path)
    model = load_model(SAVE_PATH)
    y = model.predict(x)
    conn.send(bytes("BOO" + "\r\n", 'UTF-8'))
