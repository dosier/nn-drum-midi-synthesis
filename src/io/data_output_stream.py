import struct


class DataOutputStream:

    def __init__(self, stream):
        self.stream = stream
        self.offset = 0

    def write_int(self, value):
        struct.pack_into('>i', self.stream, self.offset, value)
        self.offset += 4

    def write_float(self, value):
        struct.pack_into('>f', self.stream, self.offset, value)
        self.offset += 4

    def write_byte(self, value):
        struct.pack_into('b', self.stream, self.offset, value)
        self.offset += 1

    def write_boolean(self, value):
        struct.pack_into('?', self.stream, self.offset, value)
        self.offset += 1
