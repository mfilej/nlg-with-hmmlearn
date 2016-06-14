import re

class Seqfile():
    def __init__(self, io):
        self.t, self.seq = self.read(io)

    def read(self, io):
        line = io.readline()
        t = re.match('T= (\d+)\n', line).group(1)
        t = int(t)

        line = io.readline()
        seq = (int(x) for x in line.split(None, t))

        return t, seq
