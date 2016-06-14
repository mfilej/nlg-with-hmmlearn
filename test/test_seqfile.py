import pytest
import os, sys

from seqfile import Seqfile

class TestSeqfile:
    def file_path(self):
        return os.path.join(os.getcwd(), 'test', 'fixtures', 'example.seq')

    def test_read(self):
        with open(self.file_path()) as f:
            sf = Seqfile(f)
            assert sf.t == 15
            assert list(sf.seq) == [1, 2, 3, 4, 5, 1, 6, 2, 3, 7, 1, 2, 5, 8, 1]
