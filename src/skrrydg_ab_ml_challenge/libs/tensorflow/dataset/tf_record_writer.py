import pathlib
import tensorflow as tf

class TFRecordWriter:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.writer = tf.io.TFRecordWriter(self.path)

    def write(self, record):
        assert (self.writer is not None)

        self.writer.write(record)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()
