import pathlib
import tensorflow as tf

class TFRotatedRecordWriter:
    def __init__(self, directory: pathlib.Path):
        self.directory = directory
        self.writer_batch_size = 2 ** 30
        self.current_batch_size = self.writer_batch_size
        self.current_file_index = 0
        self.writer = None

    def write(self, record):
        if self.current_batch_size >= self.writer_batch_size:
            self.__create_new_writer()

        assert (self.current_batch_size < self.writer_batch_size)
        assert (self.writer is not None)

        self.writer.write(record)
        self.current_batch_size += len(record)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def get_count_files(self):
        return self.current_file_index

    def __create_new_writer(self):
        self.flush()

        self.writer = tf.io.TFRecordWriter(
            str(self.directory / "{:02}.tfrecords".format(self.current_file_index)))
        self.current_batch_size = 0
        self.current_file_index += 1