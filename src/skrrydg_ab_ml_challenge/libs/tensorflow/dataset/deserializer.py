import pathlib
import tensorflow as tf

from .meta_data import DatasetMetaData


class DatasetDeserializer:
    def __init__(self, directory: str):
        self.directory = pathlib.Path(directory)
        self.meta_data = DatasetMetaData(self.directory / "meta_data.json")
        assert (self.meta_data.exists())

    def deserialize(self):
        assert (self.directory.exists())

        meta_data = self.meta_data.read()

        def decode_fn(record_bytes):
            return tf.io.parse_single_example(
                record_bytes,
                {column: tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)
                    for column in meta_data["columns"]}
            )

        return tf.data.TFRecordDataset(
            self.__get_files(meta_data["count_files"]),
            buffer_size=1000000
        ).map(decode_fn)

    def __get_files(self, count_files):
        return [str(self.directory / "{:03}.tfrecords".format(i)) for i in range(count_files)]
