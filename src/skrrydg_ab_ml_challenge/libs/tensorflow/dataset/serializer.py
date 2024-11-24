import pathlib
import pandas as pd
import polars as pl
import tensorflow as tf
import shutil
import hashlib
import gc

from .meta_data import DatasetMetaData
from multiprocessing import Pool
from .tf_record_writer import TFRecordWriter

class SliceGenerator:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.slice_size = int(1e8 / len(self.dataframe.columns))

    def __len__(self):
        return self.slice_size

    def __iter__(self):
        for index, slice in enumerate(self.dataframe.iter_slices(self.slice_size)):
            yield index, slice.columns, slice.to_numpy()

class DatasetSerializer:
    def __init__(self, directory: str):
        self.directory = pathlib.Path(directory)
        self.meta_data = None
        self.serialize_directory = None

    def _get_dataframe_hash(self, table):
        hash_table = table.select(~pl.selectors.by_dtype(pl.Null))
        if len(hash_table.columns) == 0:
            return hashlib.new('sha256').hexdigest()
        
        dataframe_hash = hashlib.sha256(hash_table.hash_rows().to_numpy())
        dataframe_hash.update(str(hash_table.dtypes).encode('utf-8'))
        return dataframe_hash.hexdigest()
        
    def serialize_chunk(self, args):
        index, columns, chunk = args
        
        path = str(self.serialize_directory / "{:03}.tfrecords".format(index))

        tf_writer = TFRecordWriter(path)
        
        record_bytes = tf.train.Example(features=tf.train.Features(feature={
            column: tf.train.Feature(
                float_list=tf.train.FloatList(value=chunk[:, index]))
            for index, column in enumerate(columns)
        })).SerializeToString()

        tf_writer.write(record_bytes)
        tf_writer.flush()

    def serialize(self, dataframe, n_cpu=10):
        dataframe_hash = self._get_dataframe_hash(dataframe)
        self.serialize_directory = self.directory / str(dataframe_hash)

        self.meta_data = DatasetMetaData(self.serialize_directory / "meta_data.json")
        if (self.meta_data.exists()):
            print("Data is consistent, do nothing", flush=True)
            return self.serialize_directory

        if self.serialize_directory.exists():
            shutil.rmtree(str(self.serialize_directory))
        self.serialize_directory.mkdir(parents=True, exist_ok=True)

        with Pool(n_cpu) as p:
            result = list(p.imap(self.serialize_chunk, SliceGenerator(dataframe)))

        self.meta_data.write({
            "count_records": dataframe.shape[0],
            "columns": dataframe.columns,
            "count_files": len(result),
            "hash": dataframe_hash
        })

        return self.serialize_directory
