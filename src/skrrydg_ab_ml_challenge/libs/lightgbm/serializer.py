import lightgbm as lgb
import polars as pl
import h5py
import numpy as np
import shutil
import pathlib
import os
import gc

class HDFSequence(lgb.Sequence):
    def __init__(self, hdf_dataset):
        self.data = hdf_dataset
        self.batch_size = 32

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class LightGbmDatasetSerializer:
    def __init__(self, directory, dataset_params, target = "target"):
        self.directory = pathlib.Path(directory)
        self.dataset_params = dataset_params
        self.target = target
        self.files = []
        self.rows_batch_size = 1000000
        self.categorical_columns = []
        self.float32_columns = []
        self.columns = []
        self.clear()
        os.makedirs(self.directory, exist_ok=True)

    def serialize(self, dataframe):
        self.categorical_columns = [column for column in dataframe.columns if dataframe[column].dtype == pl.Enum]
        self.float32_columns = [column for column in dataframe.columns if dataframe[column].dtype == pl.Float32]

        for start in range(0, dataframe.shape[0], self.rows_batch_size):
            index = start // self.rows_batch_size
            current_df = dataframe[start:start + self.rows_batch_size]
            physical_current_df = current_df.with_columns(*[
                pl.col(column).to_physical()
                for column in self.categorical_columns
            ])
            physical_current_df = physical_current_df.with_columns(*[
                pl.col(column).cast(pl.Float64)
                for column in self.float32_columns
            ])

            physical_current_df = physical_current_df.drop(self.target)
            target = current_df[[self.target]]

            self.columns = physical_current_df.columns
        
            filename = self.directory / f"dataframe_{index}.h5"
            self.save_hdf({"Y": target, "X": physical_current_df[self.columns]}, filename)
            self.files.append(filename)

            del current_df
            del physical_current_df
            gc.collect()
        gc.collect()

    def deserialize(self):
        data = []
        ylist = []
        for f in self.files:
            f = h5py.File(f, "r")
            data.append(HDFSequence(f["X"]))
            ylist.append(f["Y"][:])
    
        y = np.concatenate(ylist)
        dataset = lgb.Dataset(
            data,
            label=y,
            params=self.dataset_params, 
            feature_name = self.columns,
            categorical_feature=self.categorical_columns,
            free_raw_data=True
        )
        gc.collect()
        return dataset

    def save_hdf(self, input_data, filename):
        with h5py.File(filename, "w") as f:
            for name, data in input_data.items():
                nrow, ncol = data.shape
                if ncol == 1:
                    chunk = (nrow,)
                    data = data.to_numpy().flatten()
                else:
                    data = data.to_numpy()
                    chunk = (32, ncol)
                f.create_dataset(name, data=data, chunks=chunk, compression="lzf")

    def clear(self):
        shutil.rmtree(self.directory, ignore_errors=True)