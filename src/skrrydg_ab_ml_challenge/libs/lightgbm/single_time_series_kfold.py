import numpy as np
import polars as pl

class SingleTimeSeriesKFold:
    def __init__(self, test_size, datetime_id="datetime_id"):
        self.test_size = test_size
        self.datetime_id = datetime_id
        
    def split(self, df):
        df = df.sort(self.datetime_id)
        unque_datetime_id = df[self.datetime_id].unique().to_numpy()
        
        train_unque_datetime_id_size = int(unque_datetime_id.shape[0] * (1 - self.test_size))

        unque_datetime_id_threashold = unque_datetime_id[train_unque_datetime_id_size]

        train_index = df.with_row_index().filter(pl.col(self.datetime_id) < unque_datetime_id_threashold)["index"].to_numpy()
        test_index = df.with_row_index().filter(pl.col(self.datetime_id) >= unque_datetime_id_threashold)["index"].to_numpy()
        
        yield train_index, test_index
