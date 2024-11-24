import numpy as np
import polars as pl

class TimeSeriesKFold:
    def __init__(self,
                 n_splits,
                 datetime_name,
                 test_datetime_size,
                 datetime_gap=0):
        self.n_splits = n_splits
        self.datetime_name = datetime_name
        self.test_datetime_size = test_datetime_size
        self.datetime_gap = datetime_gap

    def split(self, df):
        unique_datetimes = np.sort(np.unique(df[self.datetime_name]))
        
        train_datetime_size = unique_datetimes.shape[0] - self.test_datetime_size * self.n_splits - self.datetime_gap
        assert(train_datetime_size > 0)
        print(f"train_datetime_size: {train_datetime_size}, test_datetime_size: {self.test_datetime_size}")
        
        unique_datetimes = np.append(unique_datetimes, np.inf)
        for i in range(self.n_splits):
            yield df.with_row_index().filter(pl.col(self.datetime_name) >= unique_datetimes[i * self.test_datetime_size]).filter(pl.col(self.datetime_name) < unique_datetimes[i * self.test_datetime_size + train_datetime_size])["index"].to_numpy(), \
                  df.with_row_index().filter(pl.col(self.datetime_name) >= unique_datetimes[i * self.test_datetime_size + train_datetime_size + self.datetime_gap]).filter(pl.col(self.datetime_name) < unique_datetimes[i * self.test_datetime_size + train_datetime_size + self.datetime_gap + self.test_datetime_size])["index"].to_numpy()