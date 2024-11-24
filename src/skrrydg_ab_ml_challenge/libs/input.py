import polars as pl

from kaggle_jane_street_real_time_market_data_forecasting.libs.env import Env
from kaggle_jane_street_real_time_market_data_forecasting.libs.dataset import Dataset

class Input:
    def __init__(self, env: Env):
        self.env = env
        self.features = [f"feature_{i:02}" for i in range(0, 79)] + ["weight"]
        self.target = "responder_6"
        self.all_targets = [f"responder_{i}" for i in range(9)]
        self.meta_features = ["time_id", "date_id", "symbol_id"]
        
    def get_train_dataset(self, n_rows=None, filters = [], columns=None):
        columns = columns or (self.features + self.all_targets + self.meta_features)

        df = pl.scan_parquet(f"{self.env.input_directory}/jane-street-real-time-market-data-forecasting/train.parquet", n_rows=n_rows)
        for f in filters:
            df = df.filter(f)
        df = df.select(columns)
        return Dataset({"md": df.collect()})
    
    def get_test_dataset(self,  n_rows=None, filters = [], columns=None):
        columns = columns or (self.features + self.meta_features)

        df = pl.scan_parquet(f"{self.env.input_directory}/jane-street-real-time-market-data-forecasting/test.parquet", n_rows=n_rows)
        for f in filters:
            df = df.filter(f)
        df = df.select(columns)
        return Dataset({"md": df.collect()})
    
    def get_lags(self):
        lags_parquet = pl.scan_parquet("/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet/date_id=0/part-0.parquet").collect()
        return lags_parquet