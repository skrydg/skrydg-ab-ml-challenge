import polars as pl

from kaggle_jane_street_real_time_market_data_forecasting.libs.env import Env
from kaggle_jane_street_real_time_market_data_forecasting.libs.dataset import Dataset

class Input:
    def __init__(self, env: Env):
        self.env = env
        
    def get_train_dataset(self, n_rows=None, filters = []):
        df = pl.scan_parquet(f"{self.env.input_directory}/ab-ml-challenge/data", n_rows=n_rows)
        for f in filters:
            df = df.filter(f)
        return Dataset({"md": df.collect()})