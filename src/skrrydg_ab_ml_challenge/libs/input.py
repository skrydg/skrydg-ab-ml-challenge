import polars as pl

from skrrydg_ab_ml_challenge.libs.env import Env
from skrrydg_ab_ml_challenge.libs.dataset import Dataset

class Input:
    def __init__(self, env: Env):
        self.env = env
        
    def get_train_dataset(self, n_rows=None, filters = []):
        df = pl.scan_parquet(f"{self.env.input_directory}/ab-ml-challenge/train_data", n_rows=n_rows)
        for f in filters:
            df = df.filter(f)
        return Dataset({"md": df.collect()})

    def get_test_dataset(self, n_rows=None, filters = []):
        df = pl.scan_parquet(f"{self.env.input_directory}/ab-ml-challenge/test_data", n_rows=n_rows)
        for f in filters:
            df = df.filter(f)
        return Dataset({"md": df.collect()})