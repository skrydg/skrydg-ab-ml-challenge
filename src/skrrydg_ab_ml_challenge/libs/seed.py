import polars as pl
import numpy as np
import tensorflow as tf
import random

def seed_everything(seed=42):
    random.seed(seed)
    pl.set_random_seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)