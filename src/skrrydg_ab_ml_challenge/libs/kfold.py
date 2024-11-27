import numpy as np

import sklearn.model_selection as ms

class KFold:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.splitter = ms.KFold(n_splits)
        
    def split(self, df):
       yield from self.splitter.split(df, None)
