import numpy as np

import sklearn

class KFold:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.splitter = sklearn.model_selection.KFold(n_splits, random_state=42)
        
    def split(self, df):
       yield from self.splitter.split(df, None)
