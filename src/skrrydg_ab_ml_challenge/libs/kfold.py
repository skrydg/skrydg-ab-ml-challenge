import numpy as np

import sklearn

class KFold:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.splitter = sklearn.model_selection.KFold(n_splits)
        
    def split(self, df):
       yield from self.splitter.split(df, None)
