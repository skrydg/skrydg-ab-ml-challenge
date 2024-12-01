import numpy as np

import sklearn.model_selection as ms

class KFold:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.splitter = ms.KFold(n_splits)
        
    def split(self, df):
        for i in range(5):
            yield list(range(0, int(i / 5 * df.shape[0]))) + list(range(int((i + 1) / 5 * df.shape[0]), df.shape[0])), \
                list(range(int(i / 5 * df.shape[0]), int((i + 1) / 5 * df.shape[0])))
       
