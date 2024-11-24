import numpy as np

import sklearn

class GroupKFold:
    def __init__(self, n_splits, group_name="date_id"):
        self.n_splits = n_splits
        self.group_name = group_name
        self.splitter = sklearn.model_selection.GroupKFold(n_splits)
        
    def split(self, df):
       yield from self.splitter.split(df, None, df[self.group_name])
