# import numpy as np
# import tensorflow as tf

# class TimeSeriesKFold:
#     def __init__(self, n_splits):
#         self.n_splits = n_splits
        
#     def split(self, dataset, dataset_size):
#         chunk_size = dataset_size // (self.n_splits + 1)
#         for i in range(self.n_splits):
#             train_dataset = dataset.take(chunk_size * (i + 1))
#             test_dataset = dataset.skip(chunk_size * (i + 1)).take(chunk_size)
#             yield train_dataset, test_dataset