import numpy as np
import polars as pl


class AddTargetRelatedFeatures:
    def __init__(self, delay=3000, reg_window=3 * int(1e6)):
       self.delay = delay
       self.reg_window = reg_window

    def process_train_dataset(self, dataset_generator):
        for dataset in dataset_generator:
          yield self.process(dataset)
        
    def process_test_dataset(self, dataset_generator):
        for dataset in dataset_generator:
          yield self.process(dataset)
    
    def process(self, dataset):
        md = dataset.get_table("md")
        dataset.set_table("md", md)
        return dataset
