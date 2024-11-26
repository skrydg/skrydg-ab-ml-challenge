import numpy as np
import polars as pl


class DeleteColumns:
    def __init__(self, columns):
       self.columns = columns

    def process_train_dataset(self, dataset_generator):
        for dataset in dataset_generator:
          yield self.process(dataset)
        
    def process_test_dataset(self, dataset_generator):
        for dataset in dataset_generator:
          yield dataset
    
    def process(self, dataset):
        md = dataset.get_table("md")
        md = md.drop(self.columns)
        dataset.set_table("md", md)
        return dataset
