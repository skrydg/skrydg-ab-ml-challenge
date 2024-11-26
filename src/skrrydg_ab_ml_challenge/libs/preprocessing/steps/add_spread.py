import numpy as np
import polars as pl


class AddSpread:
    def process_train_dataset(self, dataset_generator):
        for dataset in dataset_generator:
          yield self.process(dataset)
        
    def process_test_dataset(self, dataset_generator):
        for dataset in dataset_generator:
          yield self.process(dataset)
    
    def process(self, dataset):
        md = dataset.get_table("md")
        md = md.with_columns((pl.col("coin2/ask") - pl.col("coin2/bid")).alias("spread"))

        dataset.set_table("md", md)
        return dataset
