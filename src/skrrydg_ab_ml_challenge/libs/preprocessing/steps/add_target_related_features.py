import numpy as np
import polars as pl


class AddTargetRelatedFeatures:
    def __init__(self, delay=20000, reg_window=3 * int(1e6)):
       self.delay = delay
       self.reg_window = reg_window

    def process_train_dataset(self, dataset_generator):
        for dataset in dataset_generator:
          yield self.process(dataset)
        
    def process_test_dataset(self, dataset_generator):
        for dataset in dataset_generator:
          yield dataset
    
    def process(self, dataset):
        md = dataset.get_table("md")
        md = md.sort(pl.col("ts"))
        delayed_index = md["ts"].search_sorted(md["ts"] + pl.duration(microseconds=self.delay), side='right')
        reg_index = md["ts"].search_sorted(md["ts"] + pl.duration(microseconds=self.reg_window), side='right')

        md = md.with_columns(delayed_index.alias("delayed_index"))
        md = md.with_columns(reg_index.alias("reg_index"))

        md = md.with_columns(md.select(pl.when(pl.col("delayed_index") >= md.shape[0]).then(None).otherwise(pl.col("delayed_index")).alias("delayed_index")))
        md = md.with_columns(md.select(pl.when(pl.col("reg_index") >= md.shape[0]).then(None).otherwise(pl.col("reg_index")).alias("reg_index")))

        md = md.with_columns(md["coin2/bid"][md["delayed_index"]].alias("delayed_bid"))
        md = md.with_columns(md["coin2/ask"][md["delayed_index"]].alias("delayed_ask"))

        md = md.with_columns(md["coin2/bid"][md["reg_index"]].alias("reg_bid"))
        md = md.with_columns(md["coin2/ask"][md["reg_index"]].alias("reg_ask"))
        
        dataset.set_table("md", md)
        return dataset
