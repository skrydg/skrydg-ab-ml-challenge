import time
import gc
import copy
import hashlib
import polars as pl


class Preprocessor:
    def __init__(self, steps):
        self.steps = steps
    
    def process_train_dataset(self, train_dataset_generator):
        for name, step in self.steps.items():
            train_dataset_generator = self.collect_garbage(step.process_train_dataset(train_dataset_generator))
            gc.collect()
            print("Step: {}".format(name), flush=True)
        
        train_dataset = next(train_dataset_generator)
        return train_dataset
    
    def process_test_dataset(self, test_dataset_generator):
        for name, step in self.steps.items():
            test_dataset_generator = step.process_test_dataset(test_dataset_generator)

        test_dataset = next(test_dataset_generator)
        return test_dataset
            
    def collect_garbage(self, test_dataset_generator):
        gc.collect()
        yield from test_dataset_generator