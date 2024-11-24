import lightgbm as lgb
import gc
import polars as pl

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
class LightGbmDatasetCreator:
    def __init__(self, dataset_params, chunk_size = 100, target="target"):
        self.dataset_params = dataset_params
        self.chunk_size = chunk_size
        self.target = target

    def create(self, dataframe):
        columns = dataframe.columns
        columns.remove(self.target)

        X = dataframe[columns]
        Y = dataframe[self.target]
        
        dataset = None
        for chunk_index, columns_chunk in enumerate(chunker(columns, self.chunk_size)):
            chunk_categorical_features = [feature for feature in columns_chunk if X[feature].dtype == pl.Enum]
            chunk_float32_features = [feature for feature in columns_chunk if X[feature].dtype == pl.Float32]

            physical_X = X[columns_chunk].with_columns(*[
                pl.col(column).to_physical()
                for column in chunk_categorical_features
            ])
    
            physical_X = physical_X.with_columns(*[
                pl.col(column).cast(pl.Float64)
                for column in chunk_float32_features
            ])

            data = lgb.Dataset(
                physical_X.to_numpy(),
                Y.to_numpy(),
                params=self.dataset_params,
                categorical_feature=chunk_categorical_features,
                feature_name=physical_X.columns,
                free_raw_data=False
            )

            data.construct()
            if dataset is None:
                dataset = data
            else:
                dataset.add_features_from(data)

            del physical_X
            gc.collect()

        return dataset