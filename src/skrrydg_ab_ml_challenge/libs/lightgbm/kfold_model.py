import numpy as np
import time

import polars as pl
import lightgbm as lgb
import gc
import sklearn

from kaggle_jane_street_real_time_market_data_forecasting.libs.env import Env
from kaggle_jane_street_real_time_market_data_forecasting.libs.lightgbm.serializer import LightGbmDatasetSerializer
from kaggle_jane_street_real_time_market_data_forecasting.libs.lightgbm.pre_trained_model import PreTrainedLightGbmModel
from kaggle_jane_street_real_time_market_data_forecasting.libs.model import VotingModel
from kaggle_jane_street_real_time_market_data_forecasting.libs.r_sqaure import r_square

class KFoldLightGbmModel:
    def __init__(self, env: Env, features, model_params, metrics=[], target="target"):
        self.env = env
        self.features = features
        self.target = target
        self.features_with_target = self.features + [self.target]
        self.model_params = model_params
        self.metrics = metrics

        self.model = None
        self.train_data = None
        self.w = None

    def predict_fast(self, dataframe):
        physical_dataframe = dataframe.with_columns(*[
            pl.col(column).to_physical()
            for column in dataframe.columns
            if (dataframe[column].dtype == pl.Enum) or (dataframe[column].dtype == pl.Categorical)
        ])
        return self.model.predict(physical_dataframe[self.features].to_numpy())

    def predict(self, dataframe, model = None, chunk_size = None, **kwargs):
        model = model or self.model
        chunk_size = chunk_size or dataframe.shape[0]
        assert(model is not None)

        Y_predicted = None
        
        for start_position in range(0, dataframe.shape[0], chunk_size):
            X = dataframe[self.features][start_position:start_position + chunk_size]

            physical_X = X.with_columns(*[
                pl.col(column).to_physical()
                for column in X.columns
                if (X[column].dtype == pl.Enum) or (X[column].dtype == pl.Categorical)
            ])

            current_Y_predicted = model.predict(physical_X, **kwargs)

            if Y_predicted is None:
                Y_predicted = current_Y_predicted
            else:
                Y_predicted = np.concatenate([Y_predicted, current_Y_predicted])
            del X 
            gc.collect()

        return Y_predicted

    
    def train(self, dataframe, k_fold):
        print("Start train for KFoldLightGbmModel")

        self.w = dataframe["weight"].to_numpy()
        self.train_data = {
            "r2_scores": [],
            "oof_predicted": [],
            "oof_indexes": []
        }
            
        fitted_models = []
        for iteration, (idx_train, idx_test) in enumerate(k_fold.split(dataframe)):
            print("Start data serialization")
            start = time.time()

            train_dataset_serializer = LightGbmDatasetSerializer(
                self.env.output_directory / "train_datasert", 
                {"max_bin": self.model_params["max_bin"]},
                target = self.target
            )
            test_dataset_serializer = LightGbmDatasetSerializer(
                self.env.output_directory / "test_datasert", 
                {"max_bin": self.model_params["max_bin"]},
                target = self.target
            )

            train_dataset_serializer.serialize(dataframe[self.features_with_target][idx_train])
            train_dataset = train_dataset_serializer.deserialize()
            train_dataset.set_position(idx_train)
            
            test_dataset_serializer.serialize(dataframe[self.features_with_target][idx_test])
            test_dataset = test_dataset_serializer.deserialize()
            test_dataset.set_position(idx_test)

            finish = time.time()
            print(f"Finish data serialization, time={finish - start}")

            start = time.time()
            model = lgb.train(
              self.model_params,
              train_dataset,
              valid_sets=[test_dataset],
              callbacks=[lgb.log_evaluation(10), lgb.early_stopping(100, first_metric_only=True)],
              feval=self.feval_metrics
            )
            model = PreTrainedLightGbmModel(model)
            finish = time.time()
            print(f"Fit time: {finish - start}, iteration={iteration}")

            fitted_models.append(model)

            test_predicted = self.predict(dataframe[idx_test], model=model, chunk_size=1000000)
            
            self.train_data["oof_indexes"].extend(idx_test)
            self.train_data["oof_predicted"].extend(test_predicted)
            self.train_data["r2_scores"].append(r_square(
                dataframe[self.target][idx_test].to_numpy(),
                test_predicted,
                weight=dataframe["weight"][idx_test].to_numpy()
            ))

            train_dataset_serializer.clear()
            test_dataset_serializer.clear()
            del train_dataset_serializer
            del test_dataset_serializer
            del train_dataset
            del test_dataset
            gc.collect()

        self.model = VotingModel(fitted_models)
        
        print("Finish train for KFoldLightGbmModel")
        return self.train_data

    @staticmethod
    def r_square(y_pred: np.ndarray, data: lgb.Dataset, weight):
        return 'r_square_metric', r_square(data.get_label(), y_pred, weight), True

    def feval_metrics(self, preds: np.ndarray, data: lgb.Dataset):
        ret = []
        if "r_square" in self.metrics:
            ret.append(KFoldLightGbmModel.r_square(preds, data, self.w[data.get_position()]))
        return ret
    
    def get_feature_importance(self, type = "split"):
        fi = np.zeros(self.model.estimators[0].model.feature_importance(type).shape)
        fn = self.model.estimators[0].model.feature_name()
        for estimator in self.model.estimators:
            fi = fi + estimator.model.feature_importance(type)
        sorted_by_importance_features = list(reversed(sorted(list(zip(fi, fn)))))
        sorted_by_importance_features = [(float(fi), fn) for fi, fn in sorted_by_importance_features]
        return sorted_by_importance_features