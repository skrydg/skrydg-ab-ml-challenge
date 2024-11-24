import tensorflow as tf
import numpy as np
import gc
import shutil
import time

from kaggle_jane_street_real_time_market_data_forecasting.libs.env import Env
from kaggle_jane_street_real_time_market_data_forecasting.libs.model import VotingModel

from kaggle_jane_street_real_time_market_data_forecasting.libs.tensorflow.r2_metric import WeightedR2Mertric
from kaggle_jane_street_real_time_market_data_forecasting.libs.tensorflow.dataset import DatasetSerializer, DatasetDeserializer
from kaggle_jane_street_real_time_market_data_forecasting.libs.tensorflow.r2_loss import weighted_r2_loss
from kaggle_jane_street_real_time_market_data_forecasting.libs.tensorflow.time_series_kfold import TimeSeriesKFold
from kaggle_jane_street_real_time_market_data_forecasting.libs.tensorflow.pretrained_model import PreTrainedDNNModel
from kaggle_jane_street_real_time_market_data_forecasting.libs.r_sqaure import r_square

@tf.keras.utils.register_keras_serializable()
class ClipLayer(tf.keras.Layer):
    def __init__(self, x_min, x_max, **kwargs):
        super().__init__(**kwargs)
        self.x_min = x_min
        self.x_max = x_max

    def call(self, x):
        return tf.clip_by_value(x, clip_value_min=self.x_min, clip_value_max=self.x_max)

class FFKFoldModel:
    def __init__(self, env: Env, features, targets, weights):
        self.env = env
        self.features = features
        self.weights = weights
        self.targets = targets
        
        self.count_rows_in_memory = int(1e8 // (len(self.features)))
        self.batch_size = int(1e6 // (len(self.features)))
        
        self.model = None
        self.train_data = None

    def __pack(self, records):
        targets = [
            tf.cast(records[target], tf.float32) 
            for target in self.targets 
        ]
        targets = tf.stack(targets, axis=-1)

        weights = [
            tf.cast(records[weight], tf.float32) 
            for weight in self.weights 
        ]
        weights = tf.stack(weights, axis=-1)
        

        features = [
            tf.cast(records[feature], tf.float32) 
            for feature in self.features 
        ]
        features = tf.stack(features, axis=-1)

        return (features, weights), targets
        
    def __build_model(self, dataframe):
        mean = dataframe[self.features].mean().to_numpy()[0]
        variance = dataframe[self.features].var().to_numpy()[0]
        
        X = tf.keras.layers.Input(shape=(len(self.features), ))
        weights = tf.keras.layers.Input(shape=(len(self.weights), ))

        normalized_X = tf.keras.layers.Normalization(mean=mean, variance=variance)(X)
        output_X = tf.keras.layers.GaussianDropout(0.35, seed=42)(normalized_X)
        output_X = tf.keras.layers.Dense(
            units=256,
            activation='linear'
        )(output_X)

        output_X = tf.keras.layers.BatchNormalization()(output_X)
        output_X = tf.keras.layers.ReLU()(output_X)

        output_X = tf.keras.layers.Concatenate(axis=1)([output_X, normalized_X])
        output_X = tf.keras.layers.GaussianDropout(0.35, seed=42)(output_X)
        output_X = tf.keras.layers.Dense(
            units=128,
            activation='linear'
        )(output_X)
        output_X = tf.keras.layers.BatchNormalization()(output_X)
        output_X = tf.keras.layers.ReLU()(output_X)
        output_X = tf.keras.layers.Concatenate(axis=1)([output_X, normalized_X])
        output_X = tf.keras.layers.GaussianDropout(0.35, seed=42)(output_X)
        output_X = tf.keras.layers.Dense(
            units=64,
            activation='linear'
        )(output_X)
        output_X = tf.keras.layers.BatchNormalization()(output_X)
        output_X = tf.keras.layers.ReLU()(output_X)
        output_X = tf.keras.layers.Concatenate(axis=1)([output_X, normalized_X])
        output_X = tf.keras.layers.GaussianDropout(0.35, seed=42)(output_X)
        output_X = tf.keras.layers.Dense(units=len(self.targets))(output_X)
        output_X = ClipLayer(-5, 5)(output_X)
        output_X = tf.keras.layers.Concatenate(axis=1)([output_X, weights])
        
        model = tf.keras.Model((X, weights), output_X)
        model.build(input_shape=(len(self.features), ))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=weighted_r2_loss,
            metrics=[WeightedR2Mertric()]
        )
        model.summary()
        return model
        
    def train(self, dataframe, kfold):
        dataframe = dataframe.fill_null(0)

        self.train_data = {
            "r2_scores": [],
            "oof_predicted": [],
            "oof_indexes": [],
            "history": []
        }
        
        fitted_models = []
        for index, (train_idx, test_idx) in enumerate(kfold.split(dataframe)):
            print("Start data serialization", flush=True)
            start = time.time()

            train_dataframe = dataframe[train_idx][self.features + self.targets + self.weights].sample(fraction=1, shuffle=True, seed=42)
            test_dataframe = dataframe[test_idx][self.features + self.targets + self.weights]

            train_dataframe_shape = train_dataframe.shape
            test_dataframe_shape = test_dataframe.shape
            
            train_serializer = DatasetSerializer(self.env.output_directory / f"train_dataset_{index}")
            train_serialized_directory = train_serializer.serialize(train_dataframe[self.features + self.targets + self.weights])
            
            train_deserializer = DatasetDeserializer(train_serialized_directory)
            train_dataset = train_deserializer.deserialize()
            train_dataset = train_dataset.map(self.__pack).unbatch().shuffle(buffer_size=self.count_rows_in_memory).batch(self.batch_size)

            test_serializer = DatasetSerializer(self.env.output_directory / f"test_dataset_{index}")
            test_serialized_directory = test_serializer.serialize(test_dataframe[self.features + self.targets + self.weights])
            
            test_deserializer = DatasetDeserializer(test_serialized_directory)
            test_dataset = test_deserializer.deserialize()
            test_dataset = test_dataset.map(self.__pack).rebatch(self.batch_size)
            
            finish = time.time()
            print(f"Finish data serialization, time={finish - start}", flush=True)
                        
            model = self.__build_model(train_dataframe)

            del train_dataframe
            del test_dataframe
            gc.collect()
            reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-5)
            earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(
                train_dataset.prefetch(tf.data.AUTOTUNE).repeat(),
                validation_data=test_dataset.prefetch(tf.data.AUTOTUNE).repeat(),
                steps_per_epoch=train_dataframe_shape[0] // self.batch_size + 1,
                validation_steps=test_dataframe_shape[0] // self.batch_size + 1,
                callbacks=[reduce_lr_callback, earlystop_callback],
                verbose=1,
                epochs=100)

            model = PreTrainedDNNModel(model)
            fitted_models.append(model)
            gc.collect()
            finish = time.time()
            print(f"Fit time: {finish - start}, iteration={index}")

            test_predicted = model.predict(test_dataset)[:, :len(self.targets)]
            self.train_data["oof_indexes"].extend(test_idx)
            self.train_data["oof_predicted"].extend(test_predicted)
            self.train_data["r2_scores"].append(r_square(
                dataframe[self.targets][test_idx].to_numpy().reshape((-1, 1)),
                test_predicted.reshape((-1, 1)),
                weight=dataframe[self.weights][test_idx].to_numpy().reshape((-1, 1))
            ))
            self.train_data["history"].append(history)

        self.model = VotingModel(fitted_models)

    def predict_fast(self, dataframe):
        y_predicted = tf.zeros(shape=dataframe.shape[0])
        for model in self.model.estimators:
            res = model.model([dataframe.fill_null(0)[self.features].to_numpy(), dataframe[self.weights].to_numpy()], training=False)
            y_predicted = y_predicted + res[:, :len(self.targets)]
        y_predicted = y_predicted / len(self.model.estimators)
        return y_predicted.numpy()
        
    def predict(self, dataframe):
        dataframe = dataframe.fill_null(0)
        serializer = DatasetSerializer(self.env.output_directory / "predict_dataset")
        serializer.serialize(dataframe[self.features + self.targets + self.weights].iter_slices(self.count_rows_in_memory), rows=dataframe.shape[0], force=True)
        
        deserializer = DatasetDeserializer(self.env.output_directory / "predict_dataset")
        dataset = deserializer.deserialize()
        dataset = dataset.map(self.__pack).unbatch()
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        predicted = self.model.predict(dataset)
        
        shutil.rmtree(str(self.env.output_directory / "predict_dataset"))
        return predicted