import tensorflow as tf
import numpy as np
import gc
import shutil
import time

from skrrydg_ab_ml_challenge.libs.env import Env
from skrrydg_ab_ml_challenge.libs.model import VotingModel

from skrrydg_ab_ml_challenge.libs.tensorflow.backtest_metric import BacktestMetric
from skrrydg_ab_ml_challenge.libs.tensorflow.dataset import DatasetSerializer, DatasetDeserializer
from skrrydg_ab_ml_challenge.libs.tensorflow.backtest_loss import backtest_loss
from skrrydg_ab_ml_challenge.libs.tensorflow.pretrained_model import PreTrainedDNNModel

class FFKFoldModel:
    def __init__(self, env: Env, features):
        self.env = env
        self.features = features
        self.target_columns = ["coin2/bid", "coin2/ask", "delayed_bid", "delayed_ask", "reg_bid", "reg_ask"]

        self.all_features = list(sorted(list(set(self.features) | set(self.target_columns))))
        self.count_rows_in_memory = int(1e8 // (len(self.features)))
        self.batch_size = int(1e6 // (len(self.features)))
        
        self.model = None
        self.train_data = None

    def __pack_train(self, records):
        target_columns = [
            tf.cast(records[column], tf.float32) 
            for column in self.target_columns
        ]
        target_columns = tf.stack(target_columns, axis=-1)        

        features = [
            tf.cast(records[feature], tf.float32) 
            for feature in self.features 
        ]
        features = tf.stack(features, axis=-1)

        return features, target_columns
    
    def __pack_test(self, records):
        features = [
            tf.cast(records[feature], tf.float32) 
            for feature in self.features 
        ]
        features = tf.stack(features, axis=-1)

        return features
    
    def __build_model(self, dataframe):
        mean = dataframe[self.features].mean().to_numpy()[0]
        variance = dataframe[self.features].var().to_numpy()[0]
        
        X = tf.keras.layers.Input(shape=(len(self.features), ))

        normalized_X = tf.keras.layers.Normalization(mean=mean, variance=variance)(X)
        output_X = normalized_X
        output_X = tf.keras.layers.GaussianDropout(0.1, seed=42)(normalized_X)
        output_X = tf.keras.layers.Dense(
            units=256,
            activation='linear'
        )(output_X)

        #output_X = tf.keras.layers.BatchNormalization()(output_X)
        output_X = tf.keras.layers.ReLU(negative_slope=0.1)(output_X)

        output_X = tf.keras.layers.Concatenate(axis=1)([output_X, normalized_X])
        output_X = tf.keras.layers.GaussianDropout(0.1, seed=42)(output_X)
        output_X = tf.keras.layers.Dense(
            units=128,
            activation='linear'
        )(output_X)
        #output_X = tf.keras.layers.BatchNormalization()(output_X)
        output_X = tf.keras.layers.ReLU(negative_slope=0.1)(output_X)
        output_X = tf.keras.layers.Concatenate(axis=1)([output_X, normalized_X])
        output_X = tf.keras.layers.GaussianDropout(0.1, seed=42)(output_X)
        output_X = tf.keras.layers.Dense(
            units=64,
            activation='linear'
        )(output_X)
        # output_X = tf.keras.layers.BatchNormalization()(output_X)
        output_X = tf.keras.layers.ReLU(negative_slope=0.1)(output_X)
        output_X = tf.keras.layers.Concatenate(axis=1)([output_X, normalized_X])
        output_X = tf.keras.layers.GaussianDropout(0.1, seed=42)(output_X)
        output_X = tf.keras.layers.Dense(units=1)(output_X)
        
        model = tf.keras.Model(X, output_X)
        model.build(input_shape=(len(self.features), ))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=backtest_loss,
            metrics=[BacktestMetric()]
        )
        model.summary()
        return model
        
    def train(self, dataframe, kfold):
        self.train_data = {
            "backtest_metric": [],
            "loss": [],
            "oof_predicted": [],
            "oof_indexes": [],
            "history": []
        }
        
        fitted_models = []
        for index, (train_idx, test_idx) in enumerate(kfold.split(dataframe)):
            print("Start data serialization", flush=True)
            start = time.time()

            train_dataframe = dataframe[train_idx][self.all_features].sample(fraction=1, shuffle=True, seed=42)
            test_dataframe = dataframe[test_idx][self.all_features]

            train_dataframe_shape = train_dataframe.shape
            test_dataframe_shape = test_dataframe.shape
            
            train_serializer = DatasetSerializer(self.env.output_directory / f"train_dataset_{index}")
            train_serialized_directory = train_serializer.serialize(train_dataframe[self.all_features])
            
            train_deserializer = DatasetDeserializer(train_serialized_directory)
            train_dataset = train_deserializer.deserialize()
            train_dataset = train_dataset.map(self.__pack_train).unbatch().shuffle(buffer_size=self.count_rows_in_memory).batch(self.batch_size)

            test_serializer = DatasetSerializer(self.env.output_directory / f"test_dataset_{index}")
            test_serialized_directory = test_serializer.serialize(test_dataframe[self.all_features])
            
            test_deserializer = DatasetDeserializer(test_serialized_directory)
            test_dataset = test_deserializer.deserialize()
            test_dataset = test_dataset.map(self.__pack_train).rebatch(self.batch_size)
            
            finish = time.time()
            print(f"Finish data serialization, time={finish - start}", flush=True)
                        
            model = self.__build_model(train_dataframe)

            del train_dataframe
            del test_dataframe
            gc.collect()
            reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=3 * 1e-5)
            earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            history = model.fit(
                train_dataset.cache().repeat(),
                validation_data=test_dataset.cache().repeat(),
                steps_per_epoch=train_dataframe_shape[0] // self.batch_size + 1,
                validation_steps=test_dataframe_shape[0] // self.batch_size + 1,
                callbacks=[reduce_lr_callback, earlystop_callback],
                verbose=1,
                epochs=50)

            model = PreTrainedDNNModel(model)
            fitted_models.append(model)
            gc.collect()
            finish = time.time()
            print(f"Fit time: {finish - start}, iteration={index}")

            test_predicted = model.predict(test_dataset)
            self.train_data["oof_indexes"].extend(test_idx)
            self.train_data["oof_predicted"].extend(test_predicted[:, 0])

            metric = BacktestMetric()
            metric.update_state(dataframe[self.target_columns][test_idx].to_numpy(), test_predicted)
            self.train_data["backtest_metric"].append(metric.result().numpy())
            self.train_data["loss"].append(backtest_loss(dataframe[self.target_columns][test_idx].to_numpy(), test_predicted).numpy())
            self.train_data["history"].append(history)

        self.model = VotingModel(fitted_models)
        
    def predict(self, dataframe):
        serializer = DatasetSerializer(self.env.output_directory / "predict_dataset")
        serialized_directory = serializer.serialize(dataframe[self.features])
        
        deserializer = DatasetDeserializer(serialized_directory)
        dataset = deserializer.deserialize()
        dataset = dataset.map(self.__pack_test).unbatch()
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        predicted = self.model.predict(dataset)
        
        shutil.rmtree(str(self.env.output_directory / "predict_dataset"))
        return predicted