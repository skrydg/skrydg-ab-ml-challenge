import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
    
    def predict(self, dataframe, **kwargs):
        y_preds = [estimator.predict(dataframe, **kwargs) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_chunked(self, dataframe, **kwargs):
        y_preds = [estimator.predict_chunked(dataframe, **kwargs) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)


class WeightVotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, w = None):
        super().__init__()
        self.estimators = np.array(estimators)
        if w is None:
            self.w = np.ones(self.estimators.shape)
        else:
            self.w = w
        self.w = self.w / np.sum(self.w)

    def predict(self, dataframe, **kwargs):
        y_preds = np.array([estimator.predict(dataframe, **kwargs) for estimator in self.estimators])
        return np.dot(self.w, y_preds)

    def predict_chunked(self, dataframe, **kwargs):
        y_preds = np.array([estimator.predict_chunked(dataframe, **kwargs) for estimator in self.estimators])
        return np.dot(self.w, y_preds)