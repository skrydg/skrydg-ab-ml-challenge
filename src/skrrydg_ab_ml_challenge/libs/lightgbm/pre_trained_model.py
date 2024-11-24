class PreTrainedLightGbmModel:
    def __init__(self, model):
        self.model = model

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
