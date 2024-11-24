import tensorflow as tf

class WeightedR2Mertric(tf.keras.Metric):
    def __init__(self, name='weighted_r2', **kwargs):
        super().__init__(name=name, **kwargs)
        self.unexplained_error = self.add_variable(
            shape=(),
            initializer='zeros',
            name='true_positives',
            dtype=tf.float32
        )
        self.total_error = self.add_variable(
            shape=(),
            initializer='zeros',
            name='true_positives',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        target_shape = y_pred.shape[1] // 2

        y_pred = y_pred[:, 0]
        delayed_bid = y_pred[:, 1]
        delayed_ask = y_pred[:, 2]

        reg_bid = y_pred[:, 3]
        reg_ask = y_pred[:, 4]

        y_true = tf.reshape(y_true, shape = (-1, 1))
        y_pred = tf.reshape(y_pred, shape = (-1, 1))
        weights = tf.reshape(weights, shape = (-1, 1))

        self.unexplained_error.assign_add(tf.math.reduce_sum(weights * tf.square(y_true - y_pred)))
        self.total_error.assign_add(tf.math.reduce_sum(weights * tf.square(y_true)))

    def result(self):
        return 1 - self.unexplained_error / self.total_error
    
    def reset_state(self):
        self.total_error.assign(0)
        self.unexplained_error.assign(0)