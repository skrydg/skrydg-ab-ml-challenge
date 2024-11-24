import tensorflow as tf

class BacktestMetric(tf.keras.Metric):
    def __init__(self, name='weighted_r2', interest=6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.interest = interest
        self.metric = self.add_variable(
            shape=(),
            initializer='zeros',
            name='metric',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        bid = y_true[:, 0]
        ask = y_true[:, 1]
        delayed_bid = y_true[:, 2]
        delayed_ask = y_true[:, 3]

        reg_bid = y_true[:, 4]
        reg_ask = y_true[:, 5]

        y_pred = tf.reshape(y_pred, shape = (-1, 1))
        bid = tf.reshape(bid, shape = (-1, 1))
        ask = tf.reshape(ask, shape = (-1, 1))
        delayed_bid = tf.reshape(delayed_bid, shape = (-1, 1))
        delayed_ask = tf.reshape(delayed_ask, shape = (-1, 1))
        reg_bid = tf.reshape(reg_bid, shape = (-1, 1))
        reg_ask = tf.reshape(reg_ask, shape = (-1, 1))

        bid_not_skip_mask = tf.logical_and(
            y_pred < bid * (1 - 1e-4 * self.interest),
            y_pred < delayed_bid * (1 - 1e-4 * self.interest)
        )
        ask_not_skip_mask = tf.logical_and(
            y_pred > ask * (1 + 1e-4 * self.interest),
            y_pred > delayed_ask * (1 + 1e-4 * self.interest),
        )

        self.metric.assign_add(tf.math.reduce_sum(-10000 * (tf.boolean_mask(reg_ask, bid_not_skip_mask) / tf.boolean_mask(delayed_bid, bid_not_skip_mask) - 1) - 1.8))
        self.metric.assign_add(tf.math.reduce_sum(10000 * (tf.boolean_mask(reg_bid, ask_not_skip_mask) / tf.boolean_mask(delayed_ask, ask_not_skip_mask) - 1) - 1.8))

    def result(self):
        return self.metric
    
    def reset_state(self):
        self.metric.assign(0)