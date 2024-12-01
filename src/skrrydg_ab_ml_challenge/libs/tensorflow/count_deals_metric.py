import tensorflow as tf

class CountDealsMetric(tf.keras.Metric):
    def __init__(self, name='backtest_metric', interest=6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.interest = interest
        self.count_deals = self.add_variable(
            shape=(),
            initializer='zeros',
            name='count_deals',
            dtype=tf.float32
        )
        self.mask = None

    def update_state(self, y_true, y_pred):
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

        mask = tf.math.logical_and(tf.math.logical_not(tf.math.is_nan(bid)), tf.math.logical_not(tf.math.is_nan(ask)))
        mask = tf.math.logical_and(mask, tf.math.logical_not(tf.math.is_nan(delayed_bid)))
        mask = tf.math.logical_and(mask, tf.math.logical_not(tf.math.is_nan(delayed_ask)))
        mask = tf.math.logical_and(mask, tf.math.logical_not(tf.math.is_nan(reg_bid)))
        mask = tf.math.logical_and(mask, tf.math.logical_not(tf.math.is_nan(reg_ask)))
        
        y_pred = tf.boolean_mask(y_pred, mask)
        bid = tf.boolean_mask(bid, mask)
        ask = tf.boolean_mask(ask, mask)
        delayed_bid = tf.boolean_mask(delayed_bid, mask)
        delayed_ask = tf.boolean_mask(delayed_ask, mask)
        reg_bid = tf.boolean_mask(reg_bid, mask)
        reg_ask = tf.boolean_mask(reg_ask, mask)
        
        mid_price = (bid + ask) / 2
        spread = mid_price - bid
        fair_price = mid_price + tf.math.multiply(y_pred, spread)
        
        bid_not_skip_mask = tf.logical_and(
            fair_price < bid * (1 - 1e-4 * self.interest),
            fair_price < delayed_bid * (1 - 1e-4 * self.interest)
        )
        ask_not_skip_mask = tf.logical_and(
            fair_price > ask * (1 + 1e-4 * self.interest),
            fair_price > delayed_ask * (1 + 1e-4 * self.interest),
        )
        self.count_deals.assign_add(tf.math.reduce_sum(tf.cast(bid_not_skip_mask, tf.int32)))
        self.count_deals.assign_add(tf.math.reduce_sum(tf.cast(ask_not_skip_mask, tf.int32)))

        self.mask = tf.logical_or(bid_not_skip_mask, ask_not_skip_mask)

    def result(self):
        return self.count_deals
    
    def reset_state(self):
        self.count_deals.assign(0)