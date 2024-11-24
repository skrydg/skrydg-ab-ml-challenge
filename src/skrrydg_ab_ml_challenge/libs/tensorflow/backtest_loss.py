import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def backtest_loss(y_true, y_pred):
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
        y_pred < bid,
        y_pred < delayed_bid
    )
    ask_not_skip_mask = tf.logical_and(
        y_pred > ask,
        y_pred > delayed_ask
    )

    res = tf.math.reduce_sum(-10000 * (tf.boolean_mask(reg_ask, bid_not_skip_mask) / tf.boolean_mask(delayed_bid, bid_not_skip_mask) - 1) - 1.8)
    res = res + tf.math.reduce_sum(10000 * (tf.boolean_mask(reg_bid, ask_not_skip_mask) / tf.boolean_mask(delayed_ask, ask_not_skip_mask) - 1) - 1.8)

    return res
