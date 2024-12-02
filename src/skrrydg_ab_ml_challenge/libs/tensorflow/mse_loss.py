import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def mse_loss(y_true, y_pred):
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
    
    return tf.keras.losses.MSE(y_pred, (reg_bid + reg_ask) / 2 - (ask + bid) / 2)
