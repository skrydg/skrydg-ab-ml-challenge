import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def weighted_r2_loss(y_true, y_pred):
    target_shape = y_pred.shape[1] // 2

    weights = y_pred[:, target_shape:2 * target_shape]
    y_pred = y_pred[:, 0:target_shape]
    

    y_true = tf.reshape(y_true, shape = (-1, 1))
    y_pred = tf.reshape(y_pred, shape = (-1, 1))
    weights = tf.reshape(weights, shape = (-1, 1))

    unexplained_error = tf.math.reduce_sum(weights * tf.square(y_true - y_pred))
    total_error = tf.math.reduce_sum(weights * tf.square(y_true))
    R_squared = unexplained_error / total_error
    return R_squared