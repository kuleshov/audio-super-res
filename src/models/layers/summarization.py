import tensorflow as tf

# ----------------------------------------------------------------------------

def create_var_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.compat.v1.name_scope('summaries'):
    mean = tf.reduce_mean(input_tensor=var)
    tf.compat.v1.summary.scalar('mean', mean)
    with tf.compat.v1.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
    tf.compat.v1.summary.scalar('stddev', stddev)
    tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
    tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
    tf.compat.v1.summary.histogram('histogram', var)