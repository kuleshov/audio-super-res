import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------------

def SubPixel1D_v2(I, r):
  """One-dimensional subpixel upsampling layer

  Based on https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py
  """
  with tf.name_scope('subpixel'):
    bsize, a, r = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.split(1, a, I)  # a, [bsize, 1, r]
    if 'axis' in tf.squeeze.func_code.co_varnames:
      X = tf.concat(1, [tf.squeeze(x, axis=1) for x in X])  # bsize, a*r
    elif 'squeeze_dims' in tf.squeeze.func_code.co_varnames:
      X = tf.concat(1, [tf.squeeze(x, squeeze_dims=[1]) for x in X])  # bsize, a*r
    else:
      raise Exception('Unsupported version of tensorflow')
    return tf.reshape(X, (bsize, a*r, 1))

def SubPixel1D(I, r):
  """One-dimensional subpixel upsampling layer

  Calls a tensorflow function that directly implements this functionality.
  We assume input has dim (batch, width, r)
  """
  with tf.name_scope('subpixel'):
    X = tf.transpose(I, [2,1,0]) # (r, w, b)
    X = tf.batch_to_space_nd(X, [r], [[0,0]]) # (1, r*w, b)
    X = tf.transpose(X, [2,1,0])
    return X

def SubPixel1D_multichan(I, r):
  """One-dimensional subpixel upsampling layer

  Calls a tensorflow function that directly implements this functionality.
  We assume input has dim (batch, width, r).

  Works with multiple channels: (B,L,rC) -> (B,rL,C)
  """
  with tf.name_scope('subpixel'):
    _, w, rc = I.get_shape()
    assert rc % r == 0
    c = rc / r
    X = tf.transpose(I, [2,1,0]) # (rc, w, b)
    X = tf.batch_to_space_nd(X, [r], [[0,0]]) # (c, r*w, b)
    X = tf.transpose(X, [2,1,0])
    return X      

# ----------------------------------------------------------------------------

# demonstration
if __name__ == "__main__":
  with tf.Session() as sess:
    x = np.arange(2*4*2).reshape(2, 4, 2)
    X = tf.placeholder("float32", shape=(2, 4, 2), name="X")
    Y = SubPixel1D(X, 2)
    y = sess.run(Y, feed_dict={X: x})

    print 'single-channel:'
    print 'original, element 0 (2 channels):', x[0,:,0], x[0,:,1]
    print 'rescaled, element 1:', y[0,:,0]
    print
    print 'original, element 0 (2 channels) :', x[1,:,0], x[1,:,1]
    print 'rescaled, element 1:', y[1,:,0]
    print

    x = np.arange(2*4*4).reshape(2, 4, 4)
    X = tf.placeholder("float32", shape=(2, 4, 4), name="X")
    Y = SubPixel1D(X, 2)
    y = sess.run(Y, feed_dict={X: x})

    print 'multichannel:'
    print 'original, element 0 (4 channels):', x[0,:,0], x[0,:,1], x[0,:,2], x[0,:,3]
    print 'rescaled, element 1:', y[0,:,0], y[0,:,1]
    print
    print 'original, element 0 (2 channels) :', x[1,:,0], x[1,:,1], x[1,:,2], x[1,:,3]
    print 'rescaled, element 1:', y[1,:,0], y[1,:,1],