import os

import numpy as np
import tensorflow as tf
# from keras import backend as K
from keras.initializers import RandomNormal as normal
from keras.layers import Dense, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from scipy import interpolate
from tensorflow.compat.v1.keras import backend as K

from .layers.subpixel import SubPixel1D, SubPixel1D_v2
from .model import Model, default_opt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'





# ----------------------------------------------------------------------------

class DNN(Model):
  """Generic tensorflow model training code"""

  def __init__(self, from_ckpt=False, n_dim=None, r=2,
               opt_params=default_opt, log_prefix='./run'):
    # perform the usual initialization
    self.r = r
    Model.__init__(self, from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                   opt_params=opt_params, log_prefix=log_prefix)

  def create_model(self, n_dim, r):
    # load inputs
    X, _, _ = self.inputs
    K.set_session(self.sess)

    with tf.compat.v1.name_scope('generator'):
      x = X

      L = self.layers
      init_x_shape = tf.shape(input=x)

      x = tf.reshape(x, [-1, n_dim])

      for l in range(L):
        x_start = x
        x_shape = tf.shape(input=x)
        if(l == L-1):
            out_units = n_dim
        else:
            if(l == 0):
                out_units = n_dim/6
            else:
                out_units = n_dim/2

        if(l == 0):
            in_shape = n_dim
        else:
            if(l==0):
                in_shape = n_dim/6
            else:
                in_shape = n_dim/2

        #x = Dense(units=out_units, input_shape=(in_shape,), init=normal_init)(x)
        x = Dense(out_units, init=normal_init ,input_dim=(in_shape,))(x)
        x_shape = tf.shape(input=x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        if(x_shape == tf.shape(input=x_start)):
            x = tf.add(x, x_start)

      x = tf.reshape(x, shape=(init_x_shape[0], init_x_shape[1], 1))
      return tf.add(x, X)


  def predict(self, X):
    assert len(X) == 1
    x_sp = spline_up(X, self.r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(self.layers+1)))]
    X = x_sp.reshape((1,len(x_sp),1))
    feed_dict = self.load_batch((X,X), train=False)
    return self.sess.run(self.predictions, feed_dict=feed_dict)

# ----------------------------------------------------------------------------
# helpers

def normal_init(shape, dim_ordering='tf', name=None):
    return normal(shape, scale=0.0000001, name=name, dim_ordering=dim_ordering)

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp
