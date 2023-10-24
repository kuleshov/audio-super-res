import numpy as np
import tensorflow as tf
from scipy import interpolate
from tensorflow.compat.v1.keras import backend as K
# from keras import backend as K
from tensorflow.compat.v1.keras.initializers import Orthogonal
from tensorflow.compat.v1.keras.initializers import RandomNormal as Normal
from tensorflow.compat.v1.keras.layers import (LSTM, Activation, Add,
                                               AveragePooling1D,
                                               BatchNormalization, Concatenate,
                                               Convolution1D, Dropout,
                                               LeakyReLU, MaxPooling1D,
                                               MaxPooling2D, UpSampling1D)

from .layers.subpixel import SubPixel1D, SubPixel1D_multichan, SubPixel1D_v2
from .model import Model, default_opt

# ----------------------------------------------------------------------------
DRATE = 2
class AudioTfilm(Model):

  def __init__(self, from_ckpt=False, n_dim=None, r=2, pool_size = 4, strides=4,
               opt_params=default_opt, log_prefix='./run'):

    #tf.compat.v1.disable_resource_variables()
    tf.compat.v1.disable_eager_execution()

    # perform the usual initialization
    self.r = r
    self.pool_size = pool_size
    self.strides = strides
    Model.__init__(self, from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                   opt_params=opt_params, log_prefix=log_prefix)

  def create_model(self, n_dim, r):
    # load inputs
    X, _, _ = self.inputs
    K.set_session(self.sess)

    with tf.compat.v1.name_scope('generator'):
      x = X
      #print("loaded inputs", x)
      L = self.layers
      n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
      n_blocks = [ 128, 64, 32, 16, 8]
      n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
      downsampling_l = []

      print('building model...')

      def _make_normalizer(x_in, n_filters, n_block):
        """applies an lstm layer on top of x_in"""
        x_shape = tf.shape(input=x_in)
        n_steps = x_shape[1] / tf.dtypes.cast(n_block, tf.int32) # will be 32 at training

        # first, apply standard conv layer to reduce the dimension
        # input of (-1, 4096, 128) becomes (-1, 32, 128)
        # input of (-1, 512, 512) becomes (-1, 32, 512)

        x_in_down = (MaxPooling1D(pool_size=(n_block,), padding='valid'))(x_in)

        # pooling to reduce dimension
        x_shape = tf.shape(input=x_in_down)

        x_rnn = LSTM(units = n_filters, return_sequences = True)(x_in_down)
        #x_rnn = LSTM(units = n_filters)(x_in_down)

        # print('X_NN', x_rnn)
        # output: (-1, n_steps, n_filters)
        return x_rnn

      def _apply_normalizer(x_in, x_norm, n_filters, n_block):
        x_shape = tf.shape(input=x_in)
        n_steps = x_shape[1] / tf.dtypes.cast(n_block, tf.int32) # will be 32 at training

        # reshape input into blocks

        x_in = tf.reshape(x_in, shape=(-1, tf.dtypes.cast(n_steps, tf.int32), tf.dtypes.cast(n_block, tf.int32), n_filters))
        x_norm = tf.reshape(x_norm, shape=(-1, tf.dtypes.cast(n_steps, tf.int32), 1, n_filters))

        # multiply
        x_out = x_norm * x_in

        # return to original shape
        x_out = tf.reshape(x_out, shape=x_shape)

        #print("x_out", x_out)

        return x_out


      # downsampling layers
      for l, nf, fs in zip(list(range(L)), n_filters, n_filtersizes):
        with tf.compat.v1.name_scope('downsc_conv%d' % l):
          x = (Convolution1D(filters=nf, kernel_size=fs, dilation_rate = DRATE,
                  activation=None, padding='same', kernel_initializer='orthogonal',
                  strides=1))(x)
          x = (MaxPooling1D(pool_size=2,padding='valid'))(x)
          x = LeakyReLU(0.2)(x)

          # create and apply the normalizer
          nb = 128 / (2**l)

          params_before = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
          x_norm = _make_normalizer(x, nf, nb)
          params_after = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])

          # print('x_norm:', x_norm)

          x = _apply_normalizer(x, x_norm, nf, nb)

          print('D-Block: ', x.get_shape())
          downsampling_l.append(x)

      # bottleneck layer
      with tf.compat.v1.name_scope('bottleneck_conv'):
          x = (Convolution1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], dilation_rate = DRATE,
                  activation=None, padding='same', kernel_initializer='orthogonal',
                  strides=1))(x)
          x = (MaxPooling1D(pool_size=2,padding='valid'))(x)
          x = Dropout(rate=0.5)(x)
          x = LeakyReLU(0.2)(x)

          # create and apply the normalizer
          nb = 128 / (2**L)
          x_norm = _make_normalizer(x, n_filters[-1], nb)
          x = _apply_normalizer(x, x_norm, n_filters[-1], nb)

      # upsampling layers
      for l, nf, fs, l_in in reversed(list(zip(list(range(L)), n_filters, n_filtersizes, downsampling_l))):
        with tf.compat.v1.name_scope('upsc_conv%d' % l):
          # (-1, n/2, 2f)
          x = (Convolution1D(filters=2*nf, kernel_size=fs, dilation_rate = DRATE,
                  activation=None, padding='same', kernel_initializer='orthogonal'))(x)

          x = Dropout(rate=0.5)(x)
          x = Activation('relu')(x)
          # (-1, n, f)
          x = SubPixel1D(x, r=2)

          # create and apply the normalizer
          x_norm = _make_normalizer(x, nf, nb)
          x = _apply_normalizer(x, x_norm, nf, nb)
          # (-1, n, 2f)
          x = Concatenate()([x, l_in])  # THIS APPEARS TO CAUSE A DIMENSION MISMATCH BY INCREASING THE SIZE OF X
          #x = merge([x, l_in], mode='concat', concat_axis=-1)
          print('U-Block: ', x.get_shape())

      # final conv layer
      with tf.compat.v1.name_scope('lastconv'):
        x = Convolution1D(filters=2, kernel_size=9,
                activation=None, padding='same', kernel_initializer='random_normal')(x)

        x = SubPixel1D(x, r=2)
      #g = x
      #g = x + X
      g = Add()([x, X])
      # print("subPixel1D", x)
      # print("subPixel1D", X)
      # print("subPixel1D", g.shape)
     # g = merge([x, X], mode='sum')
    return g

  def predict(self, X):
    assert len(X) == 1
    x_sp = spline_up(X, self.r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(self.layers+1)))]
    X = x_sp.reshape((1,len(x_sp),1))
    feed_dict = self.load_batch((X,X), train=False)
    return self.sess.run(self.predictions, feed_dict=feed_dict)

# ----------------------------------------------------------------------------
# helpers

def normal_init(shape, data_format='tf', name=None):
    return Normal(shape, scale=1e-3, name=name, data_format=data_format)

def orthogonal_init(shape, data_format='tf', dtype=None):
    return Orthogonal(shape, dtype=dtype, data_format=data_format)

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp
