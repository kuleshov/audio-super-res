import sys
import numpy as np
import tensorflow as tf

from scipy import interpolate
from .model import Model, default_opt

from .layers.subpixel import SubPixel1D, SubPixel1D_v2

from tensorflow.python.keras import backend as K
from keras.layers import Concatenate, Add, MaxPooling1D, MaxPooling2D, AveragePooling1D
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import UpSampling1D#, AtrousConvolution1D
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
#from keras.initializations import normal, orthogonal
from keras.initializers import RandomNormal, Orthogonal

# ----------------------------------------------------------------------------
DRATE = 2
class AudioTfilm(Model):

  def __init__(self, from_ckpt=False, n_dim=None, r=2, pool_size = 4, strides=4,
               opt_params=default_opt, log_prefix='./run'):
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
      L = self.layers
      n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
      n_blocks = [ 128, 64, 32, 16, 8]
      n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
      downsampling_l = []

      print('building model...')

      def _make_normalizer(x_in, n_filters, n_block):
        """applies an lstm layer on top of x_in"""
        x_shape = tf.shape(input=x_in)

        n_steps = x_shape[1] / int(n_block) # will be 32 at training

        # first, apply standard conv layer to reduce the dimension
        # input of (-1, 4096, 128) becomes (-1, 32, 128)
        # input of (-1, 512, 512) becomes (-1, 32, 512)

        x_in_down = (MaxPooling1D(pool_size=int(n_block), padding='valid'))(x_in)

        # pooling to reduce dimension
        x_shape = tf.shape(input=x_in_down)

        x_rnn = LSTM(units = n_filters, return_sequences = True)(x_in_down)

        # output: (-1, n_steps, n_filters)
        return x_rnn

      def _apply_normalizer(x_in, x_norm, n_filters, n_block):
        x_shape = tf.shape(input=x_in)
        #x_in = tf.compat.v1.Print(x_in, [x_shape], summarize=5, message=' x_in: ')
        #x_norm = tf.compat.v1.Print(x_norm, [tf.shape(x_norm)], summarize=5, message=' x_norm: ')
        n_steps = x_shape[1] / int(n_block) # will be 32 at training

        # reshape input into blocks
        x_in = tf.reshape(x_in, shape=(-1, n_steps, int(n_block), n_filters))
        x_norm = tf.reshape(x_norm, shape=(-1, n_steps, 1, n_filters))

        # multiply
        #x_norm = tf.compat.v1.Print(x_norm, [tf.shape(x_norm)], summarize=5)
        #x_in= tf.compat.v1.Print(x_in, [tf.shape(x_in)], summarize=5)
        #with tf.control_dependencies([print_op_1, print_op_2]):
        x_out = x_norm * x_in

        # return to original shape
        x_out = tf.reshape(x_out, shape=x_shape)

        return x_out


      # downsampling layers
      for l, nf, fs in zip(list(range(L)), n_filters, n_filtersizes):
        # x = tf.compat.v1.Print(x, [tf.shape(x)], summarize=5, message='down: ')
        with tf.compat.v1.name_scope('downsc_conv%d' % l):
          x = (Conv1D(filters=nf, kernel_size=fs, dilation_rate = DRATE,
                  activation=None, padding='same', kernel_initializer=Orthogonal()))(x)
          x = (MaxPooling1D(pool_size=2,padding='valid'))(x)
          x = LeakyReLU(0.2)(x)

          # create and apply the normalizer
          nb = 128 / (2**l)

          params_before = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
          x_norm = _make_normalizer(x, nf, nb)
          params_after = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])

          x = _apply_normalizer(x, x_norm, nf, nb)

          print('D-Block: ', x.get_shape())
          downsampling_l.append(x)

      # x = tf.compat.v1.Print(x, [tf.shape(x)], summarize=5, message='bottleneck: ')
      # bottleneck layer
      with tf.compat.v1.name_scope('bottleneck_conv'):
          x = (Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], dilation_rate = DRATE,
                  activation=None, padding='same', kernel_initializer=Orthogonal))(x)
          x = (MaxPooling1D(pool_size=2,padding='valid'))(x)
          x = Dropout(rate=0.5)(x)
          x = LeakyReLU(0.2)(x)

          # create and apply the normalizer
          nb = 128 / (2**L)
          x_norm = _make_normalizer(x, n_filters[-1], nb)
          x = _apply_normalizer(x, x_norm, n_filters[-1], nb)

      # upsampling layers
      for l, nf, fs, l_in in reversed(list(zip(list(range(L)), n_filters, n_filtersizes, downsampling_l))):
        # x = tf.compat.v1.Print(x, [tf.shape(x)], summarize=5, message='up: ')
        with tf.compat.v1.name_scope('upsc_conv%d' % l):
          # (-1, n/2, 2f)
          x = (Conv1D(filters=2*nf, kernel_size=fs, dilation_rate = DRATE,
                  activation=None, padding='same', kernel_initializer=Orthogonal()))(x)

          x = Dropout(rate=0.5)(x)
          x = Activation('relu')(x)
          # (-1, n, f)
          x = SubPixel1D(x, r=2)

          # create and apply the normalizer

          x_norm = _make_normalizer(x, nf, nb)
          x = _apply_normalizer(x, x_norm, nf, nb)
          # (-1, n, 2f)
          x = Concatenate()([x, l_in])
          print('U-Block: ', x.get_shape())

      # final conv layer
      with tf.compat.v1.name_scope('lastconv'):
        x = Conv1D(filters=2, kernel_size=9,
                activation=None, padding='same', kernel_initializer=RandomNormal(stddev=1e-3))(x)
        x = SubPixel1D(x, r=2)

      g = Add()([x, X])
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

def normal_init(shape, dim_ordering='tf', name=None):
    return RandomNormal(stdev=1e-3)
    #return normal(shape, scale=1e-3, name=name, dim_ordering=dim_ordering)

def orthogonal_init(shape, dim_ordering='tf', name=None):
    return Orthogonal()
    #return orthogonal(shape, name=name, dim_ordering=dim_ordering)

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp
