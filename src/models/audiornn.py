import numpy as np
import tensorflow as tf

from scipy import interpolate
from model import Model, default_opt

from keras import backend as K
from keras.layers import merge
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Convolution1D, UpSampling1D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# ----------------------------------------------------------------------------

class AudioLSTM(Model):

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

    with tf.name_scope('generator'):
      x = X
      L = self.layers
      # dim/layer: 4096, 2048, 1024, 512, 256, 128,  64,  32,
      # n_filters = [  64,  128,  256, 384, 384, 384, 384, 384]
      n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
      # n_filters = [  256,  512,  512, 512, 512, 1024, 1024, 1024]
      # n_filtersizes = [129, 65,   33,  17,  9,  9,  9, 9]
      # n_filtersizes = [31, 31,   31,  31,  31,  31,  31, 31]
      n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]

      print 'building model...'

      L=1

      # we reshape the input into blocks of length 128, so that the sequence is length 64
      # that way, it will not be too hard to train the RNN
      x = tf.reshape(x, shape=(-1, 64, 1, 128, 1))

      for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('rnn%d' % l):
          x = ConvLSTM2D(filters=nf, kernel_size=(1,fs), padding='same', 
                         return_sequences=True, implementation=2)(x)
          print 'Conv-LSTM-Block: ', x.get_shape()

      # final conv layer
      with tf.name_scope('lastconv'):
        # generate the output
        x = ConvLSTM2D(filters=1, kernel_size=(1,9), padding='same', 
                       return_sequences=True, implementation=2)(x)
        print x.get_shape()

      # make the output the original shape
      x = tf.reshape(x, shape=(-1, 8192, 1))

      g = merge([x, X], mode='sum')

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

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp
