import numpy as np
import tensorflow as tf

from scipy import interpolate
from model import Model, default_opt

from layers.subpixel import SubPixel1D, SubPixel1D_v2

from keras import backend as K
from keras.layers import merge
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.initializations import normal, orthogonal

# ----------------------------------------------------------------------------

class AudioUNet(Model):
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
      downsampling_l = []

      print 'building model...'

      # downsampling layers
      for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('downsc_conv%d' % l):
          x = (Convolution1D(nb_filter=nf, filter_length=fs, 
                  activation=None, border_mode='same', init=orthogonal_init,
                  subsample_length=2))(x)
          # if l > 0: x = BatchNormalization(mode=2)(x)
          x = LeakyReLU(0.2)(x)
          print 'D-Block: ', x.get_shape()
          downsampling_l.append(x)

      # bottleneck layer
      with tf.name_scope('bottleneck_conv'):
          x = (Convolution1D(nb_filter=n_filters[-1], filter_length=n_filtersizes[-1], 
                  activation=None, border_mode='same', init=orthogonal_init,
                  subsample_length=2))(x)
          x = Dropout(p=0.5)(x)
          # x = BatchNormalization(mode=2)(x)
          x = LeakyReLU(0.2)(x)

      # upsampling layers
      for l, nf, fs, l_in in reversed(zip(range(L), n_filters, n_filtersizes, downsampling_l)):
        with tf.name_scope('upsc_conv%d' % l):
          # (-1, n/2, 2f)
          x = (Convolution1D(nb_filter=2*nf, filter_length=fs, 
                  activation=None, border_mode='same', init=orthogonal_init))(x)
          # x = BatchNormalization(mode=2)(x)
          x = Dropout(p=0.5)(x)
          x = Activation('relu')(x)
          # (-1, n, f)
          x = SubPixel1D(x, r=2) 
          # (-1, n, 2f)
          x = merge([x, l_in], mode='concat', concat_axis=-1) 
          print 'U-Block: ', x.get_shape()

      # final conv layer
      with tf.name_scope('lastconv'):
        x = Convolution1D(nb_filter=2, filter_length=9, 
                activation=None, border_mode='same', init=normal_init)(x)    
        x = SubPixel1D(x, r=2) 
        print x.get_shape()

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

def normal_init(shape, dim_ordering='tf', name=None): 
  return normal(shape, scale=1e-3, name=name, dim_ordering=dim_ordering)

def orthogonal_init(shape, dim_ordering='tf', name=None): 
  return orthogonal(shape, name=name, dim_ordering=dim_ordering)  

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp