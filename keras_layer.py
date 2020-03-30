from keras import backend as K
from keras.layers import Layer, LSTM, MaxPooling1D
import numpy as np

class TFiLM(Layer):

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        super(tFiLM, self).__init__(**kwargs)

    def make_normalizer(self, x_in):
        """ Pools to downsample along 'temporal' dimension and then 
            runs LSTM to generate normalization weights.
        """
        x_in_down = (MaxPooling1D(pool_size=self.block_size, padding='valid'))(x_in)
        x_rnn = self.rnn(x_in_down)
        return x_rnn
      
    def apply_normalizer(self, x_in, x_norm):
        """
        Applies normalization weights by multiplying them into their respective blocks.
        """

        n_blocks = K.shape(x_in)[1] / self.block_size
        n_filters = K.shape(x_in)[2] 
            
        # reshape input into blocks
        x_norm = K.reshape(x_norm, shape=(-1, n_blocks, 1, n_filters))
        x_in = K.reshape(x_in, shape=(-1, n_blocks, self.block_size, n_filters))
        
        # multiply
        x_out = x_norm * x_in

        # return to original shape
        x_out = K.reshape(x_out, shape=(-1, n_blocks * self.block_size, n_filters))

        return x_out


    def build(self, input_shape):
        self.rnn = LSTM(units = input_shape[2], return_sequences = True, trainable=True)
        self.rnn.build(input_shape)
        self._trainable_weights = self.rnn.trainable_weights
        super(tFiLM, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert len(x.shape) == 3, 'Input should be tensor with dimension \
                                   (batch_size, steps, num_features).'
        assert x.shape[1] % self.block_size == 0, 'Number of steps must be a \
                                                   multiple of the block size.'

        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
