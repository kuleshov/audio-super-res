from keras import backend as K
from keras.layers import Layer, LSTM, MaxPooling1D

class TFiLM(Layer):
    """
    Input should be a tensor of shape (batch_size, steps, num_features)
    Output is a tensor with the same shape
    """

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        super(TFiLM, self).__init__(**kwargs)

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
        n_blocks = int(x_in.shape[1] / self.block_size)
        n_filters = x_in.shape[2] 
        
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
        super(TFiLM, self).build(input_shape)

    def call(self, x):
        assert len(x.shape) == 3, 'Input should be tensor with dimension \
                                   (batch_size, steps, num_features).'
        assert x.shape[1] % self.block_size == 0, 'Number of steps must be a \
                                                   multiple of the block size.'

        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x
