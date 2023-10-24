import os
import pickle
import time

import librosa
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers import LSTM, Dense, Input
from keras.models import Model
from tensorflow.compat.v1.keras import backend as K
from tqdm import tqdm

# from keras import backend as K
from .dataset import DataSet

# ----------------------------------------------------------------------------

default_opt   = { 'alg': 'adam', 'lr': 1e-4, 'b1': 0.99, 'b2': 0.999,
                  'layers': 2, 'batch_size': 128 }

class Model2(object):
  """Generic tensorflow model training code"""
  #N_FFT = 2048

  def __init__(self, r=2, opt_params=default_opt):

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    K.set_session(self.sess) # pass keras the session

    # save params
    self.opt_params = opt_params
    self.layers     = opt_params['layers']

  def get_power(self, x):
    S = librosa.stft(x, 2048)
    p = np.angle(S)
    S = np.log(np.abs(S)**2 + 1e-8)
    return S

  def compute_log_distortion(self, x_hr, x_pr):
      S1 = self.get_power(x_hr)
      S2 = self.get_power(x_pr)
      lsd = np.mean(np.sqrt(np.mean((S1-S2)**2 + 1e-8, axis=1)), axis = 0)
      return min(lsd, 10.)

  def create_train_op(self, X, Y, alpha):
    # load params
    opt_params = self.opt_params
    print('creating train_op with params:', opt_params)

    # create loss
    self.loss = self.create_objective(X, Y, opt_params)

    # create params
    params = self.get_params()

    # create optimizer
    self.optimizer = self.create_optimzier(opt_params)

    # create gradients
    grads = self.create_gradients(self.loss, params)

    # create training op
    with tf.compat.v1.name_scope('optimizer'):
      train_op = self.create_updates(params, grads, alpha, opt_params)

    # initialize the optimizer variabLes
    optimizer_vars = [ v for v in tf.compat.v1.global_variables() if 'optimizer/' in v.name
                                                        or 'Adam' in v.name ]
    init = tf.compat.v1.variables_initializer(optimizer_vars)
    self.sess.run(init)

    return train_op

  def create_model(self, n_dim, r):
    raise NotImplementedError()

  def get_params(self):
    return [ v for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
             if 'soundnet' not in v.name ]

  def create_optimzier(self, opt_params):
    if opt_params['alg'] == 'adam':
      lr, b1, b2 = opt_params['lr'], opt_params['b1'], opt_params['b2']
      optimizer = tf.compat.v1.train.AdamOptimizer(lr, b1, b2)
    else:
      raise ValueError('Invalid optimizer: ' + opt_params['alg'])

    return optimizer

  def calculate_snr(self, Y, Pred):
    sqrt_l2_loss = np.sqrt(np.mean((Pred-Y)**2 + 1e-6, axis=(0, 1)))
    sqrt_l2_norm = np.sqrt(np.mean(Y**2, axis=(0, 1)))
    snr = 20 * np.log(sqrt_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)
    return snr

  def create_gradients(self, loss, params):
    gv = self.optimizer.compute_gradients(loss, params)
    g, v = list(zip(*gv))
    return g

  def create_updates(self, params, grads, alpha, opt_params):
    # create a variable to track the global step.
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    # update grads
    grads = [alpha*g for g in grads]

    # use the optimizer to apply the gradients that minimize the loss
    gv = list(zip(grads, params))
    train_op = self.optimizer.apply_gradients(gv, global_step=self.global_step)

    return train_op

  def run(self, X_train, Y_train, X_val, Y_val, n_epoch=100, r=4, speaker="single", grocery="false", latent_dim=64):


    def create_objective(y_true, y_pred):
        # compute l2 loss
        sqrt_l2_loss = K.sqrt(K.mean(K.square(y_true-y_pred) + 1e-6, axis=[1,2]))
        avg_sqrt_l2_loss = K.mean(sqrt_l2_loss, axis=0)
        return avg_sqrt_l2_loss

    X = X_train
    Y = Y_train
    dim = 512
    X = X[:128,:dim]
    Y = Y[:128,:dim]

    pad = np.zeros((Y.shape[0], 1, 1))

    Y_input = np.concatenate((Y, pad), axis = 1)
    Y_target = np.concatenate((pad, Y), axis = 1)

    val_ratio = X_val.shape[1]/X.shape[1]
    patch_size = X.shape[1]

    encoder_inputs = Input(shape=(patch_size,1))
    encoder = LSTM(latent_dim, return_state=True, kernel_initializer='zeros')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(patch_size+1,1))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, kernel_initializer='zeros')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    opt = optimizers.Adam(lr=1e-6)
    model.compile(loss=create_objective, optimizer=opt)

    print((model.summary()))

    model.fit([X, Y_input], Y_target, batch_size = self.opt_params['batch_size'], epochs = 10, validation_split=val_ratio)

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_dense = Dense(1, activation='relu')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_states = [state_h, state_c]
    decoder_model = Model([decoder_inputs]+decoder_states_inputs, [decoder_outputs]+decoder_states)

    def decode_sequence(input_seq):
        states_value = encoder_model.predict(input_seq)

        target_seq = np.zeros((input_seq.shape[0], patch_size+1, 1))
        decoded_seq = np.zeros((input_seq.shape[0], patch_size,1))
        for i in tqdm(list(range(patch_size))):
            output, h, c = decoder_model.predict([target_seq]+states_value)

            output = np.reshape(output, (input_seq.shape[0], output.shape[1]))
            this_output = output[:,i]
            this_output.shape = (this_output.shape[0], 1)
            decoded_seq[:,i] = this_output
            target_seq[:, i, 0] = output[:,i]
            states_value = [h, c]

        return decoded_seq

    snrs = []
    batch_size = 256
    preds = []
    for i in tqdm(list(range(X_val.shape[0] / batch_size))):
        input_seq = X_val[i*batch_size:(i+1)*batch_size,:dim]
        decoded_seq = decode_sequence(input_seq)
        preds.append(decoded_seq)
        for j in range(decoded_seq.shape[0]):
            snr = self.calculate_snr(Y_val[i*batch_size+j:i*batch_size+j+1,:dim], decoded_seq[j])
            snrs.append(snr)

    avg_snr = np.mean(snr)
    print(("avg_snr: " + str(avg_snr)))

    preds = np.array(preds)

    print((Y_val.shape))
    Y_val = np.reshape(Y_val, (Y_val.shape[0], Y_val.shape[1]))[:preds.shape[1], :preds.shape[2]]
    preds = np.reshape(preds, (preds.shape[1], preds.shape[2]))
    Y_val = Y_val.flatten()
    preds = preds.flatten()
    print((Y_val.shape))
    print((preds.shape))
    lsd = self.compute_log_distortion(Y_val, preds)
    print(("avg lsd: " + str(lsd)))

# ----------------------------------------------------------------------------
# helpers

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
    else:
        excerpt = slice(start_idx, start_idx + batchsize)
    yield inputs[excerpt], targets[excerpt]

def count_parameters():
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        shape = variable.get_shape()
        var_params = 1
        for dim in shape:
            var_params *= dim.value
        total_parameters += var_params
    return total_parameters
