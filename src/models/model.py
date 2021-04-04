import os
import time

import numpy as np
import tensorflow as tf
import pickle

import librosa
from tensorflow.python.keras import backend as K
from .dataset import DataSet
from tqdm import tqdm

# ----------------------------------------------------------------------------

default_opt = {'alg': 'adam', 'lr': 1e-4, 'b1': 0.99, 'b2': 0.999,
               'layers': 2, 'batch_size': 128}


class Model(object):
    """Generic tensorflow model training code"""

    def __init__(self, from_ckpt=False, n_dim=None, r=2,
                 opt_params=default_opt, log_prefix='./run'):

        # create session
        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        #gpu_options = tf.GPUOptions(allow_growth=True)
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        self.sess = tf.compat.v1.Session()
        K.set_session(self.sess)  # pass keras the session

        # save params
        self.opt_params = opt_params
        self.layers = opt_params['layers']

        if from_ckpt:
            pass  # we will instead load the graph from a checkpoint
        else:
            # create input vars
            X = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 1), name='X')
            Y = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 1), name='Y')
            alpha = tf.compat.v1.placeholder(tf.float32, shape=(),
                                   name='alpha')  # weight multiplier
            # save inputs
            self.inputs = (X, Y, alpha)
            tf.compat.v1.add_to_collection('inputs', X)
            tf.compat.v1.add_to_collection('inputs', Y)
            tf.compat.v1.add_to_collection('inputs', alpha)

            # create model outputs
            self.predictions = self.create_model(n_dim, r)
            tf.compat.v1.add_to_collection('preds', self.predictions)
            # init the model
            init = tf.compat.v1.global_variables_initializer()
            self.sess.run(init)

            # create training updates
            self.train_op = self.create_train_op(X, Y, alpha)
            tf.compat.v1.add_to_collection('train_op', self.train_op)
        # logging
        lr_str = '.' + 'lr%f' % opt_params['lr']
        g_str = '.g%d' % self.layers
        b_str = '.b%d' % int(opt_params['batch_size'])
        self.logdir = log_prefix + lr_str + '.%d' % r + g_str + b_str
        self.checkpoint_root = os.path.join(self.logdir, 'model.ckpt')

    def get_power(self, x):
        S = librosa.stft(x, 2048)
        p = np.angle(S)
        S = np.log(np.abs(S)**2 + 1e-8)
        return S

    def compute_log_distortion(self, x_hr, x_pr):
        S1 = self.get_power(x_hr)
        S2 = self.get_power(x_pr)
        lsd = np.mean(np.sqrt(np.mean((S1-S2)**2 + 1e-8, axis=1)), axis=0)
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
        optimizer_vars = [v for v in tf.compat.v1.global_variables() if 'optimizer/' in v.name
                          or 'Adam' in v.name]
        #init = tf.variables_initializer(optimizer_vars)
        init = tf.compat.v1.initialize_all_variables()
        self.sess.run(init)

        return train_op

    def create_model(self, n_dim, r):
        raise NotImplementedError()

    def create_objective(self, X, Y, opt_params):
        # load model output and true output
        P = self.predictions

        # compute l2 loss
        sqrt_l2_loss = tf.sqrt(tf.reduce_mean(input_tensor=(P-Y)**2 + 1e-6, axis=[1, 2]))
        sqrn_l2_norm = tf.sqrt(tf.reduce_mean(input_tensor=Y**2, axis=[1, 2]))
        snr = 20 * tf.math.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / tf.math.log(10.)

        avg_sqrt_l2_loss = tf.reduce_mean(input_tensor=sqrt_l2_loss, axis=0)
        avg_snr = tf.reduce_mean(input_tensor=snr, axis=0)

        # track losses
        tf.compat.v1.summary.scalar('l2_loss', avg_sqrt_l2_loss)
        tf.compat.v1.summary.scalar('snr', avg_snr)

        # save losses into collection
        tf.compat.v1.add_to_collection('losses', avg_sqrt_l2_loss)
        tf.compat.v1.add_to_collection('losses', avg_snr)

        # save predicted and real outputs to collection
        y_flat = tf.reshape(Y, [-1])
        p_flat = tf.reshape(P, [-1])
        tf.compat.v1.add_to_collection('hrs', y_flat)
        tf.compat.v1.add_to_collection('hrs', p_flat)

        return avg_sqrt_l2_loss

    def get_params(self):
        return [v for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
                if 'soundnet' not in v.name]

    def create_optimzier(self, opt_params):
        if opt_params['alg'] == 'adam':
            lr, b1, b2 = opt_params['lr'], opt_params['b1'], opt_params['b2']
            optimizer = tf.compat.v1.train.AdamOptimizer(lr, b1, b2)
        else:
            raise ValueError('Invalid optimizer: ' + opt_params['alg'])

        return optimizer

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
        train_op = self.optimizer.apply_gradients(
            gv, global_step=self.global_step)

        return train_op

    def load(self, ckpt):
        # get checkpoint name
        if os.path.isdir(ckpt):
            checkpoint = tf.train.latest_checkpoint(ckpt)
        else:
            checkpoint = ckpt
        meta = checkpoint + '.meta'

        # load graph
        self.saver = tf.compat.v1.train.import_meta_graph(meta)
        g = tf.compat.v1.get_default_graph()

        # load weights
        self.saver.restore(self.sess, checkpoint)

        # get graph tensors
        X, Y, alpha = tf.compat.v1.get_collection('inputs')

        # save tensors as instance variables
        self.inputs = X, Y, alpha
        self.predictions = tf.compat.v1.get_collection('preds')[0]

        # load existing loss, or erase it, if creating new one
        g.clear_collection('losses')

        # or, get existing train op:
        self.train_op = tf.compat.v1.get_collection('train_op')

    def calc_snr(self, Y, Pred):
        sqrt_l2_loss = np.sqrt(np.mean((Pred-Y)**2+1e-6, axis=(0, 1)))
        sqrn_l2_norm = np.sqrt(np.mean(Y**2, axis=(0, 1)))
        snr = 20 * np.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)
        return snr

    def calc_snr2(self, Y, P):
        sqrt_l2_loss = np.sqrt(np.mean((P-Y)**2 + 1e-6, axis=(1, 2)))
        sqrn_l2_norm = np.sqrt(np.mean(Y**2, axis=(1, 2)))
        snr = 20 * np.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)
        avg_snr = np.mean(snr, axis=0)
        return avg_snr

    def fit(self, X_train, Y_train, X_val, Y_val, n_epoch=100, r=4, speaker="single", grocery="false", piano="false", calc_full_snr=False):
        # initialize log directory
        if tf.io.gfile.exists(self.logdir):
            tf.io.gfile.rmtree(self.logdir)
        tf.io.gfile.makedirs(self.logdir)

        # load some training params
        n_batch = self.opt_params['batch_size']

        # create saver
        self.saver = tf.compat.v1.train.Saver()

        # summarization
        summary = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter(self.logdir, self.sess.graph)

        # load data into DataSet
        train_data = DataSet(X_train, Y_train)
        val_data = DataSet(X_val, Y_val)

        # init np array to store results
        results = np.empty([n_epoch, 6])
        # train the model
        epoch_start_time = time.time()
        total_start_time = time.time()
        step, epoch = 0, train_data.epochs_completed

        print(("Parameters: " + str(count_parameters())))

        while train_data.epochs_completed < n_epoch:

            step += 1

            # load the batch
            alpha = 1.0
            batch = train_data.next_batch(n_batch)
            feed_dict = self.load_batch(batch, alpha)

            # take training step
            tr_objective = self.train(feed_dict)

            # log results at the end of each epoch
            if train_data.epochs_completed > epoch:
                epoch = train_data.epochs_completed
                end_time = time.time()

                tr_l2_loss, tr_l2_snr, tr_lsd = self.eval_err(
                    X_train, Y_train, n_batch=n_batch)
                va_l2_loss, va_l2_snr, va_lsd = self.eval_err(
                    X_val, Y_val, n_batch=n_batch)

                print("Epoch {} of {} took {:.3f}s ({} minibatches)".format(
                    epoch, n_epoch, end_time - epoch_start_time, len(X_train) // n_batch))
                print("  training l2_loss/segsnr/LSD:\t\t{:.6f}\t{:.6f}\t{:.6f}".format(
                    tr_l2_loss, tr_l2_snr, tr_lsd))
                print("  validation l2_loss/segsnr/LSD:\t\t{:.6f}\t{:.6f}\t{:.6f}".format(
                    va_l2_loss, va_l2_snr, va_lsd))

                # compute summaries for overall loss
                objectives_summary = tf.compat.v1.Summary()
                objectives_summary.value.add(
                    tag='tr_l2_loss', simple_value=tr_l2_loss)
                objectives_summary.value.add(
                    tag='tr_l2_snr', simple_value=tr_l2_snr)
                objectives_summary.value.add(
                    tag='va_l2_snr', simple_value=va_l2_snr)
                objectives_summary.value.add(tag='tr_lsd', simple_value=tr_lsd)
                objectives_summary.value.add(tag='va_lsd', simple_value=va_lsd)

                # compute summaries for all other metrics
                summary_str = self.sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(objectives_summary, step)

                # write summaries and checkpoints
                summary_writer.flush()
                self.saver.save(
                    self.sess, self.checkpoint_root, global_step=step)

                # calcuate the full snr (currenty on each epoch)
                full_snr = 0
                if(calc_full_snr and train_data.epochs_completed % 1 == 0 and grocery == 'false'):
                    s1 = ""
                    s2 = ""
                    if piano == "true":
                        s1 = "../piano/interp/full-"
                        s2 = "-piano-interp-val." + \
                            str(r) + '.16000.-1.4096.0.1'
                    elif speaker == "single":
                        s1 = "../data/vctk/speaker1/full-"
                        s2 = "-vctk-speaker1-val." + str(r) + '.16000.-1.4096'
                    elif speaker == "multi":
                        s1 = "../data/vctk/multispeaker/full-"
                        s2 = "-vctk-multispeaker-interp-val." + \
                            str(r) + '.16000.-1.8192.0.25'
                    full_clips_X = pickle.load(open(s1 + 'data' + s2, 'rb'))
                    full_clips_Y = pickle.load(open(s1 + 'label' + s2, 'rb'))

                    runs = 0

                    for X, Y in zip(full_clips_X, full_clips_Y):
                        X = np.reshape(X, (1, X.shape[0], 1))
                        Y = np.reshape(Y, (1, Y.shape[0], 1))

                        if self.__class__.__name__ == 'DNN':
                            X = X[:, :8192*(X.shape[1]/8192), :]
                            Y = Y[:, :8192*(Y.shape[1]/8192), :]

                        __, snr, __ = self.eval_err(X, Y, 1)
                        full_snr += snr

    def train(self, feed_dict):
        _, loss = self.sess.run(
            [self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def load_batch(self, batch, alpha=1, train=True):
        X_in, Y_in, alpha_in = self.inputs

        X, Y = batch
        if Y is not None:
            feed_dict = {X_in: X, Y_in: Y, alpha_in: alpha}
        else:
            feed_dict = {X_in: X, alpha_in: alpha}
        # this is ugly, but only way I found to get this var after model reload
        g = tf.compat.v1.get_default_graph()
        k_tensors = [n for n in g.as_graph_def(
        ).node if 'keras_learning_phase' in n.name]
        #breakpoint()
        #assert len(k_tensors) <= 1
        if k_tensors:
            k_learning_phase = g.get_tensor_by_name(k_tensors[0].name + ':0')
            feed_dict[k_learning_phase] = train

        return feed_dict

    def eval_err(self, X, Y, n_batch=128):
        batch_iterator = iterate_minibatches(X, Y, n_batch, shuffle=True)
        l2_loss_op, l2_snr_op = tf.compat.v1.get_collection('losses')
        y_flat, p_flat = tf.compat.v1.get_collection('hrs')

        l2_loss, snr = 0, 0
        tot_l2_loss, tot_snr = 0, 0
        Ys = np.empty([0, 0])
        Preds = np.empty([0, 0])
        d = []
        l = []
        i = 0
        for bn, batch in enumerate(batch_iterator):
            feed_dict = self.load_batch(batch, train=False)
            l2_loss, l2_snr, Y, P = self.sess.run(
                [l2_loss_op, l2_snr_op, y_flat, p_flat], feed_dict=feed_dict)

            tot_l2_loss += l2_loss
            tot_snr += l2_snr
            Ys = np.append(Ys, Y)
            Preds = np.append(Preds, P)
            if(i < 10):
                i += 1
                d.append(P)
                l.append(Y)

        # calculate lsd
        lsd = self.compute_log_distortion(Ys, Preds)

        return tot_l2_loss / (bn+1), tot_snr / (bn+1), lsd

    def predict(self, X):
        raise NotImplementedError()

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
            var_params *= dim
        total_parameters += var_params
    return total_parameters
