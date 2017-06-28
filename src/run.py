import os
os.sys.path.append(os.path.abspath('.'))
os.sys.path.append(os.path.dirname(os.path.abspath('.')))

import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np

import models
from models.model import default_opt
from models.io import load_h5, upsample_wav

# ----------------------------------------------------------------------------

def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  # train

  train_parser = subparsers.add_parser('train')
  train_parser.set_defaults(func=train)

  train_parser.add_argument('--train', required=True)
  train_parser.add_argument('--val', required=True)
  train_parser.add_argument('-e', '--epochs', type=int, default=100)
  train_parser.add_argument('--batch-size', type=int, default=128)
  train_parser.add_argument('--load', help='preload-existing model')
  train_parser.add_argument('--logname', default='tmp-run')
  train_parser.add_argument('--layers', default=4, type=int)
  train_parser.add_argument('--alg', default='adam')
  train_parser.add_argument('--lr', default=1e-3, type=float)
  train_parser.add_argument('--sr', help='sampling rate for the wav', 
                                    type=int, default=16000)

  # eval

  eval_parser = subparsers.add_parser('eval')
  eval_parser.set_defaults(func=eval)

  eval_parser.add_argument('--logname', required=True)
  eval_parser.add_argument('--out-label', default='')
  eval_parser.add_argument('--wav-file-list', help='list of audio files')
  eval_parser.add_argument('--r', help='upscaling factor', type=int)
  eval_parser.add_argument('--sr', help='sampling rate', 
                                   type=int, default=16000)
  
  return parser

# ----------------------------------------------------------------------------

def train(args):
  # get data
  X_train, Y_train = load_h5(args.train)
  X_val, Y_val = load_h5(args.val)

  # determine super-resolution level
  n_dim, n_chan = Y_train[0].shape
  r = Y_train[0].shape[1] / X_train[0].shape[1]
  assert n_chan == 1

  # # determine super-resolution level
  # n_chan, n_dim = Y_train[0].shape
  # r = Y_train[0].shape[1] / X_train[0].shape[1]
  # assert n_chan == 1

  # # transponse to match tensorflow (batch, height, width, chan) format
  # X_train, X_val = X_train.transpose([0,2,1]), X_val.transpose([0,2,1])
  # Y_train, Y_val = Y_train.transpose([0,2,1]), Y_val.transpose([0,2,1])

  # load model
  from_ckpt = True if args.load is not None else False
  model = get_model(args, n_dim, r, from_ckpt=from_ckpt, train=True)

  # load checkpoint
  if from_ckpt: model.load(args.load)

  # train model
  model.fit(X_train, Y_train, X_val, Y_val, n_epoch=args.epochs)

def eval(args):
  # load model
  model = get_model(args, 0, args.r, from_ckpt=True, train=False)
  model.load(args.logname) # from default checkpoint

  if args.wav_file_list:
    with open(args.wav_file_list) as f:
      for line in f:
        try:
          print line.strip()
          upsample_wav(line.strip(), args, model)
        except EOFError:
          print 'WARNING: Error reading file:', line.strip()

def get_model(args, n_dim, r, from_ckpt=False, train=True):
  """Create a model based on arguments"""  
  if train:
    opt_params = { 'alg' : args.alg, 'lr' : args.lr, 'b1' : 0.9, 'b2' : 0.999,
                   'batch_size': args.batch_size, 'layers': args.layers }
  else: 
    opt_params = default_opt

  # create model
  model = models.AudioUNet(from_ckpt=from_ckpt, n_dim=n_dim, r=r, 
                               opt_params=opt_params, log_prefix=args.logname)
  return model

def main():
  parser = make_parser()
  args = parser.parse_args()
  args.func(args)

if __name__ == '__main__':
  main()