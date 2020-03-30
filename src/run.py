#!/bin/sh

import os
os.sys.path.append(os.path.abspath('.'))
os.sys.path.append(os.path.dirname(os.path.abspath('.')))

import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
import cPickle

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

  train_parser.add_argument('--model', default='audiounet',
    choices=('audiounet', 'audiotfilm', 'dnn', 'spline'),
    help='model to train')
  train_parser.add_argument('--train', required=True,
    help='path to h5 archive of training patches')
  train_parser.add_argument('--val', required=True,
    help='path to h5 archive of validation set patches')
  train_parser.add_argument('-e', '--epochs', type=int, default=100,
    help='number of epochs to train')
  train_parser.add_argument('--batch-size', type=int, default=128,
    help='training batch size')
  train_parser.add_argument('--logname', default='tmp-run',
    help='folder where logs will be stored')
  train_parser.add_argument('--layers', default=4, type=int,
    help='number of layers in each of the D and U halves of the network')
  train_parser.add_argument('--alg', default='adam',
    help='optimization algorithm')
  train_parser.add_argument('--lr', default=1e-3, type=float,
    help='learning rate')
  train_parser.add_argument('--r', type=int, default=4, help='upscaling factor')
  train_parser.add_argument('--speaker', default='single', choices=('single', 'multi'), 
    help='number of speakers being trained on')
  train_parser.add_argument('--piano', default='false', choices=('true', 'false'))
  train_parser.add_argument('--grocery', default='false', choices=('true', 'false'))
  train_parser.add_argument('--pool_size', type=int, default=4, help='size of pooling window')
  train_parser.add_argument('--strides', type=int, default=4, help='pooling stide')
  train_parser.add_argument('--full', default='false', choices=('true', 'false'))

  # eval
  eval_parser = subparsers.add_parser('eval')
  eval_parser.set_defaults(func=eval)

  eval_parser.add_argument('--logname', required=True,
    help='path to training checkpoint')
  eval_parser.add_argument('--out-label', default='',
    help='append label to output samples')
  eval_parser.add_argument('--wav-file-list', 
    help='list of audio files for evaluation')
  eval_parser.add_argument('--r', help='upscaling factor', default = 4, type=int)
  eval_parser.add_argument('--sr', help='high-res sampling rate', 
                                   type=int, default=16000)
  eval_parser.add_argument('--grocery', default='false', choices=('true', 'false'))
  eval_parser.add_argument('--model', default='audiounet',
    choices=('audiounet', 'audiotfilm', 'dnn', 'spline'),
    help='model to train')
  eval_parser.add_argument('--speaker', default='single', choices=('single', 'multi'), 
    help='number of speakers being trained on')
  eval_parser.add_argument('--pool_size', type=int, default=4, help='size of pooling window')
  eval_parser.add_argument('--strides', type=int, default=4, help='pooling stide')
  eval_parser.add_argument('--patch_size', type=int, default=8192, help='Size of patches over which the model operates')
  return parser

# ----------------------------------------------------------------------------

def train(args):
  full = True if args.full == 'true' else False
    
  # get data
  if(args.grocery == 'false'):
    X_train, Y_train = load_h5(args.train)
    X_val, Y_val = load_h5(args.val)
  else:
    X_train = cPickle.load(open("../data/grocery/grocery/grocery-train-data" + args.train))
    Y_train = cPickle.load(open("../data/grocery/grocery/grocery-train-label" + args.train))
    X_val = cPickle.load(open("../data/grocery/grocery/grocery-test-data_" + args.train))
    Y_val = cPickle.load(open("../data/grocery/grocery/grocery-test-label" + args.train))
    X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
    Y_train = np.reshape(Y_train, [Y_train.shape[0], Y_train.shape[1], 1])
    X_val = np.reshape(X_val, [X_val.shape[0], X_val.shape[1], 1])
    Y_val = np.reshape(Y_val, [Y_val.shape[0], Y_val.shape[1], 1])
 
  # reshape piano data
  if args.piano == 'true':
      X_train = np.reshape(X_train, [X_train.shape[0], X_val.shape[2], X_val.shape[1]])
      Y_train = np.reshape(Y_train, [Y_train.shape[0], Y_val.shape[2], Y_val.shape[1]])
      X_val = np.reshape(X_val, [X_val.shape[0], X_val.shape[2], X_val.shape[1]])
      Y_val = np.reshape(Y_val, [Y_val.shape[0], Y_val.shape[2], Y_val.shape[1]])

  # determine super-resolution level
  n_dim, n_chan = Y_train[0].shape
  r = Y_train[0].shape[1] / X_train[0].shape[1]
  assert n_chan == 1

  # Train seq2seq model
  if(args.model == 'seq2seq'):
    model = models.Model2()
    model.run(X_train, Y_train, X_val, Y_val, n_epoch=args.epochs, r=args.r, speaker=args.speaker, grocery=args.grocery)

  else:
    # create model
    model = get_model(args, n_dim, r, from_ckpt=False, train=True)
    # train model
    model.fit(X_train, Y_train, X_val, Y_val, n_epoch=args.epochs, r=args.r, speaker=args.speaker, grocery=args.grocery, piano=args.piano, calc_full_snr = full)

def eval(args):
  # load model
  model = get_model(args, 0, args.r, from_ckpt=True, train=False, grocery=args.grocery)
  model.load(args.logname) # from default checkpoint

  if args.wav_file_list:
    with open(args.wav_file_list) as f:
      for line in f:
        try:
          print(line.strip())
          if(args.speaker == 'single'):
            upsample_wav('../data/vctk/VCTK-Corpus/wav48/p225/'+line.strip(), args, model)
          else:
            upsample_wav('../data/vctk/VCTK-Corpus/'+line.strip(), args, model)
        except EOFError:
          print 'WARNING: Error reading file:', line.strip()

def get_model(args, n_dim, r, from_ckpt=False, train=True, grocery='false'):
  """Create a model based on arguments"""  
  if train:
    opt_params = { 'alg' : args.alg, 'lr' : args.lr, 'b1' : 0.9, 'b2' : 0.999,
                   'batch_size': args.batch_size, 'layers': args.layers }
  else: 
    opt_params = default_opt

  print(args.model)
  # create model
  if args.model == 'audiounet':
    model = models.AudioUNet(from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                                 opt_params=opt_params, log_prefix=args.logname)
  elif args.model == 'audiotfilm':
    model = models.AudioTfilm(from_ckpt=from_ckpt, n_dim=n_dim, r=r, pool_size = args.pool_size, 
                                strides=args.strides, opt_params=opt_params, log_prefix=args.logname)  
  elif args.model == 'dnn':
    model = models.DNN(from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                                 opt_params=opt_params, log_prefix=args.logname)
  elif args.model == 'spline':
   model = models.Spline(from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                                 opt_params=opt_params, log_prefix=args.logname)
  else:
    raise ValueError('Invalid model')
  return model

def main():
  parser = make_parser()
  args = parser.parse_args()
  args.func(args)

if __name__ == '__main__':
  main()
