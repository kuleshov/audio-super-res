"""
Create an HDF5 file of patches for training super-resolution model.
"""

import os, argparse
import numpy as np
import h5py

import librosa
from scipy import interpolate
from scipy.signal import decimate

# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--file-list',
  help='list of input wav files to process')
parser.add_argument('--in-dir', default='~/',
  help='folder where input files are located')
parser.add_argument('--out',
  help='path to output h5 archive')
parser.add_argument('--scale', type=int, default=2,
  help='scaling factor')
parser.add_argument('--dimension', type=int, default=8192,
  help='dimension of patches')
parser.add_argument('--stride', type=int, default=3200,
  help='stride when extracting patches')
parser.add_argument('--interpolate', action='store_true', 
  help='interpolate low-res patches with cubic splines')
parser.add_argument('--low-pass', action='store_true',
  help='apply low-pass filter when generating low-res patches')
parser.add_argument('--batch-size', type=int, default=128,
  help='we produce # of patches that is a multiple of batch size')
parser.add_argument('--sr', type=int, default=16000, help='audio sampling rate')
parser.add_argument('--sam', type=float, default=1.,
  help='subsampling factor for the data')

args = parser.parse_args()

# ----------------------------------------------------------------------------

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def add_data(h5_file, inputfiles, args, save_examples=False):
  # Make a list of all files to be processed
  file_list = []
  file_extensions = set(['.wav'])
  with open(inputfiles) as f:
    for line in f:
      filename = line.strip()
      ext = os.path.splitext(filename)[1]
      if ext in file_extensions:
        file_list.append(os.path.join(args.in_dir, filename))
  num_files = len(file_list)

  # patches to extract and their size
  if args.interpolate:
    d, d_lr = args.dimension, args.dimension
    s, s_lr = args.stride, args.stride
  else:
    d, d_lr = args.dimension, args.dimension / args.scale
    s, s_lr = args.stride, args.stride / args.scale
  hr_patches, lr_patches = list(), list()

  for j, file_path in enumerate(file_list):
    if j % 10 == 0: print '%d/%d' % (j, num_files)

    # load audio file
    x, fs = librosa.load(file_path, sr=args.sr)

    # crop so that it works with scaling ratio
    x_len = len(x)
    x = x[ : x_len - (x_len % args.scale)]

    # generate low-res version
    if args.low_pass:
      # x_bp = butter_bandpass_filter(x, 0, args.sr / args.scale / 2, fs, order=6)
      # x_lr = np.array(x[0::args.scale])
      #x_lr = decimate(x, args.scale, zero_phase=True)
      x_lr = decimate(x, args.scale)
    else:
      x_lr = np.array(x[0::args.scale])
    
    if args.interpolate:
      x_lr = upsample(x_lr, args.scale)
      assert len(x) % args.scale == 0
      assert len(x_lr) == len(x)
    else:
      assert len(x) % args.scale == 0
      assert len(x_lr) == len(x) / args.scale

    # generate patches
    max_i = len(x) - d + 1
    for i in range(0, max_i, s):
      # keep only a fraction of all the patches
      u = np.random.uniform()
      if u > args.sam: continue

      if args.interpolate:
        i_lr = i
      else:
        i_lr = i / args.scale

      hr_patch = np.array( x[i : i+d] )
      lr_patch = np.array( x_lr[i_lr : i_lr+d_lr] )

      # print 'a', hr_patch
      # print 'b', lr_patch

      assert len(hr_patch) == d
      assert len(lr_patch) == d_lr

      # print hr_patch

      hr_patches.append(hr_patch.reshape((d,1)))
      lr_patches.append(lr_patch.reshape((d_lr,1)))

      # if j == 1: exit(1)

  # crop # of patches so that it's a multiple of mini-batch size
  num_patches = len(hr_patches)
  print num_patches
  num_to_keep = int(np.floor(num_patches / args.batch_size) * args.batch_size)
  hr_patches = np.array(hr_patches[:num_to_keep])
  lr_patches = np.array(lr_patches[:num_to_keep])

  if save_examples:
    librosa.output.write_wav('example-hr.wav', hr_patches[40][0], fs, norm=False)
    librosa.output.write_wav('example-lr.wav', lr_patches[40][0], fs / args.scale, norm=False)
    print hr_patches[40].shape
    print lr_patches[40].shape
    print hr_patches[40][0][:10]
    print lr_patches[40][0][:10]
    print 'two examples saved'

  print hr_patches.shape

  # create the hdf5 file
  data_set = h5_file.create_dataset('data', lr_patches.shape, np.float32)
  label_set = h5_file.create_dataset('label', hr_patches.shape, np.float32)

  data_set[...] = lr_patches
  label_set[...] = hr_patches

def upsample(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp

if __name__ == '__main__':
  # create train
  with h5py.File(args.out, 'w') as f:
    add_data(f, args.file_list, args, save_examples=False)
