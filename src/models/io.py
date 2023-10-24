import os
import numpy as np
import h5py
import librosa
import soundfile as sf

from scipy.signal import decimate

from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------

def load_h5(h5_path):
  # load training data
  with h5py.File(h5_path, 'r') as hf:
    print('List of arrays in input file:', list(hf.keys()))
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))
    print('Shape of X:', X.shape)
    print('Shape of Y:', Y.shape)

  return X, Y

def upsample_wav(wav, args, model):

  # load signal
  x_hr, fs = librosa.load(wav, sr=args.sr)

  x_lr_t = decimate(x_hr, args.r)

  # pad to mutliple of patch size to ensure model runs over entire sample
  x_hr = np.pad(x_hr, (0, args.patch_size - (x_hr.shape[0] % args.patch_size)), 'constant', constant_values=(0,0))

  # downscale signal
  x_lr = decimate(x_hr, args.r)

  # upscale the low-res version
  P = model.predict(x_lr.reshape((1,len(x_lr),1)))
  x_pr = P.flatten()

  # crop so that it works with scaling ratio
  x_hr = x_hr[:len(x_pr)]
  x_lr_t = x_lr_t[:len(x_pr)]

  # save the file
  outname = wav + '.' + args.out_label

  #librosa.write_wav(outname + '.lr.wav', x_lr_t, fs / args.r)
  #librosa.write_wav(outname + '.hr.wav', x_hr, fs)
  #librosa.write_wav(outname + '.pr.wav', x_pr, fs)

  sf.write(outname + '.lr.wav', x_lr_t, int(fs / args.r))
  sf.write(outname + '.hr.wav', x_hr, int(fs))
  sf.write(outname + '.pr.wav', x_pr, int(fs))

  # save the spectrum
  S = get_spectrum(x_pr, n_fft=int(2048))
  save_spectrum(S, outfile=outname + '.pr.png')
  S = get_spectrum(x_hr, n_fft=int(2048))
  save_spectrum(S, outfile=outname + '.hr.png')
  S = get_spectrum(x_lr, n_fft=int(2048/args.r))
  save_spectrum(S, outfile=outname + '.lr.png')

def upsample_bad_wav(wav, args, model):

  # load signal
  # x_hr, fs = librosa.load(wav, sr=args.sr)
  x_lr, fs = librosa.load(wav, sr=args.sr)

  #x_lr_t = decimate(x_hr, args.r)
  x_lr_t = x_lr
  #print('Patch Size Mod:', x_lr.shape[0])

  # pad to mutliple of patch size to ensure model runs over entire sample
  # x_hr = np.pad(x_hr, (0, args.patch_size - (x_hr.shape[0] % args.patch_size)), 'constant', constant_values=(0,0))
  x_lr = np.pad(x_lr, (0, args.patch_size - (x_lr.shape[0] % args.patch_size)), 'constant', constant_values=(0,0))

  # downscale signal
  # x_lr = decimate(x_hr, args.r)

  # upscale the low-res version
  P = model.predict(x_lr.reshape((1,len(x_lr),1)))
  x_pr = P.flatten()

  # crop so that it works with scaling ratio
  # x_hr = x_hr[:len(x_pr)]
  x_lr_t = x_lr_t[:len(x_pr)]

  # save the file
  outname = wav + '.' + args.out_label
  # Get rid of "output" for next 3 lines
  # librosa.write_wav(outname + '.lr.wav', x_lr_t, fs / args.r)
  # librosa.write_wav(outname + '.hr.wav', x_hr, fs)
  # librosa.write_wav(outname + '.pr.wav', x_pr, fs)
  # print('Sample rate:', fs)
  # print('Upsample Rate:', fs / args.r)

  sf.write(outname + '.lr.wav', x_lr_t, int(fs))
  # sf.write(outname + '.hr.wav', x_hr, fs)
  sf.write(outname + '.pr.wav', x_pr, int(fs * args.r))


  # save the spectrum
  S = get_spectrum(x_pr, n_fft=int(2048*args.r))
  save_spectrum(S, outfile=outname + '.pr.png')
  # S = get_spectrum(x_hr, n_fft=2048)
  # save_spectrum(S, outfile=outname + '.hr.png')
  S = get_spectrum(x_lr, n_fft=int(2048))
  save_spectrum(S, outfile=outname + '.lr.png')

# ----------------------------------------------------------------------------

def get_spectrum(x, n_fft=2048):
  S = librosa.stft(x, n_fft)
  p = np.angle(S)
  S = np.log1p(np.abs(S))
  return S

def save_spectrum(S, lim=1000, outfile='spectrogram.png'):
  plt.imshow(S, aspect='auto')
  plt.gca().invert_yaxis()
  plt.ylim([0,lim])
  plt.xlabel("Audio Frame")
  plt.ylabel("Frequency (Hz)")
  plt.tight_layout()
  plt.savefig(outfile)
