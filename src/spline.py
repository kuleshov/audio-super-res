import os
import numpy as np
import pickle
import librosa
from models.io import load_h5
import argparse
import pickle
from scipy import interpolate

def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="Commands")
    train_parser=subparsers.add_parser("spline")
    train_parser.set_defaults(func=spline)
    train_parser.add_argument("--val", required=True,
            help="path to h5 archive of validation set patches")
    train_parser.add_argument("--grocery", default='false', choices=('true', 'false'))
    train_parser.add_argument("--piano", default='false', choices=('true', 'false'))
    return parser

def get_power(x):
    S = librosa.stft(x, 2048)
    p = np.angle(S)
    S = np.log(np.abs(S)**2 + 1e-8)
    return S

def compute_log_distortion(x_hr, x_pr):
    S1 = get_power(x_hr)
    S2 = get_power(x_pr)
    lsd = np.mean(np.sqrt(np.mean((S1-S2)**2 + 1e-8, axis=1)), axis=0)
    return min(lsd, 10.)


def upsample(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)
    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    return x_sp

def spline(args):
    
    # Load data
    if(args.grocery == 'false'):
        X_val, Y_val = load_h5(args.val)
    else:
        X_val = pickle.load(open("../data/grocery/grocery/grocery-test-data_" + args.val))
        Y_val = pickle.load(open("../data/grocery/grocery/grocery-test-label" + args.val))
        for i in range(len(X_val)):
            urow = upsample(X_val[i,:], 1)
            X_val[i,:] = urow

    for (t, P, Y) in [("val", X_val, Y_val)]:
        if(args.piano=='true'):
            axes =(1,2)
        else:
            axes = 1
        sqrt_l2_loss =np.sqrt(np.mean((P-Y)**2 + 1e-6, axis=axes))
        sqrn_l2_norm = np.sqrt(np.mean(Y**2, axis=axes))
        snr = 20 * np.log(sqrn_l2_norm/sqrt_l2_loss + 1e-8) / np.log(10.)
        avg_snr = np.mean(snr, axis=0)
        lsd = compute_log_distortion(np.reshape(Y, (-1)), np.reshape(P, (-1)))
        avg_sqrt_l2_loss = np.mean(sqrt_l2_loss, axis=0)

        print(t + " l2 loss: " + str(avg_sqrt_l2_loss))
        print(t + " average SNR: " + str(avg_snr))
        print(t + " lsd: " + str(lsd))


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)
