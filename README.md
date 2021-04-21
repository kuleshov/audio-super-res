Audio Super Resolution Using Neural Networks
============================================

This repository implements the audio super-resolution model proposed in:

```
S. Birnbaum, V. Kuleshov, Z. Enam, P. W.. Koh, and S. Ermon. Temporal FiLM: Capturing Long-Range Sequence Dependencies with Feature-Wise Modulations. NeurIPS 2019
V. Kuleshov, Z. Enam, and S. Ermon. Audio Super Resolution Using Neural Networks. ICLR 2017 (Workshop track)
```

## Installation

### Requirements

The model is implemented in Python 3.7.10 and uses several additional libraries.

* `tensorflow==2.4.1`
* `keras==2.4.0`
* `numpy==1.19.5`
* `scipy==1.6.0`
* `librosa==0.8.3`
* `h5py==2.10.0`
* `matplotlib==3.3.4`

A full list of the packages on our enviornment is in `requirements.txt`

### Setup

To install this package, simply clone the git repo:

```
git clone https://github.com/kuleshov/audio-super-res.git;
cd audio-super-res;
```

## Running the model

### Contents

The repository is structured as follows.

* `./src`: model source code
* `./data`: code to download the model data

### Retrieving data

The `./data` subfolder contains code for preparing the VCTK speech dataset.
Make sure you have enough disk space and bandwidth (the dataset is over 18G, uncompressed).
You need to type:

```
cd ./data/vctk;
make;
```

Next, you must prepare the dataset for training:
you will need to create pairs of high and low resolution sound patches (typically, about 0.5s in length).
We have included a script called `prep_vctk.py` that does that, which works as follows.

```
usage: prep_vctk.py [-h] [--file-list FILE_LIST] [--in-dir IN_DIR] [--out OUT]
                    [--scale SCALE] [--dimension DIMENSION] [--stride STRIDE]
                    [--interpolate] [--low-pass] [--batch-size BATCH_SIZE]
                    [--sr SR] [--sam SAM]

optional arguments:
  -h, --help            show this help message and exit
  --file-list FILE_LIST
                        list of input wav files to process
  --in-dir IN_DIR       folder where input files are located
  --out OUT             path to output h5 archive
  --scale SCALE         scaling factor
  --dimension DIMENSION
                        dimension of patches (use -1 for no patching)
  --stride STRIDE       stride when extracting patches
  --interpolate         interpolate low-res patches with cubic splines
  --low-pass            apply low-pass filter when generating low-res patches
  --batch-size BATCH_SIZE
                        we produce # of patches that is a multiple of batch
                        size
  --sr SR               audio sampling rate  
  --sam SAM             subsampling factor for the data (only applicable for multispeaker data)
```

The output of the data preparation step are two `.h5` archives containing, respectively, the training and validation pairs of high/low resolution sound patches.
You can also generate these by running `make` in the corresponding directory, e.g.
```
cd ./speaker1;
make;
```

This will use a set of default parameters.

To generate the files needed for the training example below, run the following from the `speaker1` directory:
```
python ../prep_vctk.py \
  --file-list  speaker1-train-files.txt \
  --in-dir ../VCTK-Corpus/wav48/p225 \
  --out vctk-speaker1-train.4.16000.8192.4096.h5 \
  --scale 4 \
  --sr 16000 \
  --dimension 8192 \
  --stride 4096 \
  --interpolate \
  --low-pass

python ../prep_vctk.py \
  --file-list speaker1-val-files.txt \
  --in-dir ../VCTK-Corpus/wav48/p225 \
  --out vctk-speaker1-val.4.16000.8192.4096.h5.tmp \
  --scale 4 \
  --sr 16000 \
  --dimension 8192 \
  --stride 4096 \
  --interpolate \
  --low-pass

python ../prep_vctk.py \
  --file-list  speaker1-train-files.txt \
  --in-dir ../VCTK-Corpus/wav48/p225 \
  --out vctk-speaker1-train.4.16000.-1.4096.h5 \
  --scale 4 \
  --sr 16000 \
  --dimension -1 \
  --stride 4096 \
  --interpolate \
  --low-pass

python ../prep_vctk.py \
  --file-list speaker1-val-files.txt \
  --in-dir ../VCTK-Corpus/wav48/p225 \
  --out vctk-speaker1-val.4.16000.-1.4096.h5.tmp \
  --scale 4 \
  --sr 16000 \
  --dimension -1 \
  --stride 4096 \
  --interpolate \
  --low-pass
```

### Audio super resolution tasks

We have included code to prepare two datasets.

* The single-speaker dataset consists only of VCTK speaker #1; it is relatively quick to train a model (a few hours).
* The multi-speaker dataset uses the last 8 VCTK speakers for evaluation, and the rest for training; it takes several days to train the model, and several hours to prepare the data.

We suggest starting with the single-speaker dataset.

### Training the model

Running the model is handled by the `src/run.py` script.

```
usage: run.py train [-h] --train TRAIN --val VAL [-e EPOCHS]
                    [--batch-size BATCH_SIZE] [--logname LOGNAME]
                    [--layers LAYERS] [--alg ALG] [--lr LR] [--model MODEL] 
                    [--r R] [--piano PIANO] [--grocery GROCERY]

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         path to h5 archive of training patches
  --val VAL             path to h5 archive of validation set patches
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train
  --batch-size BATCH_SIZE
                        training batch size
  --logname LOGNAME     folder where logs will be stored
  --layers LAYERS       number of layers in each of the D and U halves of the
                        network
  --alg ALG             optimization algorithm
  --lr LR               learning rate
  --model               the model to use for training (audiounet, audiotfilm, 
                                                       dnn, or spline). Defaults to audiounet.
  --r                   the upscaling ratio of the data: make sure that the appropriate 
                        datafile have been generated (note: to generate data with different
                        scaling ratios change the SCA parameter in the makefile)
  --piano               false by default--make true to train on piano data 
  --grocery             false by default--make true to train on grocery imputation data
  --speaker              number of speakers being trained on (single or multi). Defaults to single
  --pools_size          size of pooling window
  --strides             size of pooling strides
  --full                false by default--whether to calculate the "full" snr after each epoch. The "full" snr 
                        is the snr acorss the non-patched data file, rather than the average snr over all the 
                        patches which is calculated by default
```
Note: to generate the data needed for the grocery imputation experiment, download train.csv.7z from 
https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data into the data/grocery/grocery directory, 
unzip the csv, and run prep_grocery.py from the data/grocery directory.

For example, to run the model on data prepared for the single speaker dataset, you would type:

```
python run.py train \
  --train ../data/vctk/speaker1/vctk-speaker1-train.4.16000.8192.4096.h5 \
  --val ../data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5 \
  -e 120 \
  --batch-size 64 \
  --lr 3e-4 \
  --logname singlespeaker \
  --model audiotfilm \
  --r 4 \
  --layers 4 \
  --piano false \
  --pool_size 2 \
  --strides 2
  --full true
```

The above run will store checkpoints in `./singlespeaker.lr0.000300.1.g4.b64`.

Note on the models: audiotfilm is the best model.

#### Pre-Trained Model
See the below link for a pre-trained single-speaker model. This model was trained with the following parameters:
```
python run.py train \
  --train ../data/vctk/speaker1/vctk-speaker1-train.4.16000.8192.4096.h5 \
  --val ../data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5 \
  -e 2   --batch-size 16  --lr 3e-4   --logname singlespeaker \
  --model audiotfilm   --r 4   --layers 4   --piano false \   
  --pool_size 2   --strides 2
```

https://drive.google.com/file/d/1pqIaxtZpt9GRc-Yp1zCzVoSbQFSLnERF/view?usp=sharing

To use the model, unzip the file in the `src` directory and run eval with the logname corresponding to the checkpoint file.

### Testing the model

The `run.py` command may be also used to evaluate the model on new audio samples.

```
usage: run.py eval [-h] --logname LOGNAME [--out-label OUT_LABEL]
                   [--wav-file-list WAV_FILE_LIST] [--r R] [--sr SR]

optional arguments:
  -h, --help            show this help message and exit
  --logname LOGNAME     path to training checkpoint
  --out-label OUT_LABEL
                        append label to output samples
  --wav-file-list WAV_FILE_LIST
                        list of audio files for evaluation
  --r R                 upscaling factor
  --sr SR               high-res sampling rate
```

In the above example, we would type:

```
python run.py eval \
  --logname ./singlespeaker.lr0.000300.1.g4.b64/model.ckpt-20101 \
  --out-label singlespeaker-out \
  --wav-file-list ../data/vctk/speaker1/speaker1-val-files.txt \
  --r 4 \
  --pool_size 2 \
  --strides 2 \
  --model audiotfilm
```

This will look at each file specified via the `--wav-file-list` argument (these must be high-resolution samples),
and create for each file `f.wav` three audio samples:

* `f.singlespeaker-out.hr.wav`: the high resolution version
* `f.singlespeaker-out.lr.wav`: the low resolution version processed by the model
* `f.singlespeaker-out.pr.wav`: the super-resolved version

These will be found in the same folder as `f.wav`. Because of how our model is defined, the number of samples in the input must be a multiple of `2**downscaling_layers`; if that's not the case, we will clip the input file (potentially shortening it by a fraction of a second).

**Disclaimer:** We recently upgraded the versions of many of the packages, including Keras and Tensorflow. The example workflow for training and predicting should work, but the codebase has not been fully tested. Please create an issue if you run into any errors.   

### Keras Layer
`keras_layer.py` implements the TFiLM layer as a customer Keras layer. The below code illustrates how to use this custom layer.

```
import numpy as np
from keras.layers import Input, Dense, Flatten, Lambda
from keras.models import Model
import tensorflow as tf

### Insert definition of TFiLM layer here. ####

x = np.random.random((2, 100, 50))
y = np.zeros((2))

inputs = Input(shape=(100, 50))
l = TFiLM(2)(inputs)
l = Flatten()(l)
outputs = Dense(1, activation='sigmoid')(l)


# This creates a model that includes
# the Input layer, a TFILM layer, and a dense layer
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())
model.fit(x, y, epochs=10)  # starts training
```

## Remarks

We would like to emphasize a few points.

* Machine learning algorithms are only as good as their training data. If you want to apply our method to your personal recordings, you will most likely need to collect additional labeled examples.
* You will need a very large model to fit large and diverse datasets (such as the 1M Songs Dataset)
* Interestingly, super-resolution works better on aliased input (no low-pass filter). This is not reflected well in objective benchmarks, but is noticeable when listening to the samples. For applications like compression (where you control the low-res signal), this may be important.
* More generally, the model is very sensitive to how low resolution samples are generated. Even the type of low-pass filter (Butterworth, Chebyshev) will affect performance.

### Extensions

The same architecture can be used on many time series tasks outside the audio domain. We have successfully used it to impute functional genomics data and denoise EEG recordings. Stay tuned for more updates!

## Feedback

Send feedback to [Sawyer Birnbaum](sawyerb@stanford.edu).
