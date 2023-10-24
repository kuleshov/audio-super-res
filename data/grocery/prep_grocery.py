"""
Create an HDF5 file of patches for training super-resolution model.
"""

import os, argparse
import numpy as np
import h5py
import pickle
import csv
from tqdm import tqdm
import pprint

import librosa
from scipy import interpolate
from scipy.signal import decimate
from scipy.signal import butter, lfilter
import re

f = 'grocery/train.csv'

TRAIN_PROB = 0.8
NUM_DATES = 1024
MASk_PROB = 0.1

# read data into a dictionary of items -> (date -> [list of counts]
items = {}
with open(f, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader, None)
    i =0
    for row in tqdm(reader):
        i +=1
        item = row[3]
        sales = row[4]
        date = row[1]
        if(date == '2015-12-03'): break # set this to a later date to capture more data 
                                        # note: also change row length below
        if(item not in items):
            items[item] = {}
        if(date not in items[item]):
            items[item][date] = [];
        items[item][date].append(float(sales))

# avergae per count per date per item data and create tensor with
# row = items, columns = date, and value = count
data = []
for vals in tqdm(list(items.values())):
    row = []
    for sales in tqdm(list(vals.values())):
        row.append(np.average(np.array(sales)))
    if(len(row) >= NUM_DATES): # cut off extra dates to keep size constant 
                          # note: change this to change size of processed date
        data.append(row[:NUM_DATES])
data = np.stack(data)
pprint.pprint(data.shape)

# split into train and test sets
trainY = data[:int(data.shape[0]*TRAIN_PROB),]
testY = data[int(data.shape[0]*TRAIN_PROB):,:]

# mask out some of the data
trainX = np.empty_like(trainY)
trainX[:] = trainY
testX = np.empty_like(testY)
testX[:] = testY
trainMask = np.random.choice([0,1],size=trainX.shape, p=[MASK_PROB, 1-MASK_PROB])
trainX = np.multiply(trainX, trainMask)
testMask = np.random.choice([0,1],size=testX.shape, p=[MASK_PROB, 1-MASK_PROB])
testX = np.multiply(testX, testMask)


# pickle the data
print(trainX.shape)
print(trainY.shape)
pickle.dump(testX, open('grocery/grocery-test-data_'+str(MASK_PROB),'w'))
pickle.dump(testY, open('grocery/grocery-test-label'+str(MASK_PROB),'w'))
pickle.dump(trainX, open('grocery/grocery-train-data'+str(MASK_PROB),'w'))
pickle.dump(trainY, open('grocery/grocery-train-label'+str(MASK_PROB),'w'))
