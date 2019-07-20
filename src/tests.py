'''
This file is for runs tests.
'''

import sys, os
from subprocess import call
import shlex


sizes = ['8192']
models = ['audiohybrid2']
rs =['4']
pool_size = '8'
pool_stride = '8'


for model in models:
    for size in sizes:
        for r in rs:
	    stride = str(int(size)/2)
            #trainstr = '../piano/interp/piano-interp-train.' +r+'.16000.'+size+'.'+stride+'.0.1.h5'
            #valstr = '../piano/interp/piano-interp-val.' +r+'.16000.'+size+'.'+stride+'.0.1.h5'
            trainstr = '../data/vctk/speaker1/vctk-speaker1-train.'+r+'.16000.' + size + '.' + stride + '.h5'
            valstr = '../data/vctk/speaker1/vctk-speaker1-val.'+r+'.16000.' + size + '.' + stride + '.h5'
            command = 'python run.py train --train ' + trainstr + ' --val ' + valstr + ' -e 50 --batch-size 8 --lr 3e-4 --layers 4 --piano false  --logname full_end_normalized_parmas_ps'+pool_size+'.s'+pool_stride+'-'+model+'.lr0.00300.1.g4.b32.d'+size+'.r'+r+' --model ' + model + ' --r ' + r + ' --pool_size ' + pool_size + ' --strides ' + pool_stride 
            print(command)
            call(shlex.split(command))

'''
for model in models:
    for size in sizes:
        for r in rs:
            if(r != '4' and size != '8192'):
                continue
            stride = size
            trainstr = '../data/vctk/multispeaker/vctk-multispeaker-interp-train.'+r+'.16000.' + size + '.' + stride + '.0.25.h5'
            valstr = '../data/vctk/multispeaker/vctk-multispeaker-interp-val.'+r+'.16000.' + size + '.' + stride + '.0.25.h5'
            command = 'python run.py train --train ' + trainstr + ' --val ' + valstr + ' -e 10 --batch-size 32 --lr 3e-4 --logname new-final-normalized_params-multispeaker_ps'+pool_size+'.s'+pool_stride+'-'+model+'.lr0.00300.1.g4.b32.d'+size+'.r'+r+' --model ' + model + ' --speaker multi  --pool_size ' + pool_size + ' --strides ' + pool_stride 
            print(command)
            call(shlex.split(command))

models = ['audiohybrid2']
ps = ['0.2']
for model in models:
    for p in ps:
        command = 'python run.py train --train ' + p + ' --val ' + "n/a" + ' -e 50 --batch-size 32 --lr 3e-4 --logname last-final-grocery_'+model+'.' + p + ' --model ' + model + ' --grocery true'
        print(command)
        call(shlex.split(command))
'''
