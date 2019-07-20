


import numpy as np
import tensorflow as tf
import pprint
import cPickle
from tqdm import tqdm
from sklearn import(manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
from time import time
import matplotlib.pyplot as plt
from keras import backend as K
import re

row_size = 5000 
genders= {'M': 'blue', 'F': 'red'}
accents = {'English': 'blue',
           'Scottish': 'red',
           'NorthernIrish': 'green',
           'Irish': 'yellow',
           'Indian': 'purple',
           'Welsh': 'brown',
           'American': 'orange',
           'Canadian': 'black',
           'SouthAfrican': 'cyan',
           'Australian_Engl': 'magenta',
           'NewZealand': 'pink'}
"""
layers = ["generator/downsc_conv0/Reshape_2:0",
          "generator/downsc_conv1/Reshape_2:0",
          "generator/downsc_conv2/Reshape_2:0",
          "generator/downsc_conv2/Reshape_2:0",
          "generator/bottleneck_conv/Reshape_2:0",
          "generator/upsc_conv3/merge_1/concat:0",
          "generator/upsc_conv2/merge_2/concat:0",
          "generator/upsc_conv1/merge_3/concat:0",
          "generator/upsc_conv0/merge_4/concat:0",
          "generator/merge_5/add:0"]
"""
layers = ["generator/merge_5/add:0"]
layers = ["generator/upsc_conv0/lstm_9/transpose_1:0"]
layers = ["generator/upsc_conv0/merge_4/concat:0"]

print("loading...")
speakerInfo = np.loadtxt("../data/vctk/VCTK-Corpus/speaker-info.txt",
                         dtype={"names": ("ID", "AGE", "GENDER", "ACCENT", "REGION"), 
                                "formats": ('|S15', '|S15', '|S15', '|S15', '|S20')})
speakerInfoDict = {}
for i in range(1, len(speakerInfo)):
    row = speakerInfo[i] 
    speakerInfoDict[row[0]] = row
print("loaded speaker info")

ids = np.load("../data/vctk/multispeaker/ID_list") # this associates each audio sample with the id of its speaker
                                                   # i.e., the id on line 1 is the id of the sample in row 1 of the datafile

print("loaded id list")
data = cPickle.load(open("../data/vctk/multispeaker/full-data-vctk-multispeaker-interp-val.4.16000.-1.8192.0.25"))
#data = np.array([[1, 2, 3, 4, 5, 6, 7]])
print("loaded data")
##acs = []
##for i in range(0, len(data)):
#    acs.append(speakerInfoDict[str(ids[i])][3])
##from collections import Counter
##print Counter(acs)

##x = 1/0
maps = {}
used_ids = []
with tf.Session() as sess:
    # setup session
    gpu_options=tf.GPUOptions(allow_growth=True)
   
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    K.set_session(sess)  
    # load model
    saver =tf.train.import_meta_graph('./full-snr-multispeaker_audiohybrid2.lr0.00300.1.g4.b32.d8192.r4.lr0.000300.1.g4.b32/model.ckpt-41761.meta')
    graph = tf.get_default_graph()
    saver.restore(sess, tf.train.latest_checkpoint('./full-snr-multispeaker_audiohybrid2.lr0.00300.1.g4.b32.d8192.r4.lr0.000300.1.g4.b32'))
    #graph.clear_collection('losses')
  
    for i in tqdm(range(0, len(data))):        
        u = np.random.uniform()
        if u > 1: continue
        used_ids.append(ids[i])
           
        #names = [n.name for n in tf.get_default_graph().as_graph_def().node if "Placeholder" in n.op]
    
        X_in = graph.get_tensor_by_name("X:0")
        alpha_in = graph.get_tensor_by_name("alpha:0")

        x =np.reshape(data[i], (1, len(data[i]), 1))
        feed_dict = {X_in:x, alpha_in: 0.1}
        k_tensors = [n for n in graph.as_graph_def().node if 'keras_learning_phase' in n.name]
        #assert len(k_tensors) <= 1
        if k_tensors:
            k_learning_phase = graph.get_tensor_by_name(k_tensors[0].name + ':0')
            feed_dict[k_learning_phase] = False

        # run op and add resulting activations to array
        restored = [graph.get_tensor_by_name(layer) for layer in layers]
        activations = sess.run(restored, feed_dict)
        graph.clear_collection('losses')
        for i in range(len(activations)):
            a = activations[i]
            shape = a.shape
            frag_size = row_size / shape[2]
            a = a[:,(shape[1]-frag_size)/2: (shape[1]+frag_size)/2, :]
            a = np.reshape(a, (-1))
            if layers[i] not in maps: maps[layers[i]] = []
            maps[layers[i]].append(a)
    
    print used_ids
    for layer, acts in tqdm(maps.iteritems()):
        acts = np.array(acts)
        print acts.shape
        name =  re.sub("[/,:]", "_", layer)
        np.save('lstm_merge_activations_' +name, acts)

        # run t-SNE
        tsne = manifold.TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(acts)

        # plot t-SNE
        gs = []
        for i in range(0, X_tsne.shape[0]):
            gs.append(speakerInfoDict[str(used_ids[i])][2])
        colors =[genders[x] for x in gs]
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X_tsne[:,0], X_tsne[:,1], c=colors)
        handles =ax.get_legend_handles_labels()
        pts = [plt.Line2D((0,1),(0,0), color = c, marker = 'o', linestyle = '') for c in genders.values()]
        ax.legend(pts, genders.keys(), loc=4)
        plt.title(name)
        fig.savefig("lstm_merge_t-SNE_gender_" + name+".png")

        plt.clf()

        acs = []
        for i in range(0, X_tsne.shape[0]):
            acs.append(speakerInfoDict[str(used_ids[i])][3])
        colors =[accents[x] for x in acs]
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X_tsne[:,0], X_tsne[:,1], c=colors)
        handles =ax.get_legend_handles_labels()
        pts = [plt.Line2D((0,1),(0,0), color = c, marker = 'o', linestyle = '') for c in accents.values()]
        ax.legend(pts, accents.keys(), loc=4)
        plt.title(name)
        fig.savefig("lstm_merge_t-SNE_accent_" + name+".png")
        plt.clf()
    np.save('lstm_merge_used_ids', np.array(used_ids))






