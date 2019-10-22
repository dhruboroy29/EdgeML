import os
import errno
import re
import sys
import tensorflow as tf
import numpy as np
import argparse
import time
import csv
import getpass
import json, codecs

# Making sure edgeml is part of python path
sys.path.insert(0, 'tf/')
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(42)
tf.set_random_seed(42)

# FastGRNN and FastRNN imports
import edgeml.utils as utils
from edgeml.graph.rnn import EMI_DataPipeline
from edgeml.graph.rnn import EMI_FastGRNN
from edgeml.trainer.emirnnTrainer import EMI_Trainer, EMI_Driver

model_path = 'buildsys_model/model_O=2_H=32_k=6_gN=quantSigm_uN=quantTanh_ep=50_it=10_rnd=5_bs=64.json'
stats_path = 'buildsys_model/train_stats.json'

# Infer params from model name
m = re.match('model_O=(?P<_0>.+)_H=(?P<_1>.+)_k=(?P<_2>.+)_gN=(?P<_3>.+)_uN=(?P<_4>.+)_ep=(?P<_5>.+)_it=(?P<_6>.+)_rnd=(?P<_7>.+)_bs=(?P<_8>.+)\.json',
                     os.path.basename(model_path))

inferred_params = [y[1] for y in sorted(m.groupdict().items())]

NUM_OUTPUT = int(inferred_params[0])
NUM_HIDDEN = int(inferred_params[1])
k = int(inferred_params[2])
GATE_NL = inferred_params[3]
UPDATE_NL = inferred_params[4]
NUM_EPOCHS = int(inferred_params[5])
NUM_ITER = int(inferred_params[6])
NUM_ROUNDS = int(inferred_params[7])
BATCH_SIZE = int(inferred_params[8])

# Load model params
load_data = json.loads(codecs.open(model_path, 'r', encoding='utf-8').read())
# Convert values to numpy arrays
load_data.update((k, np.array(v)) for k, v in load_data.items())

# Instantiate RNN params
qFC_Weight = load_data['W1:0']
qFC_Bias = load_data['B1:0']
qW1 = load_data['rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/W1:0']
qW2 = load_data['rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/W2:0']
qU1 = load_data['rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/U1:0']
qU2 = load_data['rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/U2:0']
qB_g = load_data['rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/B_g:0']
qB_h = load_data['rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/B_h:0']
zeta = load_data['rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/zeta:0']
nu = load_data['rnn/fast_grnn_cell/EMI-FastGRNN-Cell/FastGRNNcell/nu:0']

# Load mean and std for standardization
load_stats = json.loads(codecs.open(stats_path, 'r', encoding='utf-8').read())
# Convert values to numpy arrays
load_stats.update((k, np.array(v)) for k, v in load_stats.items())

# Get mean and std
mean = load_stats['mean']
std = load_stats['std']

data_dir = '/path/to/data'

x_test, y_test = np.load(os.path.join(data_dir,'x_test_unnorm.npy')), np.load(os.path.join(data_dir,'y_test_unnorm.npy'))

# BAG_TEST, BAG_TRAIN, BAG_VAL represent bag_level labels. These are used for the label update
# step of EMI/MI RNN
BAG_TEST = np.argmax(y_test[:, 0, :], axis=1)

# Inferred params
NUM_SUBINSTANCE = x_test.shape[1]
NUM_TIMESTEPS = x_test.shape[2]
NUM_FEATS = x_test.shape[-1]

print("x_test shape is:", x_test.shape)
print("y_test shape is:", y_test.shape)

'''
Insert Pranshu logic for instance-level predictions
'''
predictions =

'Insert Pranshu logic for bag predictions. See the following for reference:'
bagPredictions = emiDriver.getBagPredictions(predictions, k=k, numClass=NUM_OUTPUT)

print(', Test Accuracy (k = %d): %f, ' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))), end='')

test_acc = np.mean((bagPredictions == BAG_TEST).astype(int))

# Print confusion matrix
print('\n')
bagcmatrix = utils.getConfusionMatrix(bagPredictions, BAG_TEST, NUM_OUTPUT)
utils.printFormattedConfusionMatrix(bagcmatrix)
print('\n')

print('Obtaining trained graph variables')
emiDriver.save_model_json(graph, '../../../buildsys_model/model_O=' + str(NUM_OUTPUT) + '_H=' + str(NUM_HIDDEN) + '_k=' + str(k)
                                          + '_gN=' + GATE_NL + '_uN=' + UPDATE_NL + '_ep=' + str(NUM_EPOCHS)
                                          + '_it=' + str(NUM_ITER) + '_rnd=' + str(NUM_ROUNDS)
                                          + '_bs=' + str(BATCH_SIZE) + '.json')

