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

data_dir = '/path/to/data'

x_test, y_test = np.load(os.path.join(data_dir,'x_test.npy')), np.load(os.path.join(data_dir,'y_test.npy'))

# BAG_TEST, BAG_TRAIN, BAG_VAL represent bag_level labels. These are used for the label update
# step of EMI/MI RNN
BAG_TEST = np.argmax(y_test[:, 0, :], axis=1)

# Inferred params
NUM_SUBINSTANCE = x_train.shape[1]
NUM_TIMESTEPS = x_train.shape[2]
NUM_FEATS = x_train.shape[-1]

print("x_train shape is:", x_train.shape)
print("y_train shape is:", y_train.shape)
print("x_test shape is:", x_val.shape)
print("y_test shape is:", y_val.shape)

# Adjustment for max k: number of subinstances
if k==100:
    k = x_train.shape[1]


# Define the linear secondary classifier
def createExtendedGraph(self, baseOutput, *args, **kwargs):
    W1 = tf.Variable(np.random.normal(size=[NUM_HIDDEN, NUM_OUTPUT]).astype('float32'), name='W1')
    B1 = tf.Variable(np.random.normal(size=[NUM_OUTPUT]).astype('float32'), name='B1')
    y_cap = tf.add(tf.tensordot(baseOutput, W1, axes=1), B1, name='y_cap_tata')
    self.output = y_cap
    self.graphCreated = True


def restoreExtendedGraph(self, graph, *args, **kwargs):
    y_cap = graph.get_tensor_by_name('y_cap_tata:0')
    self.output = y_cap
    self.graphCreated = True


def feedDictFunc(self, keep_prob=None, inference=False, **kwargs):
    if inference is False:
        feedDict = {self._emiGraph.keep_prob: keep_prob}
    else:
        feedDict = {self._emiGraph.keep_prob: 1.0}
    return feedDict


EMI_FastGRNN._createExtendedGraph = createExtendedGraph
EMI_FastGRNN._restoreExtendedGraph = restoreExtendedGraph
if USE_DROPOUT is True:
    EMI_FastGRNN.feedDictFunc = feedDictFunc

inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
emiFastGRNN = EMI_FastGRNN(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS, wRank=WRANK, uRank=URANK,
                           gate_non_linearity=GATE_NL, update_non_linearity=UPDATE_NL, useDropout=USE_DROPOUT)
emiTrainer = EMI_Trainer(NUM_TIMESTEPS, NUM_OUTPUT, lossType='xentropy')


# Connect elementary parts together to create forward graph
tf.reset_default_graph()
g1 = tf.Graph()
with g1.as_default():
    # Obtain the iterators to each batch of the data
    x_batch, y_batch = inputPipeline()
    # Create the forward computation graph based on the iterators
    y_cap = emiFastGRNN(x_batch)
    # Create loss graphs and training routines
    emiTrainer(y_cap, y_batch)


with g1.as_default():
    emiDriver = EMI_Driver(inputPipeline, emiFastGRNN, emiTrainer)

emiDriver.initializeSession(g1, config=config)

'''
Evaluating the  trained model
'''

# Early Prediction Policy: We make an early prediction based on the predicted classes
#     probability. If the predicted class probability > minProb at some step, we make
#     a prediction at that step.
def earlyPolicy_minProb(instanceOut, minProb, **kwargs):
    assert instanceOut.ndim == 2
    classes = np.argmax(instanceOut, axis=1)
    prob = np.max(instanceOut, axis=1)
    index = np.where(prob >= minProb)[0]
    if len(index) == 0:
        assert (len(instanceOut) - 1) == (len(classes) - 1)
        return classes[-1], len(instanceOut) - 1
    index = index[0]
    return classes[index], index

def getEarlySaving(predictionStep, numTimeSteps, returnTotal=False):
    predictionStep = predictionStep + 1
    predictionStep = np.reshape(predictionStep, -1)
    totalSteps = np.sum(predictionStep)
    maxSteps = len(predictionStep) * numTimeSteps
    savings = 1.0 - (totalSteps / maxSteps)
    if returnTotal:
        return savings, totalSteps
    return savings

# Pick the best model
devnull = open(os.devnull, 'r')
acc = 0.0

# Read model stats file
with open(os.path.join(data_dir, 'modelstats_O=' + str(NUM_OUTPUT) + '_H=' + str(NUM_HIDDEN) + '_k=' + str(k)
                                          + '_gN=' + GATE_NL + '_uN=' + UPDATE_NL + '_ep=' + str(NUM_EPOCHS)
                                          + '_it=' + str(NUM_ITER) + '_rnd=' + str(NUM_ROUNDS)
                                          + '_bs=' + str(BATCH_SIZE) + '.csv'), 'r') as stats_csv:
    modelStats = csv.reader(stats_csv)
    header = next(modelStats)
    for row in modelStats:
        c_round_, c_acc, c_modelPrefix, c_globalStep = row
        if float(c_acc) > acc:
            round_, acc, modelPrefix, globalStep = int(c_round_), float(c_acc), c_modelPrefix, int(c_globalStep)

print('Best Model: ', modelPrefix, globalStep)

graph = emiDriver.loadSavedGraphToNewSession(modelPrefix, globalStep, redirFile=devnull)

predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                               minProb=0.99, keep_prob=1.0)

bagPredictions = emiDriver.getBagPredictions(predictions, k=k, numClass=NUM_OUTPUT)
print("Round: %2d, window length: %3d, Validation accuracy: %.4f" % (round_, ORIGINAL_NUM_TIMESTEPS, acc), end='')
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

