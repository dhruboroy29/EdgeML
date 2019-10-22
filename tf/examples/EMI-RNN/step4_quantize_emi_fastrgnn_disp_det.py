import os
import errno
import sys
import tensorflow as tf
import numpy as np
import argparse
import time
import csv
import getpass

# Making sure edgeml is part of python path
sys.path.insert(0, '../tf/')
sys.path.insert(0, 'tf/')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(42)
tf.set_random_seed(42)

# FastGRNN and FastRNN imports
import edgeml.utils as utils
from edgeml.graph.rnn import EMI_DataPipeline
from edgeml.graph.rnn import EMI_FastGRNN
from edgeml.trainer.emirnnTrainer import EMI_Trainer, EMI_Driver

parser = argparse.ArgumentParser(description='HyperParameters for EMI-FastGRNN')
parser.add_argument('-k', type=int, default=2, help='Min. number of consecutive target instances. 100 for max possible')
parser.add_argument('-H', type=int, default=16, help='Number of hidden units')
# parser.add_argument('-ts', type=int, default=48, help='Number of timesteps')
parser.add_argument('-ots', type=int, default=256, help='Original number of timesteps')
# parser.add_argument('-F', type=int, default=2, help='Number of features')
parser.add_argument('-fb', type=float, default=1.0, help='Forget bias')
parser.add_argument('-O', type=int, default=2, help='Number of outputs')
parser.add_argument('-d', type=bool, default=False, help='Dropout?')
parser.add_argument('-kp', type=float, default=0.9, help='Keep probability')
parser.add_argument('-uN', type=str, default="quantTanh", help='Update nonlinearity')
parser.add_argument('-gN', type=str, default="quantSigm", help='Gate nonlinearity')
parser.add_argument('-wR', type=int, default=5, help='Rank of W')
parser.add_argument('-uR', type=int, default=6, help='Rank of U')
parser.add_argument('-bs', type=int, default=32, help='Batch size')
parser.add_argument('-ep', type=int, default=3, help='Number of epochs per iteration')
parser.add_argument('-it', type=int, default=4, help='Number of iterations per round')
parser.add_argument('-rnd', type=int, default=10, help='Number of rounds')
parser.add_argument('-Dat', type=str, help='Data directory')
parser.add_argument('-out', type=str, default=sys.stdout, help='Output filename')

args = parser.parse_args()

# Network parameters for our FastGRNN + FC Layer
k = args.k  # 2
NUM_HIDDEN = args.H  # 16
ORIGINAL_NUM_TIMESTEPS = args.ots  # 256
FORGET_BIAS = args.fb  # 1.0
NUM_OUTPUT = args.O  # 2
USE_DROPOUT = args.d  # False
KEEP_PROB = args.kp  # 0.9

# Non-linearities can be chosen among "tanh, sigmoid, relu, quantTanh, quantSigm"
UPDATE_NL = args.uN  # "quantTanh"
GATE_NL = args.gN  # "quantSigm"

# Ranks of Parameter matrices for low-rank parameterisation to compress models.
WRANK = args.wR  # 5
URANK = args.uR  # 6

# For dataset API
PREFETCH_NUM = 5
BATCH_SIZE = args.bs  # 32

# Number of epochs in *one iteration*
NUM_EPOCHS = args.ep  # 3
# Number of iterations in *one round*. After each iteration,
# the model is dumped to disk. At the end of the current
# round, the best model among all the dumped models in the
# current round is picked up..
NUM_ITER = args.it  # 4
# A round consists of multiple training iterations and a belief
# update step using the best model from all of these iterations
NUM_ROUNDS = args.rnd  # 10
# LEARNING_RATE=0.001

# Loading the data
data_dir = args.Dat  # '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Displacement_Detection/Data/Austere_subset_features/' \
# 'Raw_winlen_256_stride_171/48_16/'

x_train, y_train = np.load(os.path.join(data_dir, 'x_train.npy')), np.load(os.path.join(data_dir, 'y_train.npy'))
x_test, y_test = np.load(os.path.join(data_dir, 'x_test.npy')), np.load(os.path.join(data_dir, 'y_test.npy'))
x_val, y_val = np.load(os.path.join(data_dir, 'x_val.npy')), np.load(os.path.join(data_dir, 'y_val.npy'))

# BAG_TEST, BAG_TRAIN, BAG_VAL represent bag_level labels. These are used for the label update
# step of EMI/MI RNN
BAG_TEST = np.argmax(y_test[:, 0, :], axis=1)
BAG_TRAIN = np.argmax(y_train[:, 0, :], axis=1)
BAG_VAL = np.argmax(y_val[:, 0, :], axis=1)

# Inferred params
NUM_SUBINSTANCE = x_train.shape[1]
NUM_TIMESTEPS = x_train.shape[2]
NUM_FEATS = x_train.shape[-1]

print("x_train shape is:", x_train.shape)
print("y_train shape is:", y_train.shape)
print("x_test shape is:", x_val.shape)
print("y_test shape is:", y_val.shape)

# Adjustment for max k: number of subinstances
if k == 100:
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
print(', Test Accuracy (k = %d): %f, ' % (k, np.mean((bagPredictions == BAG_TEST).astype(int))), end='')

test_acc = np.mean((bagPredictions == BAG_TEST).astype(int))

# Print confusion matrix
print('\n')
bagcmatrix = utils.getConfusionMatrix(bagPredictions, BAG_TEST, NUM_OUTPUT)
utils.printFormattedConfusionMatrix(bagcmatrix)
print('\n')

# Save model
print('Saving model...')
modelloc = '../../../buildsys_model/model_O=' + str(NUM_OUTPUT) + '_H=' + str(NUM_HIDDEN) + '_k=' + str(k) \
           + '_gN=' + GATE_NL + '_uN=' + UPDATE_NL + '_ep=' + str(NUM_EPOCHS) \
           + '_it=' + str(NUM_ITER) + '_rnd=' + str(NUM_ROUNDS) \
           + '_bs=' + str(BATCH_SIZE)

os.makedirs(modelloc, exist_ok=True)
emiDriver.save_model_npy(modelloc)

# Run quantization
os.system("rm -r " + modelloc + "/QuantizedFastModel/")
os.system(
    "python3 " + os.path.abspath('../../EdgeML/tf/examples/FastCells/quantizeFastModels.py') + " -dir " + modelloc)

qW1 = np.load(modelloc + "/QuantizedFastModel/qW1.npy")
qFC_Bias = np.load(modelloc + "/QuantizedFastModel/qFC_Bias.npy")
qW2 = np.load(modelloc + "/QuantizedFastModel/qW2.npy")
qU2 = np.load(modelloc + "/QuantizedFastModel/qU2.npy")
qFC_Weight = np.load(modelloc + "/QuantizedFastModel/qFC_Weight.npy")
qU1 = np.load(modelloc + "/QuantizedFastModel/qU1.npy")
qB_g = np.transpose(np.load(modelloc + "/QuantizedFastModel/qB_g.npy"))
qB_h = np.transpose(np.load(modelloc + "/QuantizedFastModel/qB_h.npy"))
q = np.load(modelloc + "/QuantizedFastModel/paramScaleFactor.npy")

print("qW1 = ", qW1)
zeta = np.load(modelloc + "/zeta.npy")

zeta = 1 / (1 + np.exp(-zeta))
# print("zeta = ",zeta)
nu = np.load(modelloc + "/nu.npy")
nu = 1 / (1 + np.exp(-nu))


# I = 1

def quantTanh(x, scale):
    return np.maximum(-scale, np.minimum(scale, x))


def quantSigm(x, scale):
    return np.maximum(np.minimum(0.5 * (x + scale), scale), 0)


def nonlin(code, x, scale):
    if (code == "quantTanh"):
        return quantTanh(x, scale)
    elif (code == "quantSigm"):
        return quantSigm(x, scale)

fpt = int

def predict(points, lbls, I):
    pred_lbls = []

    for i in range(points.shape[0]):
        h = np.array(np.zeros((hidden_dim, 1)), dtype=fpt)
        # print(h)
        for t in range(seq_max_len):
            x = np.array((I * (np.array(points[i][slice(t * stride, t * stride + window)]) - fpt(mean))) / fpt(std),
                         dtype=fpt).reshape((-1, 1))
            pre = np.array((np.matmul(np.transpose(qW2), np.matmul(np.transpose(qW1), x)) + np.matmul(np.transpose(qU2),
                                                                                                      np.matmul(
                                                                                                          np.transpose(
                                                                                                              qU1),
                                                                                                          h))) / (
                                   q * 1), dtype=fpt)
            h_ = np.array(nonlin(UPDATE_NL, pre + qB_h * I, q * I) / (q), dtype=fpt)
            z = np.array(nonlin(GATE_NL, pre + qB_g * I, q * I) / (q), dtype=fpt)
            h = np.array((np.multiply(z, h) + np.array(np.multiply(fpt(I * zeta) * (I - z) + fpt(I * nu) * I, h_) / I,
                                                       dtype=fpt)) / I, dtype=fpt)

        pred_lbls.append(np.argmax(np.matmul(np.transpose(h), qFC_Weight) + qFC_Bias))
    pred_lbls = np.array(pred_lbls)
    # print(lbls)
    # print(pred_lbls)
    print(float((pred_lbls == lbls).sum()) / lbls.shape[0])

for I in range(10):
    predict(train_cuts, train_cuts_lbls, pow(10, I))
