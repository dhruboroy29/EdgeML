import os
import sys
import tensorflow as tf
import numpy as np

# Making sure edgeml is part of python path
sys.path.insert(0, '../../')
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

np.random.seed(42)
tf.set_random_seed(42)

# MI-RNN and EMI-RNN imports
from edgeml.graph.rnn import EMI_DataPipeline
from edgeml.graph.rnn import EMI_BasicLSTM
from edgeml.trainer.emirnnTrainer import EMI_Trainer, EMI_Driver
import edgeml.utils

# Network parameters for our LSTM + FC Layer
NUM_HIDDEN = 16 # TODO: Make this an input argument
NUM_TIMESTEPS = 48 # TODO: Make this an input argument
ORIGINAL_NUM_TIMESTEPS = 256 # TODO: Make this an input argument
NUM_FEATS = 2 # TODO: Make this an input argument
FORGET_BIAS = 1.0 # TODO: Make this an input argument
NUM_OUTPUT = 2 # TODO: Make this an input argument
USE_DROPOUT = True # TODO: Make this an input argument
KEEP_PROB = 0.75 # TODO: Make this an input argument

# For dataset API
PREFETCH_NUM = 5 # TODO: Make this an input argument
BATCH_SIZE = 32 # TODO: Make this an input argument

# Number of epochs in *one iteration*
NUM_EPOCHS = 3 # TODO: Make this an input argument
# Number of iterations in *one round*. After each iteration,
# the model is dumped to disk. At the end of the current
# round, the best model among all the dumped models in the
# current round is picked up..
NUM_ITER = 4 # TODO: Make this an input argument
# A round consists of multiple training iterations and a belief
# update step using the best model from all of these iterations
NUM_ROUNDS = 10 # TODO: Make this an input argument
LEARNING_RATE=0.001 # TODO: Make this an input argument

# A staging directory to store models
MODEL_PREFIX = '/tmp/model-lstm'

# Loading the data
# TODO: Make this an input argument
data_dir = '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Displacement_Detection/Data/Austere_subset_features/' \
           'Raw_winlen_256_stride_171/48_16/'

x_train, y_train = np.load(data_dir + 'x_train.npy'), np.load(data_dir + 'y_train.npy')
x_test, y_test = np.load(data_dir + 'x_test.npy'), np.load(data_dir + 'y_test.npy')
x_val, y_val = np.load(data_dir + 'x_val.npy'), np.load(data_dir + 'y_val.npy')

# BAG_TEST, BAG_TRAIN, BAG_VAL represent bag_level labels. These are used for the label update
# step of EMI/MI RNN
BAG_TEST = np.argmax(y_test[:, 0, :], axis=1)
BAG_TRAIN = np.argmax(y_train[:, 0, :], axis=1)
BAG_VAL = np.argmax(y_val[:, 0, :], axis=1)
NUM_SUBINSTANCE = x_train.shape[1]
print("x_train shape is:", x_train.shape)
print("y_train shape is:", y_train.shape)
print("x_test shape is:", x_val.shape)
print("y_test shape is:", y_val.shape)


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


EMI_BasicLSTM._createExtendedGraph = createExtendedGraph
EMI_BasicLSTM._restoreExtendedGraph = restoreExtendedGraph

if USE_DROPOUT is True:
    EMI_Driver.feedDictFunc = feedDictFunc

inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
emiLSTM = EMI_BasicLSTM(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS,
                        forgetBias=FORGET_BIAS, useDropout=USE_DROPOUT)
emiTrainer = EMI_Trainer(NUM_TIMESTEPS, NUM_OUTPUT, lossType='xentropy',
                         stepSize=LEARNING_RATE)


# Connect elementary parts together to create forward graph
tf.reset_default_graph()
g1 = tf.Graph()
with g1.as_default():
    # Obtain the iterators to each batch of the data
    x_batch, y_batch = inputPipeline()
    # Create the forward computation graph based on the iterators
    y_cap = emiLSTM(x_batch)
    # Create loss graphs and training routines
    emiTrainer(y_cap, y_batch)

with g1.as_default():
    emiDriver = EMI_Driver(inputPipeline, emiLSTM, emiTrainer)

emiDriver.initializeSession(g1)
y_updated, modelStats = emiDriver.run(numClasses=NUM_OUTPUT, x_train=x_train,
                                      y_train=y_train, bag_train=BAG_TRAIN,
                                      x_val=x_val, y_val=y_val, bag_val=BAG_VAL,
                                      numIter=NUM_ITER, keep_prob=KEEP_PROB,
                                      numRounds=NUM_ROUNDS, batchSize=BATCH_SIZE,
                                      numEpochs=NUM_EPOCHS, modelPrefix=MODEL_PREFIX,
                                      fracEMI=0.5, updatePolicy='top-k', k=1)

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

k = 2
predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                               minProb=0.99, keep_prob=1.0)
bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
print('Accuracy at k = %d: %f' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))))
mi_savings = (1 - NUM_TIMESTEPS / ORIGINAL_NUM_TIMESTEPS)
emi_savings = getEarlySaving(predictionStep, NUM_TIMESTEPS)
total_savings = mi_savings + (1 - mi_savings) * emi_savings
print('Savings due to MI-RNN : %f' % mi_savings)
print('Savings due to Early prediction: %f' % emi_savings)
print('Total Savings: %f' % (total_savings))


# A slightly more detailed analysis method is provided.
df = emiDriver.analyseModel(predictions, BAG_TEST, NUM_SUBINSTANCE, NUM_OUTPUT)

# Pick the best model
devnull = open(os.devnull, 'r')
for val in modelStats:
    round_, acc, modelPrefix, globalStep = val
    emiDriver.loadSavedGraphToNewSession(modelPrefix, globalStep, redirFile=devnull)
    predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                               minProb=0.99, keep_prob=1.0)

    bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
    print("Round: %2d, Validation accuracy: %.4f" % (round_, acc), end='')
    print(', Test Accuracy (k = %d): %f, ' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))), end='')
    mi_savings = (1 - NUM_TIMESTEPS / ORIGINAL_NUM_TIMESTEPS)
    emi_savings = getEarlySaving(predictionStep, NUM_TIMESTEPS)
    total_savings = mi_savings + (1 - mi_savings) * emi_savings
    print('Additional savings: %f' % emi_savings)
    print("Total Savings: %f" % total_savings)