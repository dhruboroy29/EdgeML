import codecs
import json
import sys

import numpy as np

I = np.array(pow(10,5))
num_instances = 8
num_timesteps = 12
num_classes = 2

test_data = np.load('C++/test_data.npy')

# Load quantized params
modelloc = 'buildsys_model/model_O=2_H=32_k=6_gN=quantSigm_uN=quantTanh_ep=50_it=10_rnd=5_bs=64/Params'

qW1 = np.load(modelloc + "/QuantizedFastModel/qW1.npy")
qFC_Bias = np.load(modelloc + "/QuantizedFastModel/qFC_Bias.npy")
qW2 = np.load(modelloc + "/QuantizedFastModel/qW2.npy")
qU2 = np.load(modelloc + "/QuantizedFastModel/qU2.npy")
qFC_Weight = np.load(modelloc + "/QuantizedFastModel/qFC_Weight.npy")
qU1 = np.load(modelloc + "/QuantizedFastModel/qU1.npy")
qB_g = np.transpose(np.load(modelloc + "/QuantizedFastModel/qB_g.npy"))
qB_h = np.transpose(np.load(modelloc + "/QuantizedFastModel/qB_h.npy"))
q = np.load(modelloc + "/QuantizedFastModel/paramScaleFactor.npy")

# Get mean and std
statsfile = 'buildsys_model/train_stats.json'
load_stats = json.loads(codecs.open(statsfile, 'r', encoding='utf-8').read())
# Convert values to numpy arrays
load_stats.update((k, np.array(v)) for k, v in load_stats.items())

mean = load_stats['mean']
std = load_stats['std']

def quantTanh(x, scale):
    return np.maximum(-scale, np.minimum(scale, x))


def quantSigm(x, scale):
    return np.maximum(np.minimum(0.5 * (x + scale), scale), 0)


def nonlin(code, x, scale):
    if (code == "quantTanh"):
        return quantTanh(x, scale)
    elif (code == "quantSigm"):
        return quantSigm(x, scale)

NUM_HIDDEN = 32
UPDATE_NL = "quantTanh"
GATE_NL = "quantSigm"

fpt = int

def predict_quant(points, I):
    preds = []

    assert points.ndim == 3

    for i in range(points.shape[0]):
        h = np.array(np.zeros((NUM_HIDDEN, 1)), dtype=fpt)
        # print(h)
        for t in range(points.shape[1]):
            # x = np.array((I * (np.array(points[i][slice(t * stride, t * stride + window)]) - fpt(mean))) / fpt(std),
            #              dtype=fpt).reshape((-1, 1))
            x = np.array((I * (points[i, t] - mean.astype(fpt))) / std.astype(fpt), dtype=fpt).reshape((-1, 1))
            pre = np.array(
                (np.matmul(np.transpose(qW2), np.matmul(np.transpose(qW1), x)) + np.matmul(np.transpose(qU2),
                                                                                           np.matmul(np.transpose(qU1),
                                                                                                     h))) / (
                        q * 1), dtype=fpt)
            h_ = np.array(nonlin(UPDATE_NL, pre + qB_h * I, q * I) / (q), dtype=fpt)
            z = np.array(nonlin(GATE_NL, pre + qB_g * I, q * I) / (q), dtype=fpt)
            zeta, nu =1, 0
            h = np.array((np.multiply(z, h) + np.array(np.multiply(fpt(I * zeta) * (I - z) + fpt(I * nu) * I, h_) / I,
                                                       dtype=fpt)) / I, dtype=fpt)

        preds.append(np.matmul(np.transpose(h), qFC_Weight) + qFC_Bias)
    return np.array(preds)

# Run test
I = 5
softmax=[]
scale = pow(10, I)

instance_preds = predict_quant(test_data, scale)
softmax.append(instance_preds)

