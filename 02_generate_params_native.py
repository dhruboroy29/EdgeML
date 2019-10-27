import codecs
import json

import numpy as np

I = np.array(pow(10,5))
num_instances = 8
num_timesteps = 12
num_classes = 2

def formatp(v, name):
    if v.ndim == 2:
        arrs = v.tolist()
        print("static const ll " + name + "[][" + str(v.shape[1]) + "] = {", end="")
        for i in range(arrs.__len__() - 1):
            print("{", end="")
            for j in range(arrs[i].__len__() - 1):
                print("%d" % arrs[i][j], end=",")
            print("%d" % arrs[i][arrs[i].__len__() - 1], end="},")
        print("{", end="")
        for j in range(arrs[arrs.__len__() - 1].__len__() - 1):
            print("%d" % arrs[arrs.__len__() - 1][j], end=",")
        print("%d" % arrs[arrs.__len__() - 1][arrs[arrs.__len__() - 1].__len__() - 1], end="}};\n")
    elif v.ndim == 1:
        print("const ll " + name + "[" + str(v.shape[0]) + "] = {", end="")
        arrs = v.tolist()
        for i in range(arrs.__len__() - 1):
            print("%d" % arrs[i], end=",")
        print("%d" % arrs[arrs.__len__() - 1], end="};\n")
    elif v.ndim == 0:
        print("static const ll " + name + "= " + str(v.tolist()) + ";")


# Load quantized params
modelloc = 'buildsys_model/model_O=2_H=32_k=6_gN=quantSigm_uN=quantTanh_ep=50_it=10_rnd=5_bs=64/Params'

qW1 = np.load(modelloc + "/QuantizedFastModel/qW1.npy")
qFC_Bias = np.load(modelloc + "/QuantizedFastModel/qFC_Bias.npy")
qW2 = np.load(modelloc + "/QuantizedFastModel/qW2.npy")
qU2 = np.load(modelloc + "/QuantizedFastModel/qU2.npy")
qFC_Weight = np.load(modelloc + "/QuantizedFastModel/qFC_Weight.npy")
qU1 = np.load(modelloc + "/QuantizedFastModel/qU1.npy")
qB_g = np.load(modelloc + "/QuantizedFastModel/qB_g.npy").ravel()
qB_h = np.load(modelloc + "/QuantizedFastModel/qB_h.npy").ravel()
q = np.load(modelloc + "/QuantizedFastModel/paramScaleFactor.npy")

# Get mean and std
statsfile = 'buildsys_model/train_stats.json'
load_stats = json.loads(codecs.open(statsfile, 'r', encoding='utf-8').read())
# Convert values to numpy arrays
load_stats.update((k, np.array(v)) for k, v in load_stats.items())

mean = load_stats['mean']
std = load_stats['std']

# Convert matrices to C++ format
print("Copy and run below code to get model size:\n\n")
print("typedef long long ll;\n")
formatp(np.transpose(qW1), 'qW1_transp_l')
formatp(qFC_Bias, 'qFC_Bias_l')
formatp(np.transpose(qW2), 'qW2_transp_l')
formatp(np.transpose(qU2), 'qU2_transp_l')
formatp(qFC_Weight, 'qFC_Weight_l')
formatp(np.transpose(qU1), 'qU1_transp_l')
formatp(I*qB_g, 'qB_g_l')
formatp(I*qB_h, 'qB_h_l')
print("")
formatp(mean, 'mean_l')
formatp(std, 'stdev_l')
print("")
formatp(q, 'q_l')
formatp(I, 'I_l')
formatp(q*I, 'q_times_I_l')

print('\nstatic const int wRank = ' + str(qW2.shape[0]) + ";")
print('static const int uRank = ' + str(qU2.shape[0]) + ";")
print('static const int inputDims = ' + str(qW1.shape[0]) + ";")
print('static const int hiddenDims = ' + str(qU1.shape[0]) + ";")
print('static const int timeSteps = ' + str(num_timesteps) + ";")
print('static const int numInstances = ' + str(num_instances) + ";")
print('static const int numClasses = ' + str(num_classes) + ";")

print("\nint main(){\n"
      "\tint size = sizeof(qW1_transp_l) + sizeof(qFC_Bias_l) + sizeof(qW2_transp_l) "
      "+ sizeof(qU2_transp_l) + sizeof(qFC_Weight_l) + sizeof(qU1_transp_l) + sizeof(qB_g_l) "
      "+ sizeof(qB_h_l) + sizeof(q_l) + sizeof(I_l) + sizeof(mean_l) + sizeof(stdev_l) "
      "+ sizeof(I_l_vec) + sizeof(q_times_I_l);\n"
      "\tprintf(\"Model size: %d KB\\n\", size/1000);\n" \
                                    "}")
