import codecs
import json
import sys

import numpy as np

I = np.array(pow(10,5))
num_instances = 8
num_timesteps = 12
num_classes = 2

update_nl = "quantTanh"
gate_nl = "quantSigm"

test_in_path = "/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/" \
                "Data/Austere/BuildSys_Demo/Windowed/winlen_256_stride_128/12_8/x_test_unnorm.npy"

def formatp(v, name, headless=False, endwith=";\n", file=sys.stdout):
    if v.ndim == 3:
        print("static const uint " + name + "[][" + str(v.shape[1]) + "][" + str(v.shape[2]) + "] = {", end="", file=file)
        for i in range(v.__len__() - 1):
            formatp(v[i], 'xyz', headless=True, endwith=",", file=file)
        formatp(v[v.__len__() - 1], 'xyz', headless=True, endwith="};\n", file=file)
    elif v.ndim == 2:
        arrs = v.tolist()
        if headless:
            print("{", end="", file=file)
        else:
            print("static const ll " + name + "[][" + str(v.shape[1]) + "] = {", end="", file=file)
        for i in range(arrs.__len__() - 1):
            print("{", end="", file=file)
            for j in range(arrs[i].__len__() - 1):
                print("%d" % arrs[i][j], end=",", file=file)
            print("%d" % arrs[i][arrs[i].__len__() - 1], end="},", file=file)
        print("{", end="", file=file)
        for j in range(arrs[arrs.__len__() - 1].__len__() - 1):
            print("%d" % arrs[arrs.__len__() - 1][j], end=",", file=file)
        print("%d" % arrs[arrs.__len__() - 1][arrs[arrs.__len__() - 1].__len__() - 1], end="}}"+endwith, file=file)
    elif v.ndim == 1:
        print("static const ll " + name + "[" + str(v.shape[0]) + "] = {", end="", file=file)
        arrs = v.tolist()
        for i in range(arrs.__len__() - 1):
            print("%d" % arrs[i], end=",", file=file)
        print("%d" % arrs[arrs.__len__() - 1], end="};\n", file=file)
    elif v.ndim == 0:
        print("static const ll " + name + "= " + str(v.tolist()) + ";", file=file)


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
# print("Copy and run below code to get model size:\n\n")
# print("typedef long long ll;\n")
model_params = open('C++/model_params.h', 'w')

print("typedef long long ll;\n", file=model_params)
formatp(np.transpose(qW1), 'qW1_transp_l', file=model_params)
formatp(qFC_Bias, 'qFC_Bias_l', file=model_params)
formatp(np.transpose(qW2), 'qW2_transp_l', file=model_params)
formatp(np.transpose(qU2), 'qU2_transp_l', file=model_params)
formatp(np.transpose(qFC_Weight), 'qFC_Weight_l', file=model_params)
formatp(np.transpose(qU1), 'qU1_transp_l', file=model_params)
formatp(I * qB_g, 'qB_g_l', file=model_params)
formatp(I * qB_h, 'qB_h_l', file=model_params)
print("", file=model_params)
formatp(mean, 'mean_l', file=model_params)
formatp(std, 'stdev_l', file=model_params)
print("", file=model_params)
formatp(q, 'q_l', file=model_params)
formatp(I, 'I_l', file=model_params)
formatp(q * I, 'q_times_I_l', file=model_params)

formatp(I * np.ones(qU1.shape[0], dtype=int), 'I_l_vec', file=model_params)

print('\nstatic const int wRank = ' + str(qW2.shape[0]) + ";", file=model_params)
print('static const int uRank = ' + str(qU2.shape[0]) + ";", file=model_params)
print('static const int inputDims = ' + str(qW1.shape[0]) + ";", file=model_params)
print('static const int hiddenDims = ' + str(qU1.shape[0]) + ";", file=model_params)
print('static const int timeSteps = ' + str(num_timesteps) + ";", file=model_params)
print('static const int numInstances = ' + str(num_instances) + ";", file=model_params)
print('static const int numClasses = ' + str(num_classes) + ";", file=model_params)
print('\n#define UPDATE_NL', update_nl, file=model_params)
print('#define GATE_NL', gate_nl, file=model_params)

test_data = open('C++/test_data.h', 'w')
print("#ifndef MOTE", file=test_data)
test_in = np.load(test_in_path)
test_in = test_in.reshape(-1, test_in.shape[2], test_in.shape[3])
formatp(test_in, 'test_inputs', file=test_data)
print("static const int numData = " + str(test_in.shape[0]) + ";", file=test_data)
print("#else", file=test_data)
test_in_mote = test_in[0:30]
formatp(test_in_mote, 'test_inputs', file=test_data)
print("static const int numData = " + str(test_in_mote.shape[0]) + ";", file=test_data)
print("#endif", file=test_data)

np.save('C++/test_data.npy', test_in)

# print("\nint main(){\n"
#       "\tint size = sizeof(qW1_transp_l) + sizeof(qFC_Bias_l) + sizeof(qW2_transp_l) "
#       "+ sizeof(qU2_transp_l) + sizeof(qFC_Weight_l) + sizeof(qU1_transp_l) + sizeof(qB_g_l) "
#       "+ sizeof(qB_h_l) + sizeof(q_l) + sizeof(I_l) + sizeof(mean_l) + sizeof(stdev_l) "
#       "+ sizeof(I_l_vec) + sizeof(q_times_I_l);\n"
#       "\tprintf(\"Model size: %d KB\\n\", size/1000);\n" \
#                                     "}")
