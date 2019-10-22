import os
import numpy as np
from sklearn.model_selection import train_test_split
import json, codecs
from helpermethods import ReadRadarWindows, one_hot, bagData
import argparse

np.random.seed(42)

# Shuffle
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# def getRadarData(path):
#     # Shuffle
#     def unison_shuffled_copies(a, b):
#         assert len(a) == len(b)
#         p = np.random.permutation(len(a))
#         return a[p], b[p]
#
#     noise_data = ReadRadarWindows(os.path.join(path, 'Clutter'), num_features)
#     noise_label = np.array([0] * len(noise_data))
#
#     humans_data = ReadRadarWindows(os.path.join(path, 'Human'), num_features)
#     humans_label = np.array([1] * len(humans_data))
#
#     nonhumans_data = ReadRadarWindows(os.path.join(path, 'Bike'), num_features)
#     nonhumans_label = np.array([2] * len(nonhumans_data))
#
#     X = np.concatenate([humans_data, nonhumans_data, noise_data])
#     y = np.concatenate([humans_label, nonhumans_label, noise_label])
#
#     # Shuffle
#     return unison_shuffled_copies(X,y)

parser = argparse.ArgumentParser(description='Instance extracttion parameters for EMI data')
parser.add_argument('-l', type=int, default=12, help='Sub-instance length')
parser.add_argument('-s', type=int, default=8, help='Sub-instance stride length')
parser.add_argument('-f', type=int, default=8, help='Feature dimension')
parser.add_argument('-spl', type=float, default=0.2, help='Validation/test split')

args = parser.parse_args()

base_dir = '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Austere/BuildSys_Demo/Windowed'
list_dirs = ['winlen_256_stride_128']

# EMI data parameters
subinstanceLen = args.l  # 48
subinstanceStride = args.s  # 16
num_features = args.f  # 16

# Train-Val-Test (TVT) split percentage

for dir in list_dirs:
    extractedDir = os.path.join(base_dir, dir)

    humans_path = os.path.join(extractedDir, 'Human')
    nonhumans_path = os.path.join(extractedDir, 'Bike')
    noise_path = os.path.join(extractedDir, 'Clutter')

    humans_data = ReadRadarWindows(humans_path, num_features)
    humans_label = np.array([1] * len(humans_data))
    nonhumans_data = ReadRadarWindows(nonhumans_path, num_features)
    nonhumans_label = np.array([1] * len(nonhumans_data))
    noise_data = ReadRadarWindows(noise_path, num_features)
    noise_label = np.array([0] * len(noise_data))

    X = np.concatenate([humans_data, nonhumans_data, noise_data])
    y = np.concatenate([humans_label, nonhumans_label, noise_label])

    # Shuffle the data
    X, y = unison_shuffled_copies(X, y)

    # Create TVT splits
    # Splitting data into train/test/validation (size of test set = validation set)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=args.spl, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=args.spl * (len(y_train) + len(y_test)) / len(y_train),
                                                      random_state=42)

    # Feature dimension and number of timesteps
    feats = x_train.shape[-1]
    timesteps = x_train.shape[-2]

    # Create EMI data
    outDir = extractedDir + '/%d_%d/' % (subinstanceLen, subinstanceStride)

    print('subinstanceLen', subinstanceLen)
    print('subinstanceStride', subinstanceStride)
    print('Feature_length', feats)
    print('outDir', outDir)
    try:
        os.mkdir(outDir)
    except OSError:
        exit("Could not create %s" % outDir)
    assert len(os.listdir(outDir)) == 0

    # one-hot encoding of labels
    numOutput = 2
    y_train = one_hot(y_train, numOutput)
    y_val = one_hot(y_val, numOutput)
    y_test = one_hot(y_test, numOutput)

    # Save test points before normalization (for testing execution pipeline)
    x_bag_test, y_bag_test = bagData(x_test, y_test, subinstanceLen, subinstanceStride,
                                     numClass=numOutput, numSteps=timesteps, numFeats=feats)
    print('Shape of x_bag_test (unnormalized)', x_bag_test.shape)

    np.save(outDir + '/x_test_unnorm.npy', x_bag_test)


    # Normalize train, test, validation
    x_train = np.reshape(x_train, [-1, feats])
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    # Save training mean and std
    data = {"mean": mean.tolist(), "std": std.tolist()}
    json.dump(data, codecs.open('../../../buildsys_model/train_stats.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    # normalize train
    x_train = x_train - mean
    x_train = x_train / std
    x_train = np.reshape(x_train, [-1, timesteps, feats])

    # normalize val
    x_val = np.reshape(x_val, [-1, feats])
    x_val = x_val - mean
    x_val = x_val / std
    x_val = np.reshape(x_val, [-1, timesteps, feats])

    # normalize test
    x_test = np.reshape(x_test, [-1, feats])
    x_test = x_test - mean
    x_test = x_test / std
    x_test = np.reshape(x_test, [-1, timesteps, feats])

    x_bag_train, y_bag_train = bagData(x_train, y_train, subinstanceLen, subinstanceStride,
                                       numClass=numOutput, numSteps=timesteps, numFeats=feats)
    print('Shape of x_bag_train', x_bag_train.shape)

    np.save(outDir + '/x_train.npy', x_bag_train)
    np.save(outDir + '/y_train.npy', y_bag_train)
    print('Num train %d' % len(x_bag_train))
    x_bag_test, y_bag_test = bagData(x_test, y_test, subinstanceLen, subinstanceStride,
                                       numClass=numOutput, numSteps=timesteps, numFeats=feats)
    print('Shape of x_bag_test', x_bag_test.shape)

    np.save(outDir + '/x_test.npy', x_bag_test)
    np.save(outDir + '/y_test.npy', y_bag_test)
    print('Num test %d' % len(x_bag_test))
    x_bag_val, y_bag_val = bagData(x_val, y_val, subinstanceLen, subinstanceStride,
                                       numClass=numOutput, numSteps=timesteps, numFeats=feats)
    print('Shape of x_bag_val', x_bag_val.shape)
    np.save(outDir + '/x_val.npy', x_bag_val)
    np.save(outDir + '/y_val.npy', y_bag_val)
    print('Num val %d' % len(x_bag_val))


print('\nAll done!')