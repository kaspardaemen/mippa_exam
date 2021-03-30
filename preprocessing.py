from sklearn import preprocessing
from keras.utils import np_utils
import numpy as np
import pandas as pd


def onehot_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    classes_list = list(le.classes_)
    n_classes = len(classes_list)
    labels_le = le.transform(labels)
    labels = np_utils.to_categorical(labels_le, n_classes)
    print(f'classes list: {classes_list}')
    return labels


def get_data_a(file):
    # dataframe of [event ID, process ID, weight]
    df = pd.read_csv(file, sep=';', header=None, usecols=range(0, 3))
    df.columns = ['event_id', 'process_id', 'event_weight']

    f = open(file, "r")
    data = []

    for line in f.readlines():
        # append MET and METphi
        met_fs = np.array(line.split(';')[3:5])

        # append low-level features
        low_level = [x for x in line.split(';')[5:-1]]
        low_lvl_fs = np.array([x.split(',')[1:] for x in low_level]).reshape(-1)
        features = np.concatenate((met_fs, low_lvl_fs))

        data.append(features)

    max_length = np.max([len(x) for x in data])

    # pad data
    padded_data = []
    for x in data:
        padded_data.append(np.pad(x, (0, max_length - len(x)), mode='constant'))

    # scale data
    scaler = preprocessing.MinMaxScaler(copy=False)
    scaler.fit(padded_data)
    transformed_data = scaler.transform(padded_data)

    # labels
    labels = np.array(df['process_id'])
    foreground = labels == '4top'

    #transform from boolean to vector
    binary_labels = onehot_labels(foreground )

    print(
        f'number of foreground samples: {len(binary_labels[binary_labels == 1])}\nnumber of background samples: {len(binary_labels[binary_labels == 0])}')
    return transformed_data, binary_labels, df