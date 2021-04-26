
from sklearn import preprocessing
from keras.utils import np_utils
import numpy as np
import pandas as pd

def make_conv_ready(data):
    indices = range(data.shape[1])
    indices = indices[2:]
    transformed_data = data[:, indices].reshape(-1, 19, 4)
    return transformed_data


def load_data(binary=True):
    X_test = np.load(f'data/X_test.npy')
    X_train = np.load(f'data/X_train.npy')

    if(binary):
        y_test = np.load(f'data/y_test_binary.npy')
        y_train = np.load(f'data/y_train_binary.npy')
        return X_train, X_test, y_train, y_test
    y_test = np.load(f'data/y_test.npy')
    y_train = np.load(f'data/y_train.npy')
    return X_train, X_test, y_train, y_test

def onehot_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    classes_list = list(le.classes_)
    n_classes = len(classes_list)
    labels_le = le.transform(labels)
    labels = np_utils.to_categorical(labels_le, n_classes)
    print(f'classes list: {classes_list}')
    return labels
def process_data(file):
    # dataframe of [event ID, process ID, weight]
    df = pd.read_csv(file, sep=';', header=None, usecols=range(0, 5))
    df.columns = ['event_id', 'process_id', 'event_weight', 'MET', 'METphi']

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



    #max_length = np.max([len(x) for x in data])
    max_length = 78 #hacky, but otherwise the models cannot handle the test data

    # pad data
    padded_data = []
    for x in data:
        padded_data.append(np.pad(x, (0, max_length - len(x)), mode='constant'))

    # scale data
    scaler = preprocessing.MinMaxScaler(copy=False)
    scaler.fit(padded_data)
    transformed_data = scaler.transform(padded_data)

    return transformed_data, df


def get_data(file):
    # data
    transformed_data, df = process_data(file)

    # labels
    labels = np.array(df['process_id'])
    # transform from boolean to vector
    one_hot_labels = onehot_labels(labels)

    print(
        f'number of foreground samples: {len(one_hot_labels[one_hot_labels[:,0] == 1])}\nnumber of background samples: {len(one_hot_labels[one_hot_labels[:,0] == 0])}')
    return transformed_data, one_hot_labels, df

