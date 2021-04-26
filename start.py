import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from network import train_network_conv_2, train_network_conv_5  , train_simple_network, train_simple_network_as2, train_multi_input_2
from preprocessing import get_data_a1, get_data_a2, own_train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import load_model
from performance import check_model_performance_as1, check_model_performance_as2, check_model_performance_as3
import statsmodels.api as sm
from keras.utils.vis_utils import plot_model



def train_simple(model_name):
    file = 'TrainingValidationData_200k_shuffle.csv'
    dropout = 0.25

    # classes list: [False, True]
    data, labels, df = get_data_a1(file)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


    model, hist = train_simple_network(X_train, X_test, y_train, y_test, name=model_name, dropout=dropout)


    np.save(f'data/X_test_{model_name}.npy', X_test)
    np.save(f'data/y_test_{model_name}.npy', y_test)

    hist_df = pd.DataFrame(hist.history)
    hist_df['epoch'] = hist_df.index + 1
    vals = ['accuracy', 'val_accuracy']

    sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
    plt.show()
    sns.lineplot(data=hist_df[['loss', 'val_loss']])
    plt.show()

def train_simple_as2(model_name):
    file = 'TrainingValidationData_200k_shuffle.csv'
    dropout = 0.25

    # classes list: [False, True]
    data, labels, df = get_data_a2(file)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    output = labels.shape[1]

    model, hist = train_simple_network_as2(X_train, X_test, y_train, y_test, name=model_name, dropout=dropout, output=output)

    np.save(f'data/X_test_{model_name}.npy', X_test)
    np.save(f'data/y_test_{model_name}.npy', y_test)

    hist_df = pd.DataFrame(hist.history)
    hist_df['epoch'] = hist_df.index + 1
    vals = ['accuracy', 'val_accuracy']

    sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
    plt.show()
    sns.lineplot(data=hist_df[['loss', 'val_loss']])
    plt.show()


def train_as1(model_name):
    dropout = 0.4
    file = 'TrainingValidationData_200k_shuffle.csv'
    events_only = True
    kernel_size = 1

    # classes list: [False, True]
    data, labels, df = get_data_a1(file)

    # don't use met data for conv. network
    indices = range(data.shape[1])
    if (events_only):
        indices = indices[2:]
    data = data[:, indices].reshape(-1, 19, 4)



    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model, hist = train_network_conv_2(X_train, X_test, y_train, y_test, kernel_size=kernel_size, name=model_name,
                                    dropout=dropout)
    np.save(f'data/X_test_{model_name}.npy', X_test)
    np.save(f'data/y_test_{model_name}.npy', y_test)

    hist_df = pd.DataFrame(hist.history)
    hist_df['epoch'] = hist_df.index + 1
    vals = ['accuracy', 'val_accuracy']

    sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
    plt.show()
    sns.lineplot(data=hist_df[['loss', 'val_loss']])
    plt.show()

def train_as2(model_name):
    dropout = 0.4
    file = 'TrainingValidationData_200k_shuffle.csv'
    events_only = True
    kernel_size = 3
    batch_size = 128

    # classes list: [False, True]
    data, labels, df = get_data_a2(file)

    # don't use met data for conv. network
    indices = range(data.shape[1])
    if (events_only):
        indices = indices[2:]
    data = data[:, indices].reshape(-1, 19, 4)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)



    model, hist = train_network_conv_5(X_train, X_test, y_train, y_test, kernel_size=kernel_size, name=model_name,
                                    dropout=dropout, output=labels.shape[1])
    np.save(f'data/X_test_{model_name}.npy', X_test)
    np.save(f'data/y_test_{model_name}.npy', y_test)

    hist_df = pd.DataFrame(hist.history)
    hist_df['epoch'] = hist_df.index + 1
    vals = ['accuracy', 'val_accuracy']

    sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
    plt.show()
    sns.lineplot(data=hist_df[['loss', 'val_loss']])
    plt.show()

def train_mi_2(model_name):
    dropout = 0.2
    file = 'TrainingValidationData_200k_shuffle.csv'
    kernel_size = 3
    batch_size = 128

    # classes list: [False, True]
    data, labels, df = get_data_a2(file)

    # don't use met data for conv. network
    indices = range(data.shape[1])
    conv_indices = indices[2:]
    dense_indices = indices[:2]
    conv_data = data[:, conv_indices].reshape(-1, 19, 4)

    dense_data = data[:, dense_indices]

    data = [(dense_data[i], conv_data[i]) for i in range(len(conv_data))]

    X_train, X_test, y_train, y_test =  train_test_split(data, labels, test_size=0.2)

    model, hist = train_multi_input_2(X_train, X_test, y_train, y_test, kernel_size=kernel_size, name=model_name,
                                    dropout=dropout, output=labels.shape[1])
    np.save(f'data/X_test_{model_name}.npy', X_test)
    np.save(f'data/y_test_{model_name}.npy', y_test)

    hist_df = pd.DataFrame(hist.history)
    hist_df['epoch'] = hist_df.index + 1
    vals = ['accuracy', 'val_accuracy']

    sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
    plt.show()
    sns.lineplot(data=hist_df[['loss', 'val_loss']])
    plt.show()

def check_performance(model_name):
    model = load_model(f'models/{model_name}.h5')
    plot_model(model, to_file=f'models/model_{model_name}.png', show_shapes=True, show_layer_names=True)
    X_test = np.load(f'data/X_test_{model_name}.npy')
    y_test = np.load(f'data/y_test_{model_name}.npy')

    if(y_test.shape[1] == 5):
        check_model_performance_as2(X_test, y_test, model)
    elif(y_test.shape[1] == 2):
        check_model_performance_as1(X_test, y_test, model)

#takes an task C model to do task A
def check_performance_as3(model_name, threshold):

    model = load_model(f'models/{model_name}.h5')
    plot_model(model, to_file=f'model_{model_name}.png', show_shapes=True, show_layer_names=True)
    X_test = np.load(f'data/X_test_{model_name}.npy')
    y_test = np.load(f'data/y_test_{model_name}.npy')

    check_model_performance_as3(X_test, y_test, model, threshold)


if __name__ == "__main__":
    #as1: 2 classes
    #as2: 5 classes

    #train_as1('conv_2')
    check_performance_as3('simple', 0.5)

    #train_mi_2('mi_2')

    #train_as2('balanced_as2')
    #check_performance_as3('balanced_as2')

