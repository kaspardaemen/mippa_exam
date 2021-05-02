import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from network import train_network_conv_2, train_network_conv_5  , train_simple_network, train_simple_network_as2, train_multi_input_2
from preprocessing import  get_data, load_data, make_conv_ready
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import load_model
from performance import check_model_performance_as1, check_model_performance_as2, check_model_performance_as3, get_binary_results, get_test_probs, get_taskc_results
import statsmodels.api as sm
from keras.utils.vis_utils import plot_model
np.random.seed(42)


def preprocess_general_data ():
    file = 'TrainingValidationData_200k_shuffle.csv'
    data, labels, df = get_data(file)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    np.save(f'data/X_test.npy', X_test)
    np.save(f'data/y_test.npy', y_test)
    np.save(f'data/X_train.npy', X_train)
    np.save(f'data/y_train.npy', y_train)

    y_train_binary =  [ np.array([1-y, y*1]) for y in y_train[:,0] == 1]
    y_test_binary = [np.array([1-y , y*1]) for y in y_test[:, 0] == 1]

    np.save(f'data/y_test_binary.npy', y_test_binary)
    np.save(f'data/y_train_binary.npy', y_train_binary)
def train_simple_as1(model_name):
    dropout = 0.25

    #2 this is TaskA with 2 output classes
    X_train, X_test, y_train, y_test = load_data(binary=True)

    model, hist = train_simple_network(X_train, X_test, y_train, y_test, name=model_name, dropout=dropout)

    hist_df = pd.DataFrame(hist.history)
    hist_df['epoch'] = hist_df.index + 1
    vals = ['accuracy', 'val_accuracy']

    sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
    plt.show()
    sns.lineplot(data=hist_df[['loss', 'val_loss']])
    plt.show()
def train_simple_as2(model_name):
    dropout = 0.25

    # outptut=5 this is TaskB with 5 output classes
    X_train, X_test, y_train, y_test = load_data(binary=False)

    output = y_train.shape[1]

    model, hist = train_simple_network_as2(X_train, X_test, y_train, y_test, name=model_name, dropout=dropout, output=output)


    hist_df = pd.DataFrame(hist.history)
    hist_df['epoch'] = hist_df.index + 1
    vals = ['accuracy', 'val_accuracy']

    sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
    plt.show()
    sns.lineplot(data=hist_df[['loss', 'val_loss']])
    plt.show()
def train_as1(model_name):
    dropout = 0.25
    events_only = True
    kernel_size = 3

    # classes list: [False, True]
    X_train, X_test, y_train, y_test = load_data(binary=True)

    # don't use met and metPHI data for conv. network


    #transform the data to a proper input for a 1d convolutional network
    X_train = make_conv_ready(X_train)
    X_test = make_conv_ready(X_test)

    model, hist = train_network_conv_2(X_train, X_test, y_train, y_test, kernel_size=kernel_size, name=model_name,
                                    dropout=dropout)


    hist_df = pd.DataFrame(hist.history)
    hist_df['epoch'] = hist_df.index + 1
    vals = ['accuracy', 'val_accuracy']

    sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
    plt.show()
    sns.lineplot(data=hist_df[['loss', 'val_loss']])
    plt.show()
def train_as2(model_name):
    dropout = 0.2

    kernel_size = 3
    batch_size = 128

    # 5 this is TaskB with 5 output classes
    X_train, X_test, y_train, y_test = load_data(binary=False)

    # don't use met data for conv. network
    # transform the data to a proper input for a 1d convolutional network
    X_train = make_conv_ready(X_train)
    X_test = make_conv_ready(X_test)

    model, hist = train_network_conv_5(X_train, X_test, y_train, y_test, kernel_size=kernel_size, name=model_name,
                                    dropout=dropout, output=5)

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
    data, labels, df = get_data(file)

    # don't use met data for conv. network
    indices = range(data.shape[1])
    conv_indices = indices[2:]
    dense_indices = indices[:2]
    conv_data = data[:, conv_indices].reshape(-1, 19, 4)

    dense_data = data[:, dense_indices]

    data = [(dense_data[i], conv_data[i]) for i in range(len(conv_data))]

    X_train, X_test, y_train, y_test =  train_test_split(data, labels, test_size=0.2, random_state=42)

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
def check_performance(model_name, binary=True, conv=False, mc=False):
    model = load_model(f'models/{model_name}.h5')
    _, X_test, _, y_test = load_data(binary)
    if (conv):
        X_test = make_conv_ready(X_test)
    plot_model(model, to_file=f'models/model_{model_name}.png', show_shapes=True, show_layer_names=True)


    if(y_test.shape[1] == 5):
        check_model_performance_as2(X_test, y_test, model)
    elif(y_test.shape[1] == 2):
        check_model_performance_as1(X_test, y_test, model)
#takes an task C model to do task A
def check_performance_as3(model_name, binary=True, conv=False, threshold=.5):

    model = load_model(f'models/{model_name}.h5')

    #plot_model(model, to_file=f'model_{model_name}.png', show_shapes=True, show_layer_names=True)
    _, X_test, _, y_test = load_data(binary)
    if (conv):
        X_test = make_conv_ready(X_test)

    check_model_performance_as3(X_test, y_test, model, threshold)

def plot_combined_perfomances():
    binary_models = ['simple', 'conv_2']
    multiclass_models = ['simple_5', 'conv_5']
    _, X_test, _, y_test_binary = load_data(binary=True)
    _, _, _, y_test_multi = load_data(binary=False)

    results = {}

    #simple_2
    name = 'task a'
    model = load_model(f'models/simple_2.h5')
    fpr, tpr, auc = get_binary_results(X_test, y_test_binary, model )
    results[f'{name}'] = [fpr, tpr, auc]

    #ctask d
    name = 'task d'
    model = load_model(f'models/conv_2.h5')
    fpr, tpr, auc = get_binary_results(make_conv_ready(X_test), y_test_binary, model)
    results[f'{name}'] = [fpr, tpr, auc]

    # simple_5_t20
    name = 'task c'
    model = load_model(f'models/simple_5.h5')
    fpr, tpr, auc = get_taskc_results(X_test, y_test_multi, model, threshold=.5)
    results[f'{name}'] = [fpr, tpr, auc]


    for model_name, values in results.items():
        plt.plot(values[0], values[1], label = f'{model_name}: {values[2]}')
    plt.xlabel('fpr')
    plt.ylabel('tpr')

    plt.legend()
    plt.show()
def produce_test_results(model_name, binary=False, conv=False):
    file = 'ExamData2.csv'
    data, _, df = get_data(file)
    if(conv):
        data = make_conv_ready(data)
    model = load_model(f'models/{model_name}.h5')

    output_probs = get_test_probs(model, data)

    if (binary):
        results = {'prob': [f'4top={x[1]}' for x in output_probs]}
        df = pd.DataFrame.from_dict(results, orient='columns')
        df.to_csv('TEST_RESULTS_BINARY.csv', index=True, header=False)
        return

    classes = ['4top', 'ttbar', 'ttbarHiggs', 'ttbarW', 'ttbarZ']
    results = {}
    #results['event_id'] = range(0, len(output_probs))
    for i in range(len(classes)):
        results[classes[i]] = [f'{classes[i]}={x[i]}' for x in output_probs]
    df = pd.DataFrame.from_dict(results, orient='columns')
    df.to_csv('TEST_RESULTS_MULTIPLE.csv',index=True, header=False)


if __name__ == "__main__":
    preprocess_general_data()

    # --------------------------------------------
    #task A (as1) with 2 output classes
    # --------------------------------------------

    train_simple_as1('simple_2')
    check_performance('simple_2')

    #--------------------------------------------
    #task B (as2) with  5  output classes
    #--------------------------------------------

    train_simple_as2('simple_5')
    check_performance('simple_5', binary=False)

    # --------------------------------------------
    # task C (as2) with  5  output classes
    # --------------------------------------------

    check_performance_as3('simple_5', conv=False, binary=False, threshold=.25)


    #as3: --> task C with 2 output classes
    #TASK D: the conv solutions

    train_as1('conv_2')
    check_performance('conv_2', binary=True, conv=True)


    plot_combined_perfomances()
    produce_test_results('simple_5', binary=False)

