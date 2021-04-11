import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from network import train_network_as1, train_simple_network
from preprocessing import get_data_a1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import load_model
from performance import check_model_performance_as1
import statsmodels.api as sm

def regression():
    file = 'TrainingValidationData_200k_shuffle.csv'
    data, labels, df = get_data_a1(file)

    df = (
        df.assign(lMET=lambda x: np.log(x['MET']))
        .assign(lMETphi=lambda x: np.log(x['METphi']))
        .assign(ftop=lambda x: (x['process_id'] == '4top'))
    )

    X_train, X_test, y_train, y_test = train_test_split(df[['lMET','METphi']], df['ftop'], test_size=0.2, random_state=42)
    logit_model = sm.Logit('ftop ~ event_wight + MET + METphi + lMET', df).fit()
    print(logit_model.summary())
    preds = (logit_model.predict(X_test))>0.5
    accuracy_score(y_test, preds)

def train_simple(model_name):
    file = 'TrainingValidationData_200k_shuffle.csv'
    dropout = 0.25

    # classes list: [False, True]
    data, labels, df = get_data_a1(file)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


    model, hist = train_simple_network(X_train, X_test, y_train, y_test, name=model_name, dropout=dropout)

    np.save(f'X_test_{model_name}.npy', X_test)
    np.save(f'y_test_{model_name}.npy', y_test)

    hist_df = pd.DataFrame(hist.history)
    hist_df['epoch'] = hist_df.index + 1
    vals = ['accuracy', 'val_accuracy']

    sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
    plt.show()
    sns.lineplot(data=hist_df[['loss', 'val_loss']])
    plt.show()


def train_as1(model_name):
    dropout = 0.25
    file = 'TrainingValidationData_200k_shuffle.csv'
    events_only = True
    kernel_size = 3

    # classes list: [False, True]
    data, labels, df = get_data_a1(file)

    # don't use met data for conv. network
    indices = range(data.shape[1])
    if (events_only):
        indices = indices[2:]
    data = data[:, indices].reshape(-1, 19, 4)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model, hist = train_network_as1(X_train, X_test, y_train, y_test, kernel_size=kernel_size, name=model_name,
                                    dropout=dropout)
    np.save(f'X_test_{model_name}.npy', X_test)
    np.save(f'y_test_{model_name}.npy', y_test)

    hist_df = pd.DataFrame(hist.history)
    hist_df['epoch'] = hist_df.index + 1
    vals = ['accuracy', 'val_accuracy']

    sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
    plt.show()
    sns.lineplot(data=hist_df[['loss', 'val_loss']])
    plt.show()

def check_performance(model_name):
    model = load_model(f'{model_name}.h5')
    X_test = np.load(f'X_test_{model_name}.npy')
    y_test = np.load(f'y_test_{model_name}.npy')

    check_model_performance_as1(X_test,y_test, model)


if __name__ == "__main__":
    train_as1('as1')
    check_performance('as1')

