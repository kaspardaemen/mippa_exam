import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



def onehot_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    classes_list = list(le.classes_)
    n_classes = len(classes_list)  # 11
    labels_le = le.transform(labels)
    labels = np_utils.to_categorical(labels_le, n_classes)
    print(f'classes list: {classes_list}')
   
    return labels

def build_neural_network(data_size_in):
 
    model = Sequential()
    model.add(Dense(20, input_dim=data_size_in, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def get_data (file):
    #dataframe of [event ID, process ID, weight]
    df = pd.read_csv(file, sep=';' , header = None, usecols = range(0,3))
    df.columns = ['event_id', 'process_id', 'event_weight']
    
    f = open(file, "r")
    data = []
    
    for line in f.readlines():
        
        #append MET and METphi
        met_fs = np.array(line.split(';')[3:5])
       
        #append low-level features
        low_level = [x for x in line.split(';')[5:-1]]
        low_lvl_fs = np.array([x.split(',')[1:] for x in low_level]).reshape(-1)
        features = np.concatenate((met_fs, low_lvl_fs))
        
        data.append(features)
        
    max_length = np.max([len(x) for x in data])
    
    #pad data
    padded_data = [] 
    for x in data:
        padded_data.append(np.pad(x, (0, max_length - len(x)), mode = 'constant' ))
       
    #scale data
    scaler = preprocessing.MinMaxScaler(copy=False)
    scaler.fit(padded_data)
    transformed_data = scaler.transform(padded_data)  
    
    #labels
    labels = np.array(df['process_id'])
    foreground = labels == '4top'
    
    binary_labels = foreground*1
    print(f'number of foreground samples: {len(binary_labels[binary_labels == 1])}\nnumber of background samples: {len(binary_labels[binary_labels == 0])}')
    return transformed_data, binary_labels  ,df

# =============================================================================
# MAIN
# =============================================================================

file = 'TrainingValidationData_200k_shuffle.csv'
data, labels, df = get_data(file)
n_epochs = 6

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = build_neural_network(X_train.shape[1])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=n_epochs, batch_size=128, validation_data=(X_test, y_test))

plt.plot(range(1,n_epochs+1), hist.history['accuracy'], label='Train')
plt.plot(range(1,n_epochs+1), hist.history['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


