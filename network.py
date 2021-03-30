from keras.layers import Activation, BatchNormalization,  Dropout, Flatten, Dense, Conv1D, MaxPooling1D
from keras.models import Sequential



def build_neural_network(data_size_in, dropout):
    model = Sequential()
    model.add(Dense(64, input_dim=data_size_in, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(128, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(265, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(2, activation='softmax'))

    model.summary()
    return model


def build_conv_network(data_size_in, dropout):

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(19, 4)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    # model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    return model