from keras.layers import Activation, BatchNormalization,  Dropout, Flatten, Dense, Conv1D, MaxPooling1D
from keras.models import Sequential
import keras



def build_neural_network(data_size_in, dropout):
    model = Sequential()
    model.add(Dense(64, input_dim=data_size_in,  activation='relu'))
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


def build_as1_network(data_size_in, dropout, kernel_size):
    dropout = dropout
    kernel_size = 3


    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu', input_shape=data_size_in))
    model.add(Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=128, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    return model

def train_simple_network(X_train, X_test, y_train, y_test, name="simple", dropout=0.2):

    model = build_neural_network(X_train.shape[1], dropout)

    callbacks_list = [
        # stop the training loop when the val loss did not improve after more than 9 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10
        ),
        # save the weights into best_model.h5 every time the val loss has improved
        keras.callbacks.ModelCheckpoint(
            filepath='{0}.h5'.format(name),
            monitor='val_loss',
            save_best_only=True
        )
    ]

    batch_size = 128
    epochs = 15

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
                        validation_data=(X_test, y_test), verbose=1)

    return model, history


def train_network_as1(X_train, X_test, y_train, y_test, name="model", dropout=0.2, kernel_size=3):
    data_size_in = (19, 4)
    model = build_as1_network(data_size_in, dropout, kernel_size)
    callbacks_list = [
        # stop the training loop when the val loss did not improve after more than 9 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10
        ),
        # save the weights into best_model.h5 every time the val loss has improved
        keras.callbacks.ModelCheckpoint(
            filepath='{0}.h5'.format(name),
            monitor='val_loss',
            save_best_only=True
        )
    ]

    batch_size = 128
    epochs = 15

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
                        validation_data=(X_test, y_test), verbose=1)

    return model, history