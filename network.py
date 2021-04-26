from keras.layers import Activation, BatchNormalization,  Dropout, Flatten, Dense, Conv1D, MaxPooling1D
from keras.models import Sequential
import keras
from keras.models import Model
from keras import layers
from keras import Input
from keras.utils.vis_utils import plot_model
from keras_balanced_batch_generator import make_generator

def build_multi_input_2(simple_input, conv_input, dropout, output=2, kernel_size = 3):

    #dense side
    dense_input = Input(shape=(simple_input,), name='dense')
    dense_output = Dense(16, activation="relu")(dense_input)
    dense_output = Dropout(dropout)(dense_output)

    dense_output = Dense(32, activation="relu")(dense_output)
    dense_output = Dropout(dropout)(dense_output)

    dense_output = Dense(64, activation="relu")(dense_output)
    dense_output = Dropout(dropout)(dense_output)

    dense_output = Dense(5, activation="relu")(dense_output)


    #conv side
    conv_input = Input(shape=(conv_input), name='conv')
    conv_output = Conv1D(filters=16, kernel_size=kernel_size, padding='same', activation='relu')(conv_input)
    conv_output = Conv1D(filters=16, kernel_size=kernel_size, padding='same', activation='relu')(conv_output)
    conv_output = Dropout(dropout)(conv_output)
    conv_output = MaxPooling1D(pool_size=2)(conv_output)

    conv_output = Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(conv_output)
    conv_output = Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(conv_output)
    conv_output = Dropout(dropout)(conv_output)
    conv_output = MaxPooling1D(pool_size=2)(conv_output)

    conv_output = Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu')(conv_output)
    conv_output = Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu')(conv_output)
    conv_output = Dropout(dropout)(conv_output)
    conv_output = MaxPooling1D(pool_size=2)(conv_output)

    conv_output = Flatten()(conv_output)
    conv_output = Dense(16, activation='relu')(conv_output)

    concatenated = layers.concatenate([dense_output, conv_output])
    concatenated = layers.Dense(32, activation="relu")(concatenated)
    output = Dense(output, activation='softmax')(concatenated)

    model = Model(inputs=[dense_input, conv_input], outputs=[output])

    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names= True)
    return model
def train_multi_input_2(X_train, X_test, y_train, y_test,  name="simple", dropout=0.2, output=2, kernel_size = 3):

    conv_size_in = (19, 4)
    dense_size_in = 2
    model = build_multi_input_2(dense_size_in, conv_size_in, dropout, output)

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
    epochs = 20

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #unpack values
    X_train_dense = [x[0] for x in X_train]
    X_train_conv = [x[1] for x in X_train]
    X_test_dense = [x[0] for x in X_test]
    X_test_conv = [x[1] for x in X_test]


    history = model.fit([X_train_dense, X_train_conv], y_train,  batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
                        validation_data=([X_test_dense,X_test_conv], y_test), verbose=1)

    return model, history
def build_neural_network(data_size_in, dropout, output=2):

    model = Sequential()
    model.add(Dense(32, input_dim=data_size_in,  activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(32, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(output, activation='softmax'))

    model.summary()

    return model
def build_conv_network(data_size_in, dropout, kernel_size=3, output=2):

    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=kernel_size, padding='same', activation='relu', input_shape=data_size_in))
    model.add(Conv1D(filters=16, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output, activation='softmax'))
    model.summary()
    return model
def train_simple_network(X_train, X_test, y_train, y_test, name="simple", dropout=0.2, output=2):

    model = build_neural_network(X_train.shape[1], dropout, output)

    callbacks_list = [
        # stop the training loop when the val loss did not improve after more than 9 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10
        ),
        # save the weights into best_model.h5 every time the val loss has improved
        keras.callbacks.ModelCheckpoint(
            filepath='models/{0}.h5'.format(name),
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
def train_simple_network_as2(X_train, X_test, y_train, y_test, name="simple", dropout=0.2, output=2):

    model = build_neural_network(X_train.shape[1], dropout, output)

    callbacks_list = [
        # stop the training loop when the val loss did not improve after more than 9 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10
        ),
        # save the weights into best_model.h5 every time the val loss has improved
        keras.callbacks.ModelCheckpoint(
            filepath='models/{0}.h5'.format(name),
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
def train_network_conv_2(X_train, X_test, y_train, y_test, name="model", dropout=0.2, kernel_size=3, output=2):
    data_size_in = (19, 4)
    model = build_conv_network(data_size_in, dropout, kernel_size, output)
    callbacks_list = [
        # stop the training loop when the val loss did not improve after more than 9 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10
        ),
        # save the weights into best_model.h5 every time the val loss has improved
        keras.callbacks.ModelCheckpoint(
            filepath='models/{0}.h5'.format(name),
            monitor='val_loss',
            save_best_only=True
        )
    ]

    batch_size = 128
    epochs = 15

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
                        validation_data=(X_test, y_test), verbose=5)

    return model, history
def train_network_conv_5(X_train, X_test, y_train, y_test, name="model", dropout=0.2, kernel_size=3, output=2):

    data_size_in = (19, 4)
    model = build_conv_network(data_size_in, dropout, kernel_size, output)
    callbacks_list = [
        # stop the training loop when the val loss did not improve after more than 9 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10
        ),
        # save the weights into best_model.h5 every time the val loss has improved
        keras.callbacks.ModelCheckpoint(
            filepath='models/{0}.h5'.format(name),
            monitor='val_loss',
            save_best_only=True
        )
    ]


    batch_size = 128
    epochs = 15


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
                        validation_data=(X_test, y_test), verbose=2)

    return model, history