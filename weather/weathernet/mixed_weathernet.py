import numpy as np 
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import load_model

from CNN_weathernet import read_data
from NN_weathernet import read_weather_data
from split_train_test import split_train_test 


def load_dataset(random=False):
    """
    Loads in the dataset for the one dimensional CNN (the tod) and the dataset for
    the NN (the weather data).
    """

    if random:
        split_train_test('training_data_random.txt', 'testing_data_random.txt')
        X_train_CNN, y_train, index_train, obsids_train = read_data('data/training_data_random.txt')
        X_test_CNN, y_test, index_test, obsids_test = read_data('data/testing_data_random.txt')
        
        X_test_NN, y_test, index_test, obsids_test = read_weather_data('data/testing_data_random.txt')
        X_test_NN, y_test, index_test, obsids_test = read_weather_data('data/testing_data_random.txt')
        
    else:
        print('Reading in data')
        X_train_CNN, y_train, index_train, obsids_train = read_data('data/training_data.txt')
        X_test_CNN, y_test, index_test, obsids_test = read_data('data/testing_data.txt')

        X_train_NN, y_train, index_train, obsids_train = read_data('data/training_data.txt')
        X_test_NN, y_test, index_test, obsids_test = read_data('data/testing_data.txt')


    print('Training samples:', len(y_train))
    print('Testing samples:', len(y_test))

    X_train_CNN = X_train_CNN.reshape(len(y_train), np.shape(X_train_CNN)[1],1)
    X_test_CNN = X_test_CNN.reshape(len(y_test), np.shape(X_test_CNN)[1],1)

    # Convert label array to one-hot encoding                                                                                     
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    return X_train_CNN, X_test_CNN, X_train_NN, X_test_NN, y_train, y_test, index_test, obsids_test


def mixed_weathernet():    
    verbose, epochs, batch_size = 1, 30, 64
    X_train_CNN, X_test_CNN, X_train_NN, X_test_NN, y_train, y_test, index_test, obsids_test = load_dataset()

    input_NN = Input(np.shape(X_train_NN)[1:])
    input_CNN = Input(np.shape(X_train_CNN)[1:])
    n_outputs = y_train.shape[1]

    adam = optimizers.Adam(lr=1e-4)

    x = Dense(32, activation='relu')(input_NN)
    x = Dense(16, activation='relu')(x)
    x = Model(inputs=input_NN, outputs=x)

    y = Conv1D(filters=32, kernel_size=6, activation='relu')(input_CNN)
    y = Conv1D(filters=64, kernel_size=3, activation='relu')(y)
    y = Dropout(0.4)(y)
    y = MaxPooling1D(pool_size=3)(y)
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(16, activation='relu')(y)
    y = Model(inputs=input_CNN, outputs=y)

    combined = concatenate([x.output, y.output])

    z = Dense(8, activation='relu')(combined)
    z = Dense(n_outputs, activation='softmax')(z)

    model = Model(inputs=[x.input, y.input], outputs=z)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


    history = model.fit([X_train_NN, X_train_CNN], y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=([X_test_NN, X_test_CNN], y_test))

    _, accuracy = model.evaluate([X_test_NN, X_test_CNN], y_test, batch_size=64, verbose=0)
    print(accuracy)

mixed_weathernet() 


