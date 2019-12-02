import numpy as np
import os, glob, sys
import random
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

import scipy as sp
import scipy.interpolate

from sklearn.metrics import confusion_matrix
from scipy.optimize import curve_fit
from split_train_test import split_train_test

from keras.models import Sequential
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


def read_weather_data(textfile):
    f = open(textfile, 'r')
    lines = f.readlines()
    obsids = []
    index = []
    labels = []
    data = []

    for line in lines:
        filename = line.split()[0]
        index1 = int(line.split()[1])
        index2 = int(line.split()[2])
        label = int(line.split()[3])
        month = filename[14:21]
        obsid = filename[9:13]

        labels.append(label)
        obsids.append(obsid)
        index.append((index1, index2))
        path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/' + month + '/'
        with h5py.File(path + filename, 'r') as hdf:
            temp      = np.array(hdf['hk/array/weather/airTemperature'])
            dewpoint  = np.array(hdf['hk/array/weather/dewPointTemp'])
            pressure  = np.array(hdf['hk/array/weather/pressure'])
            rain      = np.array(hdf['hk/array/weather/rainToday'])
            humidity  = np.array(hdf['hk/array/weather/relativeHumidity'])
            status    = np.array(hdf['hk/array/weather/status'])
            winddeg   = np.array(hdf['hk/array/weather/windDirection'])
            windspeed = np.array(hdf['hk/array/weather/windSpeed'])
            features  = np.array(hdf['spectrometer/features'])

            tod_length = np.shape(np.array(hdf['spectrometer/band_average']))[2]
            weather_length = len(temp)

        features = [np.mean(temp), np.mean(dewpoint), np.mean(pressure), np.mean(rain), \
                    np.mean(humidity), np.mean(status), np.mean(winddeg), np.mean(windspeed)]
        
        """
        # Making bolean array for Tsys measurements                
        boolTsys = (features != 8192)
        indexTsys = np.where(boolTsys==False)[0]

        if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
            boolTsys[:np.min(indexTsys)] = False
            boolTsys[np.max(indexTsys):] = False

        # Interpolating to get same length as tod and removing Tsys measurements
        x = np.linspace(0, tod_length, weather_length)
        x_new = np.linspace(0, tod_length, tod_length)

        temp_new = sp.interpolate.interp1d(x, temp)(x_new)[boolTsys]
        dewpoint_new = sp.interpolate.interp1d(x, dewpoint)(x_new)[boolTsys]
        pressure_new = sp.interpolate.interp1d(x, pressure)(x_new)[boolTsys]
        rain_new = sp.interpolate.interp1d(x, rain)(x_new)[boolTsys]
        humidity_new = sp.interpolate.interp1d(x, humidity)(x_new)[boolTsys]
        status_new = sp.interpolate.interp1d(x, status)(x_new)[boolTsys]
        winddeg_new = sp.interpolate.interp1d(x, winddeg)(x_new)[boolTsys]
        windspeed_new = sp.interpolate.interp1d(x, windspeed)(x_new)[boolTsys]

        # Extracting subsequence and calculating the mean  
        temp_new = np.mean(temp_new[index1:index2])
        dewpoint_new = np.mean(dewpoint_new[index1:index2])
        pressure_new = np.mean(pressure_new[index1:index2])
        rain_new = np.mean(rain_new[index1:index2])
        humidity_new = np.mean(humidity_new[index1:index2])
        status_new = np.mean(status_new[index1:index2])
        winddeg_new = np.mean(winddeg_new[index1:index2])
        windspeed_new = np.mean(windspeed_new[index1:index2])

        features = [temp_new, dewpoint_new, pressure_new, rain_new, humidity_new, \
                     status_new, winddeg_new, windspeed_new]
        """

        data.append(features)

    return np.array(data), np.array(labels), index, obsids


def load_dataset(random=False):
    if random:
        split_train_test('training_data_random.txt', 'testing_data_random.txt')
        X_train, y_train, index_train, obsids_train = read_weather_data('data/training_data_random.txt')
        X_test, y_test, index_test, obsids_test = read_weather_data('data/testing_data_random.txt')
    else:
        X_train, y_train, index_train, obsids_train = read_weather_data('data/training_data.txt')
        X_test, y_test, index_test, obsids_test = read_weather_data('data/testing_data.txt')

    print('Training samples:', len(y_train))
    print('Testing samples:', len(y_test))

    # Convert label array to one-hot encoding
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    return X_train, y_train, X_test, y_test, index_test, obsids_test


def evaluate_NN(X_train, y_train, X_test, y_test, save_model=False):
    verbose, epochs, batch_size = 1, 1500, 64
    n_features, n_outputs = X_train.shape[1], y_train.shape[1]
    adam = optimizers.Adam(lr=1e-5)
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(n_features,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # fit network     
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(X_test, y_test))

    # evaluate model                            
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

    if save_model:
        model.save("weathernet_NN.h5")
        print("Saved model to disk")

    return accuracy, history

def plot_history(history, save_figure = False):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train data', 'Testing data'], loc='upper left')
    plt.grid()
    plt.title('Model accuracy')
    if save_figure:
        plt.savefig('history_accuracies.png')                       

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.ylim((-0.1, 2))
    plt.legend(['Training data', 'Testing data'], loc='upper left')
    if save_figure:
        plt.savefig('history_loss.png')   
    plt.show()


def mean_accuracy(runs=10):
    trainX, trainy, testX, testy, index_test, obsids_test = load_dataset()
    accuracies = []
    for r in range(runs):
        accuracy, _ = evaluate_NN(trainX, trainy, testX, testy)

        accuracy = accuracy * 100.0
        print('>#%d: %.3f' % (r+1, accuracy))
        accuracies.append(accuracy)

    print(accuracies)
    m, std = np.mean(accuracies), np.std(accuracies)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, std))


if __name__ == '__main__':
    mean_accuracy(runs=10)
