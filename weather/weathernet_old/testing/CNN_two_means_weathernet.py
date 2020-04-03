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
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import load_model
from keras.utils import plot_model

from CNN_weathernet import create_dataset_v2

def read_data(textfile):
    f = open(textfile, 'r')
    lines = f.readlines()
    data = []
    power_spectrum = []
    obsids = []
    labels = []
    index = []

    for line in lines:
        filename = line.split()[0]
        index1 = int(line.split()[1])
        index2 = int(line.split()[2])
        label = int(line.split()[3])
        month = filename[14:21]
        obsid = int(filename[9:13])

        labels.append(label)
        obsids.append(obsid)
        index.append((index1, index2))
        path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/' + month + '/'
        with h5py.File(path + filename, 'r') as hdf:
            tod       = np.array(hdf['spectrometer/band_average'])
            el        = np.array(hdf['spectrometer/pixel_pointing/pixel_el'][0])
            az        = np.array(hdf['spectrometer/pixel_pointing/pixel_az'][0])
            features  = np.array(hdf['spectrometer/features'])
        
        # Removing Tsys measurements
        boolTsys = (features != 8192)
        indexTsys = np.where(boolTsys==False)[0]

        if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
            boolTsys[:np.min(indexTsys)] = False
            boolTsys[np.max(indexTsys):] = False

        tod       = tod[:,:,boolTsys]
        el        = el[boolTsys]
        az        = az[boolTsys]

        # Extracting subsequence
        tod       = tod[:,:,index1:index2]
        el        = el[index1:index2]
        az        = az[index1:index2]

        # Preprocessing
        tod = preprocess_data(tod, el, az, obsids[-1], index[-1])
 
        data.append(tod)

    return np.array(data), np.array(labels), index,  obsids

def remove_elevation_gain(X, g, a, c, d, e):
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2 


def preprocess_data(data, el, az, obsid, index):
    # Normalizing by dividing each feed on its own mean
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            data[i][j] = data[i][j]/np.nanmean(data[i][j])
    
    # Mean over feeds and sidebands
    data_m1 = np.nanmean(data[:int(np.shape(data)[0]/2)], axis=0)
    data_m2 = np.nanmean(data[int(np.shape(data)[0]/2):], axis=0)
    data_m1 = np.nanmean(data_m1, axis=0)
    data_m2 = np.nanmean(data_m2, axis=0)

    data_list = [data_m1, data_m2]
    preprocessed_data = []
    for data in data_list:
        part = int(len(el)/4)
    
        t = np.arange(len(el))
        g = np.zeros(len(el))
        a = np.zeros(len(el))
        std = np.zeros(len(el))
        diff = np.zeros(len(el))
    
        for i in range(4):
            popt, pcov = curve_fit(remove_elevation_gain, (t[part*i:part*(i+1)],el[part*i:part*(i+1)], az[part*i:part*(i+1)]), data[part*i:part*(i+1)])
            g[part*i:part*(i+1)] = popt[0]
            a[part*i:part*(i+1)] = popt[1]
            std[part*i:part*(i+1)] = np.std(data[part*i:part*(i+1)])
            diff[part*i:part*(i+1)] = (data[part*i-1] - g[part*i-1]/np.sin(el[part*i-1]*np.pi/180) - a[part*i-1]*az[part*i-1]) - (data[part*i] - g[part*i]/np.sin(el[part*i]*np.pi/180) - a[part*i]*az[part*i]) + diff[part*(i-1)]


        # Removing elevation gain 
        data = data - g/np.sin(el*np.pi/180) - a*az + diff 

        # Normalizing 
        data = (data - np.mean(data))/np.std(data)
        
        preprocessed_data.append(data)
    

    data = np.vstack(preprocessed_data).T
    
    return data


def load_dataset_fromfile():
    with h5py.File('dataset_new.h5', 'r') as hdf:
            X_train = np.array(hdf['X_train'])
            y_train = np.array(hdf['y_train'])
            X_test  = np.array(hdf['X_test'])
            y_test  = np.array(hdf['y_test'])
            index_test  = np.array(hdf['index_test'])
            obsids_test  = np.array(hdf['obsids_test'])
    
    return X_train, y_train, X_test, y_test, index_test, obsids_test


def load_dataset(random=False):
    if random:
        split_train_test('training_data_random.txt', 'testing_data_random.txt')
        X_train, y_train, ps_train, index_train, obsids_train = read_data('data/training_data_random.txt')
        X_test, y_test, ps_test, index_test, obsids_test = read_data('data/testing_data_random.txt')    
                   
    else:
        X_train, y_train, index_train, obsids_train = read_data('data/training_set_more_good.txt')
        X_test, y_test, index_test, obsids_test = read_data('data/testing_set_more_good.txt')

    print('Training samples:', len(y_train))
    print('Testing samples:', len(y_test))
    print('Of the testing samples, %d is bad weather.' %np.sum(y_test))
    print('Of the training samples, %d is bad weather.' %np.sum(y_train))

    # Convert label array to one-hot encoding
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    with h5py.File('dataset_new_test.h5', 'w') as hdf:
        hdf.create_dataset('X_train', data=X_train)
        hdf.create_dataset('y_train', data=y_train)
        hdf.create_dataset('X_test', data=X_test)
        hdf.create_dataset('y_test', data=y_test)
        hdf.create_dataset('index_test', data=index_test)
        hdf.create_dataset('obsids_test', data=obsids_test)
    

    return X_train, y_train, X_test, y_test, index_test, obsids_test


def evaluate_CNN(X_train, y_train, X_test, y_test, save_model=False):
    verbose, epochs, batch_size = 1, 40, 64#128 #64
    
    print(X_train.shape)
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    adam = optimizers.Adam(lr=1e-4)
    sgd = optimizers.SGD(learning_rate=0.0001)
    
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=6, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu')) ##     
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X_test, y_test))

    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

    if save_model:
        model.save("CNN_weathernet.h5")
        print("Saved model to disk")

    return model, accuracy, history
        

def plot_history(history, cl1, cl2, cl3, num1, num2, num3, save=False):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train data', 'Testing data'], loc='upper left')
    plt.grid()
    plt.title('Model accuracy')
    if save:
        plt.savefig('history_accuracies_cl1_%d_cl2_%d_cl3_%d_num1_%d_num2_%d_num3_%d.png' %(cl1, cl2, cl3, num1, num2, num3))

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.ylim((-0.1, 2))
    plt.legend(['Training data', 'Testing data'], loc='upper left')
    if save:
        plt.savefig('history_loss_cl1_%d_cl2_%d_cl3_%d_num1_%d_num2_%d_num3_%d.png' %(cl1, cl2, cl3, num1, num2, num3))
    #plt.show()


def mean_accuracy(runs=10):
    trainX, trainy, testX, testy, index_test, obsids_test = load_dataset()
    accuracies = []
    for r in range(runs):
        model, accuracy, _ = evaluate_CNN(trainX, trainy, testX, testy)
        analyse_classification_results(model, testX, testy, index_test, obsids_test)

        accuracy = accuracy * 100.0
        print('>#%d: %.3f' % (r+1, accuracy))
        accuracies.append(accuracy)

    print(accuracies)
    m, std = np.mean(accuracies), np.std(accuracies)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, std))


def analyse_classification_results(model, X_test, y_test, index_test, obsids_test, plot=False):
    predictions = model.predict(X_test)
    y_pred = predictions.argmax(axis=-1)
    y_true = y_test.argmax(axis=-1)
    cm = confusion_matrix(y_true, y_pred)
    
    print('\nConfusion matrix:')
    print(cm)
    print('Normalized confusion matrix:')
    print(cm/np.shape(y_test)[0])

    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=0)
    print('Accuracy: ', accuracy)
    
    if plot:
        for k in range(len(y_test)):
            ymin = np.min(X_test[k]) -1
            ymax = np.max(X_test[k]) +2
            subseq = int(index_test[k][1]/30000)
            if y_pred[k] == 0 and y_true[k] == 1:
                plt.figure(figsize=(5,3))
                plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k], alpha=0.8)
                plt.title('ObsID: %s, subsequence: %d' %(obsids_test[k], subseq))
                plt.text(index_test[k][1]-1e4, 2, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[k][1], predictions[k][0]), bbox={'facecolor': 'lightblue', 'alpha': 0.7, 'pad': 10})
                plt.ylim(ymin,ymax)
                plt.grid()
                plt.tight_layout()
                #plt.savefig('figures/fn_%s_%d.png' %(obsids_test[k], subseq))
                plt.show()
    
            if y_pred[k] == 1 and y_true[k] == 0:
                plt.figure(figsize=(5,3))
                plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k], alpha=0.8)
                plt.title('ObsID: %s, subsequence: %d' %(obsids_test[k], subseq))
                plt.text(index_test[k][1]-1e4, 2, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[k][1], predictions[k][0]), bbox={'facecolor': 'lightblue', 'alpha': 0.7, 'pad': 10})
                plt.ylim(ymin,ymax)
                plt.grid()
                plt.tight_layout()
                # plt.savefig('figures/fp_%s_%d.png' %(obsids_test[k], subseq))
                plt.show()


if __name__ == '__main__':
    #X_train, y_train, X_test, y_test, index_test, obsids_test = load_dataset()
    X_train, y_train, X_test, y_test, index_test, obsids_test = load_dataset_fromfile()
    model, accuracy, history = evaluate_CNN(X_train, y_train, X_test, y_test)
    print(accuracy)
    analyse_classification_results(model, X_test, y_test, index_test, obsids_test, plot=True)
