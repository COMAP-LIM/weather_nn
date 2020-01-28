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
from keras.utils import plot_model

def read_data(textfile, generate_more_data=False, n=10):
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

        # Calculating power spectrum
        ps = calculate_power_spectrum(tod)

        # Normalizing
        tod = normalize(tod)

        data.append(tod)
        power_spectrum.append(ps)
    
    return np.array(data), np.array(labels), np.array(power_spectrum), index,  obsids

def remove_elevation_gain(X, g, a, c, d, e):
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2 


def preprocess_data(data, el, az, obsid, index):
    # Normalizing by dividing each feed on its own mean
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            data[i][j] = data[i][j]/np.nanmean(data[i][j])
    
    #print(np.shape(data))
    # Mean over feeds and sidebands
    data = np.nanmean(data, axis=0)
    data = np.nanmean(data, axis=0)


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


    """
    if np.max(abs(a/std)) > 0.28:
        f1 = open('data/azimuth_effect.txt')
        lines = f1.readlines()
        new_line = '%s   %.4f   %d   %d\n' %(obsid, np.max(abs(a/std)), index[0], index[1])
        with open('data/azimuth_effect.txt', 'a') as f:
            if new_line not in lines:
                f.write(new_line)
    """

    # Removing elevation gain 
    data = data - g/np.sin(el*np.pi/180) - a*az + diff 

    #data = data[::10]
    return data

def normalize(data):
    data = (data - np.mean(data))/np.std(data)

    return data


def calculate_power_spectrum(data):
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(len(data))

    logbins = np.logspace(-5, -0.2, 10)
    ps_binned, bins = np.histogram(freqs, bins=logbins, weights=ps)
    ps_binned_2, bins = np.histogram(freqs, bins=logbins)

    #plt.plot(ps_binned/ps_binned_2)
    #plt.show()

    return ps_binned/ps_binned_2/1e6#1e18

def load_dataset_fromfile():
    with h5py.File('dataset.h5', 'r') as hdf:
            X_train = np.array(hdf['X_train'])
            y_train = np.array(hdf['y_train'])
            ps_train  = np.array(hdf['ps_train'])
            X_test  = np.array(hdf['X_test'])
            y_test  = np.array(hdf['y_test'])
            ps_test  = np.array(hdf['ps_test'])
            index_test  = np.array(hdf['index_test'])
            obsids_test  = np.array(hdf['obsids_test'])


    return X_train, y_train, ps_train, X_test, y_test, ps_test,  index_test, obsids_test

def create_dataset_v2(path_good, path_bad, n=False):
    random.seed(24)
    
    files_good = glob.glob(path_good + '*.hd5')
    files_bad = glob.glob(path_bad + '*.hd5')
    labels_good = [0] * len(files_good)
    labels_bad = [1] * len(files_bad)

    random.shuffle(files_good)
    random.shuffle(files_bad)

    n_test_samples = int(0.125*(len(files_good) + len(files_bad)))
    # Do not want to use more than 25 % of the bad data as testing data
    if n_test_samples > 0.25*len(files_bad):
        n_test_samples = int(0.25*len(files_bad))


    testing_data = files_good[:n_test_samples] + files_bad[:n_test_samples]
    testing_labels = labels_good[:n_test_samples] + labels_bad[:n_test_samples]
    training_data = files_good[n_test_samples:] + files_bad[n_test_samples:]
    training_labels = labels_good[n_test_samples:] + labels_bad[n_test_samples:]

    # Shuffling the filenames and labels to the same order
    testing_set = list(zip(testing_data, testing_labels))
    random.shuffle(testing_set)
    testing_data, testing_labels = zip(*testing_set)

    training_set = list(zip(training_data, training_labels))
    random.shuffle(training_set)
    training_data, training_labels = zip(*training_set)

    # Read files
    X_train, ps_train, indices_train, obsids_train = read_files(training_data)
    X_test, ps_test, indices_test, obsids_test = read_files(testing_data)

    X_train = X_train.reshape(len(obsids_train), np.shape(X_train)[1],1)
    X_test = X_test.reshape(len(obsids_test), np.shape(X_test)[1],1)

    print('Training samples: %d  (%.1f percent)' %(len(training_labels), 100*len(training_labels)/(len(training_labels) + len(testing_labels))))
    print('Testing samples: %d  (%.1f percent)' %(len(testing_labels), 100*len(testing_labels)/(len(training_labels) + len(testing_labels)))) 
    print('Of the testing samples, %d is bad weather.' %np.sum(training_labels))
    print('Of the training samples, %d is bad weather.' %np.sum(testing_labels))


    # Convert label array to one-hot encoding
    y_train = to_categorical(training_labels, 2)
    y_test = to_categorical(testing_labels, 2)

    return X_train, y_train, ps_train, X_test, y_test, ps_test, indices_test, obsids_test


def read_files(files):
    data = []
    power_spectra =  []
    obsids = []
    indices = []

    for filename in files:
        with h5py.File(filename, 'r') as hdf:
            tod   = np.array(hdf['tod'])
            ps    = np.array(hdf['ps'])
            obsid = np.array(hdf['obsid'])
            index = np.array(hdf['index'])

            data.append(tod)
            power_spectra.append(ps)
            obsids.append(obsid)
            indices.append(index)

    return np.array(data), np.array(power_spectra), indices, obsids


def load_dataset(random=False):
    if random:
        split_train_test('training_data_random.txt', 'testing_data_random.txt')
        X_train, y_train, ps_train, index_train, obsids_train = read_data('data/training_data_random.txt')
        X_test, y_test, ps_test, index_test, obsids_test = read_data('data/testing_data_random.txt')    
                   
    else:
        X_train, y_train, ps_train, index_train, obsids_train = read_data('data/training_set_more_good.txt')
        X_test, y_test, ps_test, index_test, obsids_test = read_data('data/testing_set_more_good.txt')

    print('Training samples:', len(y_train))
    print('Testing samples:', len(y_test))
    print('Of the testing samples, %d is bad weather.' %np.sum(y_test))
    print('Of the training samples, %d is bad weather.' %np.sum(y_train))

    X_train = X_train.reshape(len(y_train), np.shape(X_train)[1],1)
    X_test = X_test.reshape(len(y_test), np.shape(X_test)[1],1)

    # Convert label array to one-hot encoding
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    with h5py.File('dataset.h5', 'w') as hdf:
        hdf.create_dataset('X_train', data=X_train)
        hdf.create_dataset('y_train', data=y_train)
        hdf.create_dataset('X_test', data=X_test)
        hdf.create_dataset('y_test', data=y_test)
        hdf.create_dataset('ps_train', data=ps_train)
        hdf.create_dataset('ps_test', data=ps_test)
        hdf.create_dataset('index_test', data=index_test)
        hdf.create_dataset('obsids_test', data=obsids_test)


    return X_train, y_train, ps_train, X_test, y_test, ps_test, index_test, obsids_test


def evaluate_CNN_with_ps(X_train, y_train, ps_train, X_test, y_test, ps_test):
    verbose, epochs, batch_size = 1, 100, 64

    input_ps = Input(np.shape(ps_train)[1:])
    input_CNN = Input(np.shape(X_train)[1:])
    n_outputs = y_train.shape[1]

    adam = optimizers.Adam(lr=1e-3)#4

    x = Dense(32, activation='relu')(input_ps)
    x = Dense(16, activation='relu')(input_ps)
    x = Dense(6, activation='relu')(input_ps)
    x = Model(inputs=input_ps, outputs=x)

    y = Conv1D(filters=32, kernel_size=6, activation='relu')(input_CNN)
    y = Conv1D(filters=64, kernel_size=3, activation='relu')(y)
    y = Dropout(0.4)(y)
    y = MaxPooling1D(pool_size=3)(y)
    y = Conv1D(filters=64, kernel_size=3, activation='relu')(y)
    y = Flatten()(y)
    y = Model(inputs=input_CNN, outputs=y)

    combined = concatenate([x.output, y.output])
    
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(16, activation='relu')(z)
    predictions = Dense(n_outputs, activation='softmax')(z)

    #model = Model(inputs=[x.input, y.input], outputs=z)
    model = Model(inputs=[input_ps, input_CNN], outputs=predictions)
    plot_model(model, to_file='mixed_input_model.png')
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    history = model.fit([ps_train, X_train], y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=([ps_test, X_test], y_test))
    #history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X_test, y_test))


    _, accuracy = model.evaluate([ps_test, X_test], y_test, batch_size=64, verbose=0)
    #_, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=0)


    return history, accuracy


def evaluate_CNN_with_ps_v2(X_train, y_train, ps_train, X_test, y_test, ps_test):
    #CNN_model, CNN_accuracy, CNN_history = evaluate_CNN(X_train, y_train, X_test, y_test, save_model=True)
    #NN_model, NN_accuracy, NN_history = evaluate_NN(ps_train, y_train, ps_test, y_test, save_model=True)
    CNN_model = load_model('CNN_weathernet.h5')
    NN_model = load_model('NN_weathernet.h5')


    CNN_predictions_train = CNN_model.predict(X_train)
    NN_predictions_train = NN_model.predict(ps_train)
    sum_predictions_train = (CNN_predictions_train + NN_predictions_train)/2

    CNN_predictions_test = CNN_model.predict(X_test)
    NN_predictions_test = NN_model.predict(ps_test)
    sum_predictions_test = (CNN_predictions_test + NN_predictions_test)/2

    predicted_test = np.argmax(sum_predictions_test, axis=1)
    print(predicted_test[:10])
    print(np.argmax(y_test[:10], axis=1))
    
    s = 0
    for i in range(len(predicted_test)):
        if predicted_test[i] == np.argmax(y_test, axis=1)[i]:
            s += 1

    accuracy = s/len(predicted_test)
    print(accuracy)

    return accuracy


def evaluate_NN(ps_train, y_train, ps_test, y_test, save_model=False):
    verbose, epochs, batch_size = 1, 300, 64
    n_timesteps, n_outputs = ps_train.shape[1], y_train.shape[1]
    adam = optimizers.Adam(lr=2e-2)

    model = Sequential()
    model.add(Dense(6, activation='relu', input_shape=(n_timesteps,)))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    # fit network
    history = model.fit(ps_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(ps_test, y_test))

    #predicted_values = model.predict(ps_test)
    
    # evaluate model
    _, accuracy = model.evaluate(ps_test, y_test, batch_size=batch_size, verbose=0)

    if save_model:
        model.save("NN_weathernet.h5")
        print("Saved model to disk")


    return model, accuracy, history
    

def evaluate_CNN(X_train, y_train, X_test, y_test, save_model=False):
    verbose, epochs, batch_size = 1, 15, 64 #256#128 #64
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    adam = optimizers.Adam(lr=1e-4)

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=6, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.4)) #0.4
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
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
                print(obsids_test[k])
                print(y_pred[k], y_true[k])
                print(predictions[k])
                print()
                plt.figure(figsize=(5,3))
                plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k])
                plt.title('ObsID: %s, subsequence: %d' %(obsids_test[k], subseq))
                plt.text(index_test[k][1]-1e4, 2, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[k][1], predictions[k][0]), bbox={'facecolor': 'lightblue', 'alpha': 0.7, 'pad': 10})
                plt.ylim(ymin,ymax)
                plt.grid()
                plt.tight_layout()
                #plt.savefig('figures/fn_%s_%d.png' %(obsids_test[k], subseq))
                plt.show()
    
            if y_pred[k] == 1 and y_true[k] == 0:
                print(obsids_test[k])
                print(y_pred[k], y_true[k])
                print(predictions[k])
                print()
                plt.figure(figsize=(5,3))
                plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k])
                plt.title('ObsID: %s, subsequence: %d' %(obsids_test[k], subseq))
                plt.text(index_test[k][1]-1e4, 2, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[k][1], predictions[k][0]), bbox={'facecolor': 'lightblue', 'alpha': 0.7, 'pad': 10})
                plt.ylim(ymin,ymax)
                plt.grid()
                plt.tight_layout()
                # plt.savefig('figures/fp_%s_%d.png' %(obsids_test[k], subseq))
                plt.show()


def heatmap_convolving_layers():
    X_train, y_train, X_test, y_test, index_test, obsids_test = load_dataset(random=False)

    cl1_list = [3,6]
    cl2_list = [3,6]
    cl3_list = [3,6]
    
    num_cl1_list = [16,32,64]
    num_cl2_list = [16,32,64]
    num_cl3_list = [16,32,64]

    results = []
    for cl1 in cl1_list:
        for cl2 in cl2_list:
            for cl3 in cl3_list:
                for num_cl1 in num_cl1_list:
                    for num_cl2 in num_cl2_list:
                        for num_cl3 in num_cl3_list:
                            _, accuracy, history = evaluate_CNN(X_train, y_train, X_test, y_test, \
                                                          cl1, cl2, cl3, num_cl1, num_cl2, num_cl3)
                            results.append((cl1, cl2, cl3, num_cl1, num_cl2, num_cl3, accuracy))
                            print(cl1, cl2, cl3, num_cl1, num_cl2, num_cl3, accuracy)
                            plot_history(history, cl1, cl2, cl3, num_cl1, num_cl2, num_cl3, save=True)

    for i in range(len(results)):
        print(results[i])
    

    

if __name__ == '__main__':
    #X_train, y_train, ps_train, X_test, y_test, ps_test, indices_test, obsids_test = load_dataset()
    #X_train, y_train, ps_train, X_test, y_test, ps_test, index_test, obsids_test = load_dataset_fromfile()
    #history, accuracy = evaluate_CNN_with_ps_v2(X_train, y_train, ps_train, X_test, y_test, ps_test)
    #history, accuracy = evaluate_CNN_with_ps(X_train, y_train, ps_train, X_test, y_test, ps_test)
    #model, history, accuracy = evaluate_CNN(X_train, y_train, X_test, y_test)
    #model, accuracy, history = evaluate_NN(ps_train, y_train, ps_test, y_test, save_model=False)
    #print(accuracy)

    X_train, y_train, ps_train, X_test, y_test, ps_test, indices_test, obsids_test = create_dataset_v2('good_samples/', 'bad_samples/')
    #model = load_model('CNN_weathernet.h5')
    model, history, accuracy = evaluate_CNN(X_train, y_train, X_test, y_test)
    analyse_classification_results(model, X_test, y_test, indices_test, obsids_test, plot=True)

    # Ser ut som at jeg kan ha vært litt upresis på klassifiseringen jeg har gjort by-eye. Ser ut som 
    # noen samples med ca like mye struktur er labeled forskjellig. Burde ta hensyn til power scale. 
    # Var også av ca 26 samples, 1 eller 2 spike-samples som ble klassifisert som vær. 
