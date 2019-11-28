import numpy as np
import os, glob
import random
import h5py
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.optimize import curve_fit


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

random.seed(24)

def read_data(textfile, preprocess=True):
    f = open(textfile, 'r')
    lines = f.readlines()
    data = []
    obsids = []
    labels = []
    index = []

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
            tod      = np.array(hdf['spectrometer/band_average'])
            el       = np.array(hdf['spectrometer/pixel_pointing/pixel_el'][0])
            az       = np.array(hdf['spectrometer/pixel_pointing/pixel_az'][0])
            features = np.array(hdf['spectrometer/features'])
        

        tod = np.nanmean(tod, axis=1)
        tod = np.nanmean(tod, axis=0)

        # Removing Tsys measurements
        boolTsys = (features != 8192)
        indexTsys = np.where(boolTsys==False)[0]

        if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
            boolTsys[:np.min(indexTsys)] = False
            boolTsys[np.max(indexTsys):] = False

        tod = tod[boolTsys]
        el  = el[boolTsys]
        az  = az[boolTsys]

        # Choosing only the subsequence
        tod = tod[index1:index2]
        el  = el[index1:index2]
        az  = az[index1:index2]

        # Preprocessing
        if preprocess:
            tod = preprocess_data(tod, el, az, obsids[-1], index[-1])

        data.append(tod)
    return np.array(data), np.array(labels), index,  obsids


def remove_elevation_gain(X, g, a, c, d, e):
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2 


def preprocess_data(data, el, az, obsid, index):
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
    fig = plt.figure(figsize=(9,3))
    plt.subplot(121)
    plt.suptitle('ObsID: ' + obsid)
    plt.plot(data, label='Data')
    plt.plot(t[:part], remove_elevation_gain((t[:part],el[:part],az[:part]), *popt1), 'r-',  alpha=0.7, linewidth=0.9, label='Fitted function')
    plt.plot(t[part:part*2], remove_elevation_gain((t[part:part*2],el[part:part*2],az[part:part*2]), *popt2), 'r-',  alpha=0.7, linewidth=0.9)
    plt.plot(t[part*2:part*3], remove_elevation_gain((t[part*2:part*3],el[part*2:part*3],az[part*2:part*3]), *popt3), 'r-', alpha=0.7, linewidth=0.9)
    plt.plot(t[-part:], remove_elevation_gain((t[-part:],el[-part:],az[-part:]), *popt4), 'r-',  alpha=0.7, linewidth=0.9)
    plt.xlabel('Sample')
    plt.ylabel('Power')
    plt.grid()
    plt.legend()
    plt.xticks(rotation=30)
    
    plt.subfigure(212)
    plt.plot(data - g/np.sin(el*np.pi/180) - a*az + diff, label="Data'")
    plt.tight_layout()
    plt.xlabel('Sample')
    plt.ylabel('Power') 
    plt.grid()
    plt.legend()
    plt.xticks(rotation=30)
    #fig.subplots_adjust(top=0.90)
    #plt.savefig(obsid + '_az_el_fix.png')
    plt.show()
    """

    if np.max(abs(a/std)) > 0.28:
        f1 = open('data/azimuth_effect.txt')
        lines = f1.readlines()
        new_line = '%s   %.4f   %d   %d\n' %(obsid, np.max(abs(a/std)), index[0], index[1])
        with open('data/azimuth_effect.txt', 'a') as f:
            if new_line not in lines:
                f.write(new_line)


    # Removing elevation gain 
    data = data - g/np.sin(el*np.pi/180) - a*az + diff 

    # Normalizing 
    data = (data - np.mean(data))/np.std(data)
   
    return data



def load_dataset(preprocess=True):
    X_train, y_train, index_train, obsids_train = read_data('data/training_data.txt', preprocess=preprocess)
    X_test, y_test, index_test, obsids_test = read_data('data/testing_data.txt', preprocess=preprocess)

    print('Training samples:', len(y_train))
    print('Testing samples:', len(y_test))

    X_train = X_train.reshape(len(y_train), np.shape(X_train)[1],1)
    X_test = X_test.reshape(len(y_test), np.shape(X_test)[1],1)

    # Convert label array to one-hot encoding
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    return X_train, y_train, X_test, y_test, index_test, obsids_test


# Fit and evaluate a model
def evaluate_model(X_train, y_train, X_test, y_test, save_model=False):
    verbose, epochs, batch_size = 1, 15, 64 
    n_timesteps, n_outputs = X_train.shape[1], y_train.shape[1]
    adam = optimizers.Adam(lr=1e-4)
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=6, activation='relu', input_shape=(n_timesteps,1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
   
    # fit network
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X_test, y_test))

    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

    if save_model:
        model.save("weathernet.h5")
        print("Saved model to disk")

    return accuracy, history


def plot_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train data', 'Testing data'], loc='upper left')
    plt.grid()
    plt.title('Model accuracy')
    #plt.savefig('history_accuracies.png')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.ylim((-0.1, 2))
    plt.legend(['Training data', 'Testing data'], loc='upper left')
    #plt.savefig('history_loss.png')
    plt.show()


def mean_accuracy(runs=10):
    trainX, trainy, testX, testy = load_dataset()
    
    accuracies = []
    for r in range(runs):
        accuracy, _ = evaluate_model(trainX, trainy, testX, testy)
        accuracy = accuracy * 100.0
        print('>#%d: %.3f' % (r+1, accuracy))
        accuracies.append(accuracy)

    print(accuracies)
    m, std = np.mean(accuracies), np.std(accuracies)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, std))


def analyse_classification_results(model, X_test, y_test, index_test, obsids_test):
    predictions = model.predict(X_test)
    y_pred = predictions.argmax(axis=-1)
    y_true = y_test.argmax(axis=-1)
    cm = confusion_matrix(y_true, y_pred)
    
    print('\nConfusion matrix:')
    print(cm)
    print('Normalized confusion matrix:')
    print(cm/np.shape(y_test)[0])

    for k in range(len(y_test)):
        ymin = np.min(X_test[k]) -1
        ymax = np.max(X_test[k]) +2
        subseq = int(index_test[k][1]/30000)
        if y_pred[k] == 0 and y_true[k] == 1:
            plt.figure(figsize=(5,3))
            plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k])
            plt.title('ObsID: %s, subsequence: %d' %(obsids_test[k], subseq))
            plt.text(index_test[k][1]-1e4, 2, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[k][1], predictions[k][0]), bbox={'facecolor': 'lightblue', 'alpha': 0.7, 'pad': 10})
            plt.ylim(ymin,ymax)
            plt.grid()
            plt.tight_layout()
            plt.savefig('figures/fn_%s_%d.png' %(obsids_test[k], subseq))
            plt.show()
    
        if y_pred[k] == 1 and y_true[k] == 0:
            plt.figure(figsize=(5,3))
            plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k])
            plt.title('ObsID: %s, subsequence: %d' %(obsids_test[k], subseq))
            plt.text(index_test[k][1]-1e4, 2, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[k][1], predictions[k][0]), bbox={'facecolor': 'lightblue', 'alpha': 0.7, 'pad': 10})
            plt.ylim(ymin,ymax)
            plt.grid()
            plt.tight_layout()
            plt.savefig('figures/fp_%s_%d.png' %(obsids_test[k], subseq))
            plt.show()


X_train, y_train, X_test, y_test, index_test, obsids_test = load_dataset()
accuracy, _ = evaluate_model(X_train, y_train, X_test, y_test, save_model=True)
print('Accuracy:', accuracy)

# load model
model = load_model('weathernet.h5')
model.summary()
analyse_classification_results(model, X_test, y_test, index_test, obsids_test)
