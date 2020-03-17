import numpy as np
import os, glob, sys
import random
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

import scipy as sp
import scipy.interpolate

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from keras.utils.vis_utils import plot_model

def create_dataset(path_good, path_bad, n=False):
    random.seed(24)
    
    files_good = glob.glob(path_good + '*.hd5')
    files_bad = glob.glob(path_bad + '*.hd5')
    labels_good = [0] * len(files_good)
    labels_bad = [1] * len(files_bad)

    random.shuffle(files_good)
    random.shuffle(files_bad)

    # Do not want to use more than 25 % of the bad data as testing data
    n_test_samples = int(0.125*(len(files_good) + len(files_bad)))
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

    std = np.std(X_train)
    X_train = X_train/std
    X_test = X_test/std

    print('Standard deviation of training data:', std)

    X_train = X_train.reshape(len(obsids_train), np.shape(X_train)[1],1)
    X_test = X_test.reshape(len(obsids_test), np.shape(X_test)[1],1)

    print('Training samples: %d  (%.1f percent)' %(len(training_labels), 100*len(training_labels)/(len(training_labels) + len(testing_labels))))
    print('Testing samples: %d  (%.1f percent)' %(len(testing_labels), 100*len(testing_labels)/(len(training_labels) + len(testing_labels)))) 
    print('Of the testing samples, %d is bad weather.' %np.sum(training_labels))
    print('Of the training samples, %d is bad weather.' %np.sum(testing_labels))


    # Convert label array to one-hot encoding
    y_train = to_categorical(training_labels, 2)
    y_test = to_categorical(testing_labels, 2)

    return X_train, y_train, ps_train, X_test, y_test, ps_test, indices_test, obsids_test, std


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
    

def evaluate_CNN(X_train, y_train, X_test, y_test, std, save_model=False):
    verbose, epochs, batch_size = 1, 60, 128 #64
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    adam = optimizers.Adam(lr=5e-5)

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=6, activation='relu', input_shape=(n_timesteps,n_features)))
    #model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    #model.add(Dropout(0.3)) #0.4
    #model.add(MaxPooling1D(pool_size=3))
    #model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.4))
    #model.add(Dense(16, activation='relu')) ##     
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X_test, y_test))

    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    if save_model:
        model.save("CNN_weathernet.h5")
        with open('CNN_weathernet_std.txt', 'w') as f:
            f.write(str(std))
        print("Saved model to disk")

    return model, accuracy, history
        

def plot_history(history, save=False):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train data', 'Testing data'], loc='upper left')
    plt.grid()
    plt.title('Model accuracy')
    if save:
        plt.savefig('history_accuracies_feb.png')

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
        plt.savefig('history_loss_feb.png')
    plt.show()


def mean_accuracy(runs=10):
    X_train, y_train, ps_train, X_test, y_test, ps_test, indices_test, obsids_test = create_dataset('good_samples/', 'bad_samples/')

    accuracies = []
    for r in range(runs):
        model, accuracy, _ = evaluate_CNN(X_train, y_train, X_test, y_test)
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

    for i in range(20):
        print(obsids_test[i])
        print(index_test[i])
        print(predictions[i])
        print()

    if plot:
        for k in range(len(y_test)):
            ymin = np.min(X_test[k]) -0.5
            ymax = np.max(X_test[k]) +1
            subseq = int(index_test[k][1]/30000)
            if y_pred[k] == 0 and y_true[k] == 1:
                print(obsids_test[k])
                print(y_pred[k], y_true[k])
                print(predictions[k])
                print()
                plt.figure(figsize=(5,3))
                plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k])
                plt.title('ObsID: %s, subsequence: %d' %(obsids_test[k], subseq))
                plt.text(index_test[k][1]-1e4, 1, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[k][1], predictions[k][0]), bbox={'facecolor': 'lightblue', 'alpha': 0.7, 'pad': 10})
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
                plt.text(index_test[k][1]-1e4, 1, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[k][1], predictions[k][0]), bbox={'facecolor': 'lightblue', 'alpha': 0.7, 'pad': 10})
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
    X_train, y_train, ps_train, X_test, y_test, ps_test, indices_test, obsids_test, std = create_dataset('good_samples/', 'bad_samples/')
    model, accuracy, history = evaluate_CNN(X_train, y_train, X_test, y_test, std, save_model=True)
    #mean_accuracy()
