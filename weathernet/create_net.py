import numpy as np 
import glob
import random
import h5py
import matplotlib.pyplot as plt 

from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Flatten 
from keras.layers.convolutional import Conv1D
from keras import optimizers
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.models import load_model

def train_test_split(path_good, path_bad, n=False):
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

    if len(np.shape(X_train)) < 3:
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

def evaluate_CNN(X_train, y_train, X_test, y_test, std, save_model=False):
    verbose, epochs, batch_size = 1, 90, 128                                      
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    adam = optimizers.Adam(lr=5e-5)

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=6, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Flatten())
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # fit network                                    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X_test, y_test))

    # evaluate model                                                                    
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) 

    if save_model:
        model.save("weathernet_only_mean.h5")
        with open('weathernet_only_mean_std.txt', 'w') as f:
            f.write(str(std))
        print("Saved model to disk")

    return model, accuracy, history


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



if __name__ == '__main__':
    X_train, y_train, ps_train, X_test, y_test, ps_test, indices_test, obsids_test, std = train_test_split('data/training_data_preprocess/good_two_means/', 'data/training_data_preprocess/bad_two_means/')
    model, accuracy, history = evaluate_CNN(X_train, y_train, X_test, y_test, std, save_model=False)
    #model = load_model('weathernet_current.h5')
    analyse_classification_results(model, X_test, y_test, indices_test, obsids_test, plot=False)
