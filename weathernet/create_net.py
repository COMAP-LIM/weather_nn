import numpy as np 
import glob
import random
import h5py
import matplotlib.pyplot as plt 
import matplotlib

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Flatten 
from keras.layers.convolutional import Conv1D
from keras import optimizers
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from keras.layers import MaxPooling1D

import seaborn as sns


colors_master = ['#173F5F', '#7ea3be', '#20639b', '#b1cce1', '#3caea3', '#a1d8d2', '#f6d55c', '#faedc\
0','#ed553b', '#f2c1ba'] 
font = {'size'   : 15}#, 'family':'serif'}
matplotlib.rc('font', **font)


def train_test_split(path_good, path_bad, seed=24):
    random.seed(seed)

    files_good = glob.glob(path_good + '*.hd5')
    files_bad_all = glob.glob(path_bad + '*.hd5')

    random.shuffle(files_good)
    random.shuffle(files_bad_all)
    
    # Ensure that generated data will not be a part of the validation data
    files_bad_generated = []
    files_bad_original = []

    for i in range(len(files_bad_all)):
        if len(files_bad_all[i]) < (len(path_bad)+38):
            files_bad_original.append(files_bad_all[i])
        else:
            files_bad_generated.append(files_bad_all[i])
    
    files_bad = files_bad_original + files_bad_generated

    n_test_samples = int(0.125*(len(files_good) + len(files_bad)))

    
    if len(files_bad_original) < n_test_samples:
        print('NB: Generated data (%d samples) is used as validation data!' %(n_test_samples - len(files_bad_original)))
    

    labels_good = [0] * len(files_good)
    labels_bad = [1] * len(files_bad)    
    
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
    X_train, ps_train, weather_train, indices_train, obsids_train = read_files(training_data)
    X_test, ps_test, weather_test, indices_test, obsids_test = read_files(testing_data)
    
    std = np.std(X_train)
    X_train = X_train/std
    X_test = X_test/std

    weather_train_max = np.max(weather_train)
    weather_train_min = np.min(weather_train)
    weather_train = (weather_train - weather_train_min)/(weather_train_max - weather_train_min)
    weather_test = (weather_test - weather_train_min)/(weather_train_max - weather_train_min)

    if len(np.shape(X_train)) < 3:
        X_train = X_train.reshape(len(obsids_train), np.shape(X_train)[1],1)
        X_test = X_test.reshape(len(obsids_test), np.shape(X_test)[1],1)


    print('Training samples: %d  (%.1f percent)' %(len(training_labels), 100*len(training_labels)/(len(training_labels) + len(testing_labels))))
    print('Testing samples: %d  (%.1f percent)' %(len(testing_labels), 100*len(testing_labels)/(len(training_labels) + len(testing_labels))))
    print('Of the training samples, %d is bad weather.' %np.sum(training_labels))
    print('Of the testing samples, %d is bad weather.' %np.sum(testing_labels))


    # Convert label array to one-hot encoding  
    y_train = to_categorical(training_labels, 2)
    y_test = to_categorical(testing_labels, 2)

    return X_train, y_train, ps_train, weather_train, X_test, y_test, ps_test, weather_test, indices_test, obsids_test, std



def read_files(files):
    data = []
    power_spectra =  []
    obsids = []
    indices = []
    weathers = []

    for filename in files:
        with h5py.File(filename, 'r') as hdf:
            tod     = np.array(hdf['tod'])
            ps      = np.array(hdf['ps'])
            obsid   = np.array(hdf['obsid'])
            index   = np.array(hdf['index'])
            try:
                weather = np.array(hdf['weather']) 
            except:
                weather = 0

            data.append(tod)
            power_spectra.append(ps)
            obsids.append(obsid)
            indices.append(index)
            weathers.append(weather)

    return np.array(data), np.array(power_spectra), np.array(weathers), indices, obsids


def evaluate_CNN(X_train, y_train, X_test, y_test, std, params = False, best_params = False, save_model=False):
    if params == False:
        params = {'epochs': 150, 'batch_size': 256, 'patience': 20, 'lr': 1e-4, \
                  'filters': 32, 'kernel_size': 6, 'activation1': 'relu', 'activation2': 'relu', 'poolsize': 3} 
    if best_params == True:
        params = {'epochs': 1000, 'batch_size': 256, 'patience': 100, 'lr': 1e-5, \
                  'activation1': 'relu', 'activation2': 'relu', 'poolsize': 3} 

    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    adam = optimizers.Adam(lr=params['lr']) 
    es = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)

    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=24, activation=params['activation1'], input_shape=(n_timesteps,n_features)))
    model.add(MaxPooling1D(pool_size = 3))
    model.add(Conv1D(filters=64, kernel_size=12, activation=params['activation2']))
    model.add(Flatten())
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # fit network                                    
    history1 = model.fit(X_train, y_train, epochs=2, batch_size=params['batch_size'],  verbose=1, validation_data=(X_test, y_test))
    history2 = model.fit(X_train, y_train, epochs=params['epochs']-2, batch_size=params['batch_size'],  verbose=1, validation_data=(X_test, y_test), callbacks=[es])

    
    train_accuracy = history1.history['accuracy'] + history2.history['accuracy'][:-params['patience']]
    test_accuracy = history1.history['val_accuracy'] + history2.history['val_accuracy'][:-params['patience']]
    train_loss = history1.history['loss'] + history2.history['loss'][:-params['patience']]
    test_loss = history1.history['val_loss'] + history2.history['val_loss'][:-params['patience']]
    
        
    history =  {'accuracy': train_accuracy, 'val_accuracy': test_accuracy, 'loss': train_loss, 'val_loss': test_loss}

    predictions = model.predict(X_test)
    y_pred = predictions.argmax(axis=-1)
    y_true = y_test.argmax(axis=-1)        
    recall = recall_score(y_true, y_pred)


    if save_model:
        model.save(save_model)
        np.save(save_model[:-3] + '_history.npy', history)
        with open(save_model[:-3] + '_std.txt', 'w') as f:
            f.write(str(std))
        with open(save_model[:-3] + '_patience.txt', 'w') as f:
            f.write(str(params['patience']))
        print("Saved model to disk")

    return model, history, recall


            
def plot_history(history, save=False, patience=False):
    epochs = np.arange(1,len(history['accuracy'])+1)

    plt.figure(figsize=(10,8))

    ax = plt.subplot(2,1,1)
    if patience:
        plt.axvline(x=epochs[-1]-patience, color=colors_master[8], linestyle='--')
        plt.text(epochs[-1]-patience*1.08, (np.max(history['accuracy'])-np.min(history['accuracy']))/2 + np.min(history['accuracy']), 'Early stop',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        color = colors_master[8])


    plt.plot(epochs, np.array(history['accuracy']), label='Training data', color=colors_master[2])
    plt.plot(epochs, np.array(history['val_accuracy']), label='Validation data', color=colors_master[4])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.grid()
    plt.title('Model accuracy')
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))

    plt.subplot(2,1,2)
    if patience:
        plt.axvline(x=epochs[-1]-patience, color= colors_master[8], linestyle='--')
        plt.text(epochs[-1]-patience*1.08, (np.max(history['loss'])-np.min(history['loss']))*0.5 + np.min(history['loss']), 'Early stop',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        color = colors_master[8])
    plt.plot(epochs, history['loss'], label='Training data', color=colors_master[2])
    plt.plot(epochs, history['val_loss'], label='Validation data', color=colors_master[4])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(loc='best')
    plt.tight_layout()
    if save:
        plt.savefig('figures/history_accuracy_loss.pdf')
    plt.show()


def mean_accuracy(good_samples_folder, bad_samples_folder, runs=10):
    test_accuracies = []
    train_accuracies = []
    test_losses = []
    train_losses = []
    recalls = []

    for r in range(runs):
        print('RUN:', r+1)
        X_train, y_train, ps_train, weather_train, X_test, y_test, ps_test, weather_test, indices_test, obsids_test, std = train_test_split(good_samples_folder, bad_samples_folder, seed=r+1)
        model, history, recall = evaluate_CNN(X_train, y_train, X_test, y_test, std)
       
        train_accuracy = history['accuracy'][-1]
        test_accuracy = history['val_accuracy'][-1]
        test_loss = history['val_loss'][-1]
        train_loss = history['loss'][-1]

        test_accuracies.append(test_accuracy*100.0)
        train_accuracies.append(train_accuracy*100.0)
        test_losses.append(test_loss)
        train_losses.append(train_loss)
        recalls.append(recall)
       
 
    m_test, std_test = np.mean(test_accuracies), np.std(test_accuracies)
    m_train, std_train = np.mean(train_accuracies), np.std(train_accuracies)
    m_recall, std_recall = np.mean(recalls), np.std(recalls)
    m_test_loss, std_test_loss = np.mean(test_losses), np.std(test_losses)
    m_train_loss, std_train_loss = np.mean(train_losses), np.std(train_losses)
    
    print(train_accuracies)
    print(test_accuracies)
    print(recalls)

    print()
    print('Train accuracy: %.3f\%% ($\pm$ %.3f)' % (m_train, std_train))
    print('Test accuracy:  %.3f\%% ($\pm$ %.3f)' % (m_test, std_test))
    print('Train loss:     %.4f   ($\pm$ %.3f)' % (m_train_loss, std_train_loss))
    print('Test loss:      %.4f   ($\pm$ %.3f)' % (m_test_loss, std_test_loss))    
    print('Recall:         %.4f   ($\pm$ %.3f)' % (m_recall, std_recall))
    print()


def plot_recall(model, X_train, y_train, X_test, y_test, save=False):
    cutoff = np.linspace(0,1, 1000)
    
    predictions_test = model.predict(X_test)
    y_true_test = y_test.argmax(axis=-1)
    y_pred_test = np.zeros(len(y_true_test))  
    recall_test = np.zeros(len(cutoff))
    accuracy_test = np.zeros(len(cutoff))
    for j in range(len(cutoff)):
        for i, predicted_test in enumerate(predictions_test): 
            if predicted_test[1] > cutoff[j]:         
                y_pred_test[i] = 1 
            else:      
                y_pred_test[i] = 0  
        recall_test[j] = recall_score(y_true_test, y_pred_test)
        accuracy_test[j] = accuracy_score(y_true_test, y_pred_test)

    predictions_train = model.predict(X_train)
    y_true_train = y_train.argmax(axis=-1)
    y_pred_train = np.zeros(len(y_true_train))  
    recall_train = np.zeros(len(cutoff))
    accuracy_train = np.zeros(len(cutoff))
    for j in range(len(cutoff)):
        for i, predicted_train in enumerate(predictions_train): 
            if predicted_train[1] > cutoff[j]:         
                y_pred_train[i] = 1 
            else:      
                y_pred_train[i] = 0  
        recall_train[j] = recall_score(y_true_train, y_pred_train)
        accuracy_train[j] = accuracy_score(y_true_train, y_pred_train)


    plt.figure(figsize=(6,4))
    plt.plot(cutoff, recall_train, label='Training recall', color=colors_master[2])
    plt.plot(cutoff, recall_test, label='Validation recall', color=colors_master[4])
    plt.plot(cutoff, accuracy_train, label='Training accuracy', color=colors_master[8])
    plt.plot(cutoff, accuracy_test, label='Validation accuracy', color=colors_master[6])
    plt.ylabel('Recall / Accuracy')
    plt.xlabel('Cutoff value')
    plt.ylim(0.7, 1.04)
    plt.grid()
    plt.legend(loc='lower center')
    plt.tight_layout()
    if save:
        plt.savefig('figures/accuracy_recall.pdf')
    plt.show()
    


if __name__ == '__main__':
    good_samples_folder = 'data/good_data/'
    bad_samples_folder = 'data/bad_data/' 

    #mean_accuracy(good_samples_folder, bad_samples_folder, runs=10)    

    X_train, y_train, ps_train, weather_train, X_test, y_test, ps_test, weather_test, indices_test, obsids_test, std = train_test_split(good_samples_folder, bad_samples_folder, seed=2) 
    model, history, recall = evaluate_CNN(X_train, y_train, X_test, y_test, std, save_model='saved_nets/weathernet_TEST.h5', best_params=True) 
