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

from plot_tod import plot_subsequence


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

    print('Standard deviation of training data:', std)

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


def final_testing_values(path_final_testing_good, path_final_testing_bad, std):
    files_final_good = glob.glob(path_final_testing_good + '*.hd5')
    files_final_bad = glob.glob(path_final_testing_bad + '*.hd5')

    
    labels_good = [0] * len(files_final_good)
    labels_bad = [1] * len(files_final_bad)    

    data = files_final_good + files_final_bad
    labels = labels_good + labels_bad

    
    # Shuffling the filenames and labels to the same order                              
    data_set = list(zip(data, labels))
    random.shuffle(data_set)
    data, labels = zip(*data_set)

    # Read files                                                                        
    X_final_test, ps_final_test, weather_final_test, indices_final_test, obsids_final_test = read_files(data)

    X_final_test = X_final_test/std

    print('Total number of final testing samples: %d' %(len(labels)))
    print('Of the final testing samples, %d is bad weather.' %np.sum(labels))

    # Convert label array to one-hot encoding  
    y_final_test = to_categorical(labels, 2)

    return X_final_test, y_final_test, indices_final_test, obsids_final_test



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
                weather = np.array(hdf['weather']) #[0],hdf['weather'][1],hdf['weather'][2], hdf['weather'][6]])
            except:
                weather = 0

            data.append(tod)
            power_spectra.append(ps)
            obsids.append(obsid)
            indices.append(index)
            weathers.append(weather)

    return np.array(data), np.array(power_spectra), np.array(weathers), indices, obsids


def evaluate_CNN_new(X_train, y_train, X_test, y_test, std, params = False, best_params = False, save_model=False):
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
    #model.add(Conv1D(filters=32, kernel_size=6, activation=params['activation1'], input_shape=(n_timesteps,n_features)))
    model.add(MaxPooling1D(pool_size = 3))
    model.add(Conv1D(filters=64, kernel_size=12, activation=params['activation2']))
    #model.add(Conv1D(filters=16, kernel_size=3, activation=params['activation2']))
    model.add(Flatten())
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #plot_model(model, to_file='default_architecture.pdf', show_shapes=False, show_layer_names=True) 

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

    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) 

    if save_model:
        save_name = 'saved_nets/weathernet_TEST2.h5'
        model.save(save_name)
        np.save(save_name[:-3] + '_history.npy', history)
        with open(save_name[:-3] + '_std.txt', 'w') as f:
            f.write(str(std))
        with open(save_name[:-3] + '_patience.txt', 'w') as f:
            f.write(str(params['patience']))
        print("Saved model to disk")

    return model, history, recall


def grid_search(good_samples_folder, bad_samples_folder, parameter1, parameter2):
    parameter1 = 'lr'
    parameter2 = 'batch_size'

    activation1 = ['relu', 'tanh', 'sigmoid']
    activation2 = ['relu', 'tanh', 'sigmoid']
    lr = [1e-5] #[1e-2, 1e-3, 1e-4, 1e-5]
    bs = [512]   #[64, 128, 256, 512]
    filters = [16, 32, 64, 128, 256]
    kernel_size = [3, 6, 9, 12, 24, 48]
    poolsizes = [3, 6, 9, 12]

    suggested_params = {'batch_size': bs, 'lr': lr, 'filters': filters, 'kernel_size': kernel_size,\
                        'activation1': activation1, 'activation2': activation2, 'poolsize': poolsizes}

    default_params = {'epochs': 1000, 'batch_size': 256, 'patience': 100, 'lr': 1e-4, \
                      'filters': 32, 'kernel_size': 6, 'activation1': 'relu', \
                      'activation2': 'relu', 'poolsize': 3} 


    all_test_accuracies = np.zeros((len(suggested_params[parameter1]), len(suggested_params[parameter2])))
    all_recall = np.zeros((len(suggested_params[parameter1]), len(suggested_params[parameter2])))
    all_train_accuracies = np.zeros((len(suggested_params[parameter1]), len(suggested_params[parameter2])))
    all_test_loss = np.zeros((len(suggested_params[parameter1]), len(suggested_params[parameter2])))
    all_train_loss = np.zeros((len(suggested_params[parameter1]), len(suggested_params[parameter2])))

    results = []
    for i, p1 in enumerate(suggested_params[parameter1]):
        for j, p2 in enumerate(suggested_params[parameter2]):
            print(p1, p2) 

            params = default_params.copy()
            params[parameter1] = p1
            params[parameter2] = p2
                    
            test_accuracies = []
            train_accuracies = []
            train_losses = []
            test_losses = []
            recalls = []

            for r in range(10):
                print('RUN:', r)
                print(p1,p2)
                X_train, y_train, ps_train, weather_train, X_test, y_test, ps_test, weather_test, indices_test, obsids_test, std = train_test_split(good_samples_folder, bad_samples_folder, seed=r+1)
                model, history, recall = evaluate_CNN_new(X_train, y_train, X_test, y_test, std, params)

                test_accuracy = history['val_accuracy'][-1]
                train_accuracy = history['accuracy'][-1]
                test_loss = history['val_loss'][-1]
                train_loss = history['loss'][-1]

                test_accuracies.append(test_accuracy*100.0)
                train_accuracies.append(train_accuracy*100.0)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
                recalls.append(recall)

            m_test = np.mean(test_accuracies)
            m_train = np.mean(train_accuracies)
            m_recall = np.mean(recalls)
            m_test_loss = np.mean(test_losses)
            m_train_loss = np.mean(train_losses)
            
            all_test_accuracies[i,j] = m_test
            all_train_accuracies[i,j] = m_train
            all_test_loss[i,j] = m_test_loss
            all_train_loss[i,j] = m_train_loss
            all_recall[i,j] = m_recall
            
            results.append((p1, p2, m_train, m_test, m_train_loss, m_test_loss, m_recall))


    np.save('figures/heatmaps/separate_values/heatmap_test_accuracy_%s_%s_owl21.npy' %(parameter1, parameter2), all_test_accuracies) 
    np.save('figures/heatmaps/separate_values/heatmap_train_accuracy_%s_%s_owl21.npy' %(parameter1, parameter2), all_train_accuracies) 
    np.save('figures/heatmaps/separate_values/heatmap_test_loss_%s_%s_owl21.npy' %(parameter1, parameter2), all_test_loss) 
    np.save('figures/heatmaps/separate_values/heatmap_train_loss_%s_%s_owl21.npy' %(parameter1, parameter2), all_train_loss) 
    np.save('figures/heatmaps/separate_values/heatmap_recall_%s_%s_owl21.npy' %(parameter1, parameter2), all_recall) 

    for i in range(len(results)):
        print(results[i])


    """
    all_test_accuracies = np.zeros((len(suggested_params[parameter1])))
    all_recall = np.zeros((len(suggested_params[parameter1])))
    all_train_accuracies = np.zeros((len(suggested_params[parameter1])))
    all_test_loss = np.zeros((len(suggested_params[parameter1])))
    all_train_loss = np.zeros((len(suggested_params[parameter1])))

    results = []
    for i, p1 in enumerate(suggested_params[parameter1]):
            print(p1) 
            params = default_params.copy()
            params[parameter1] = p1
                    
            test_accuracies = []
            train_accuracies = []
            train_losses = []
            test_losses = []
            recalls = []

            for r in range(10):
                print('RUN:', r)
                print(p1)
                X_train, y_train, ps_train, weather_train, X_test, y_test, ps_test, weather_test, indices_test, obsids_test, std = train_test_split(good_samples_folder, bad_samples_folder, seed=r+1)
                model, history, recall = evaluate_CNN_new(X_train, y_train, X_test, y_test, std, params)

                test_accuracy = history['val_accuracy'][-1]
                train_accuracy = history['accuracy'][-1]
                test_loss = history['val_loss'][-1]
                train_loss = history['loss'][-1]

                test_accuracies.append(test_accuracy*100.0)
                train_accuracies.append(train_accuracy*100.0)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
                recalls.append(recall)

            m_test = np.mean(test_accuracies)
            m_train = np.mean(train_accuracies)
            m_recall = np.mean(recalls)
            m_test_loss = np.mean(test_losses)
            m_train_loss = np.mean(train_losses)
            
            all_test_accuracies[i] = m_test
            all_train_accuracies[i] = m_train
            all_test_loss[i] = m_test_loss
            all_train_loss[i] = m_train_loss
            all_recall[i] = m_recall
            
            results.append((p1, m_train, m_test, m_train_loss, m_test_loss, m_recall))


    np.save('figures/heatmaps/separate_values/heatmap_test_accuracy_%s_beehive18.npy' %(parameter1), all_test_accuracies) 
    np.save('figures/heatmaps/separate_values/heatmap_train_accuracy_%s_beehive18.npy' %(parameter1), all_train_accuracies) 
    np.save('figures/heatmaps/separate_values/heatmap_test_loss_%s_beehive18.npy' %(parameter1), all_test_loss) 
    np.save('figures/heatmaps/separate_values/heatmap_train_loss_%s_beehive18.npy' %(parameter1), all_train_loss) 
    np.save('figures/heatmaps/separate_values/heatmap_recall_%s_beehive18.npy' %(parameter1), all_recall) 

    """



def evaluate_NN_weather(X_train, y_train, X_test, y_test, save_model=False):
    verbose, epochs, patience, batch_size = 1, 1500, 20, 128
    n_timesteps, n_outputs = X_train.shape[1], y_train.shape[1]
    adam = optimizers.Adam(lr=1e-3)
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    model = Sequential()
    model.add(Dense(6, activation='relu', input_shape=(n_timesteps,)))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # fit network   
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X_test, y_test), callbacks=[es])

    train_accuracy = history.history['accuracy'][:-patience]
    test_accuracy = history.history['val_accuracy'][:-patience]
    train_loss = history.history['loss'][:-patience]
    test_loss = history.history['val_loss'][:-patience]
        
    history =  {'accuracy': train_accuracy, 'val_accuracy': test_accuracy, 'loss': train_loss, 'val_loss': test_loss}

    predictions = model.predict(X_test)
    y_pred = predictions.argmax(axis=-1)
    y_true = y_test.argmax(axis=-1)        
    recall = recall_score(y_true, y_pred)


    if save_model:
        model.save("saved_nets/NN_weathernet_weather_test.h5")
        print("Saved model to disk")

    return model, history, recall

def evaluate_NN_ps(X_train, y_train, X_test, y_test, save_model=False):
    verbose, epochs, patience, batch_size = 1, 3000, 20, 128
    n_timesteps, n_outputs, n_features = X_train.shape[2], y_train.shape[1], X_train.shape[1]
    adam = optimizers.Adam(lr=1e-3)  
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))  #64
    model.add(Dense(128, activation='relu'))  #32
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # fit network   
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X_test, y_test), callbacks=[es])


    train_accuracy = history.history['accuracy'][:-patience]
    test_accuracy = history.history['val_accuracy'][:-patience]
    train_loss = history.history['loss'][:-patience]
    test_loss = history.history['val_loss'][:-patience]
        
    history =  {'accuracy': train_accuracy, 'val_accuracy': test_accuracy, 'loss': train_loss, 'val_loss': test_loss}


    predictions = model.predict(X_test)
    y_pred = predictions.argmax(axis=-1)
    y_true = y_test.argmax(axis=-1)        
    recall = recall_score(y_true, y_pred)


    if save_model:
        model.save("saved_nets/NN_weathernet_ps_test.h5")
        print("Saved model to disk")

    return model, history, recall


def evaluate_mixed(X_train, X2_train, y_train, X_test, X2_test, y_test, CNN_model=False, NN_model=False, save_model=False):
    if CNN_model == False and NN_model == False:
        CNN_model = load_model('saved_nets/weathernet_test.h5')
        #NN_model = load_model('saved_nets/NN_weathernet_ps_test.h5')
        NN_model = load_model('saved_nets/NN_weathernet_weather_test.h5')

    CNN_predictions_train = CNN_model.predict(X_train)
    NN_predictions_train = NN_model.predict(X2_train)
    sum_predictions_train = (CNN_predictions_train + NN_predictions_train)/2

    CNN_predictions_test = CNN_model.predict(X_test)
    NN_predictions_test = NN_model.predict(X2_test)
    sum_predictions_test = (CNN_predictions_test + NN_predictions_test)/2

    verbose, epochs, batch_size = 1, 100, 256
    n_timesteps, n_outputs = X_train.shape[1], y_train.shape[1]
    adam = optimizers.Adam(lr=1e-2) 
    patience = 20
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    model = Sequential()
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # fit network   
    history = model.fit(sum_predictions_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(sum_predictions_test, y_test), callbacks=[es])

    train_accuracy = history.history['accuracy'][:-patience]
    test_accuracy = history.history['val_accuracy'][:-patience]
    train_loss = history.history['loss'][:-patience]
    test_loss = history.history['val_loss'][:-patience]
        
    history =  {'accuracy': train_accuracy, 'val_accuracy': test_accuracy, 'loss': train_loss, 'val_loss': test_loss}

    predictions = model.predict(sum_predictions_test)
    y_pred = predictions.argmax(axis=-1)
    y_true = y_test.argmax(axis=-1)        
    recall = recall_score(y_true, y_pred)


    """
    predicted_test = np.argmax(sum_predictions_test, axis=1)
    print(predicted_test[:10])
    print(np.argmax(y_test[:10], axis=1))
    
    s = 0
    for i in range(len(predicted_test)):
        if predicted_test[i] == np.argmax(y_test, axis=1)[i]:
            s += 1

    accuracy = s/len(predicted_test)
    print(accuracy)
    """
    return model, history, recall




def analyse_classification_results(model, X_test, y_test, index_test, obsids_test, plot=False):
    cutoff = 0.23

    predictions = model.predict(X_test)
    y_true = y_test.argmax(axis=-1)
    #y_pred = predictions.argmax(axis=-1)
    y_pred = np.zeros(len(y_true))
    for i, predicted in enumerate(predictions): 
        if predicted[1] > cutoff:     
            y_pred[i] = 1 
        else:      
            y_pred[i] = 0  

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])

    print('\nConfusion matrix:')
    print(cm)
    print('Normalized confusion matrix:')
    print(cm/np.shape(y_test)[0])

    # evaluate model                                                                     
    #_, accuracy = model.evaluate(X_test, y_test)#, batch_size=64, verbose=0)
    #print('Accuracy: ', accuracy)
    print('Accuracy: ', accuracy_score(y_true, y_pred))
    recall = recall_score(y_true, y_pred)
    print('Recall: ', recall) 

    filenames = {'12321': 'comap-0012321-2020-04-01-034523.hd5', '12440': 'comap-0012440-2020-04-05-093528.hd5', '12465': 'comap-0012465-2020-04-06-054333.hd5', '12439': 'comap-0012439-2020-04-05-082759.hd5', '12551': 'comap-0012551-2020-04-09-053140.hd5', '12320': 'comap-0012320-2020-04-01-023753.hd5'}

    if plot:
        for k in range(len(y_test)):
            ymin = -1.1#np.min(X_test[k]) -0.5
            ymax = 2.1#np.max(X_test[k]) +1
            subseq = int(index_test[k][1]/30000)
            if y_pred[k] == 0 and y_true[k] == 1:
                print(k)
                print(obsids_test[k])
                print(y_pred[k], y_true[k])
                print(predictions[k])
                print()
                plot_subsequence(filenames[str(obsids_test[k])], subseq=subseq)
                plt.subplot(122)
                #plt.figure(figsize=(5,4))
                #plt.subplots_adjust(right=0.6)
                plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k][:,0], color=colors_master[2])
                plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k][:,1], color=colors_master[4])
                plt.suptitle('ObsID: %s, subsequence: %d' %(obsids_test[k], subseq))
                plt.text(index_test[k][1]-1.8e4, 1, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[k][1], predictions[k][0]), bbox={'facecolor': colors_master[3], 'pad': 10},  fontsize=13)
                plt.ylim(ymin,ymax)
                plt.xticks(rotation=30)
                plt.xlabel('Sample')
                plt.grid()
                plt.tight_layout()
                plt.subplots_adjust(top=0.88)
                #plt.savefig('figures/fn_%s_%d.pdf' %(obsids_test[k], subseq))          
                plt.show()

            
            if y_pred[k] == 1 and y_true[k] == 0:
                print(obsids_test[k])
                print(y_pred[k], y_true[k])
                print(predictions[k])
                print()
                plot_subsequence(filenames[str(obsids_test[k])], subseq=subseq)
                plt.subplot(122)
                #plt.figure(figsize=(5,4))
                #plt.subplots_adjust(right=0.6)
                plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k][:,0], color=colors_master[2])
                plt.plot(range(index_test[k][0], index_test[k][1]), X_test[k][:,1], color=colors_master[4])
                plt.suptitle('ObsID: %s, subsequence: %d' %(obsids_test[k], subseq))
                plt.text(index_test[k][1]-1.7e4, 1.3, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[k][1], predictions[k][0]), bbox={'facecolor': colors_master[3], 'pad': 0.5, 'edgecolor': colors_master[2], 'boxstyle': 'round'},  fontsize=13)
                plt.ylim(ymin,ymax)
                plt.xticks(rotation=30)
                plt.xlabel('Sample')
                plt.grid()
                plt.tight_layout()
                plt.subplots_adjust(top=0.88)
                #plt.savefig('figures/fp_%s_%d.pdf' %(obsids_test[k], subseq))          
                plt.show()
            
            
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
    #ax.set_xticklabels([])
    plt.legend(loc='best')
    plt.grid()
    plt.title('Model accuracy')
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    #if save:
    #    plt.savefig('figures/history_accuracies_BEST.pdf')

    #plt.figure()
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
        plt.savefig('figures/history_accuracy_loss_BEST.pdf')
    plt.show()


def mean_accuracy(good_samples_folder, bad_samples_folder, runs=10, weather=False, ps=False, mixed_weather=False, mixed_ps=False):
    test_accuracies = []
    train_accuracies = []
    test_losses = []
    train_losses = []
    recalls = []

    for r in range(runs):
        print('RUN:', r+1)
        X_train, y_train, ps_train, weather_train, X_test, y_test, ps_test, weather_test, indices_test, obsids_test, std = train_test_split(good_samples_folder, bad_samples_folder, seed=r+1)
        print(np.shape(X_train))
        if weather:
            X_test = weather_test
            X_train = weather_train
            model, history, recall = evaluate_NN_weather(X_train, y_train, X_test, y_test)
        elif ps:
            X_test = ps_test
            X_train = ps_train
            model, history, recall = evaluate_NN_ps(X_train, y_train, X_test, y_test)
        elif mixed_weather:
            CNN_model, history, recall = evaluate_CNN_new(X_train, y_train, X_test, y_test, std) 
            NN_model, history, recall = evaluate_NN_weather(weather_train, y_train, weather_test, y_test)
            model, history, recall = evaluate_mixed(X_train, weather_train, y_train, X_test, weather_test, y_test, CNN_model=CNN_model, NN_model=NN_model)
        elif mixed_ps:
            CNN_model, history, recall = evaluate_CNN_new(X_train, y_train, X_test, y_test, std, best_params=True) 
            NN_model, history, recall = evaluate_NN_ps(ps_train, y_train, ps_test, y_test)
            model, history, recall = evaluate_mixed(X_train, ps_train, y_train, X_test, ps_test, y_test, CNN_model=CNN_model, NN_model=NN_model)
        else:
            model, history, recall = evaluate_CNN_new(X_train, y_train, X_test, y_test, std)
       
        train_accuracy = history['accuracy'][-1]
        test_accuracy = history['val_accuracy'][-1]
        test_loss = history['val_loss'][-1]
        train_loss = history['loss'][-1]

        """
        predictions = model.predict(X_test)
        y_pred = predictions.argmax(axis=-1)
        y_true = y_test.argmax(axis=-1)
        cm_old = confusion_matrix(y_true, y_pred, labels=[1,0])

        print('Recall real:', recall_score(y_true, y_pred))
        print('Test accuracy real:',  accuracy_score(y_true, y_pred))
   
        predictions = model.predict(X_train)
        y_pred = predictions.argmax(axis=-1)
        y_true = y_train.argmax(axis=-1)
        print('Train accuracy real:', accuracy_score(y_true, y_pred))


        predictions_test = model.predict(X_test)
        y_true_test = y_test.argmax(axis=-1)    
        y_pred_test = np.zeros(len(y_true_test))
        for i, predicted_test in enumerate(predictions_test):
            if predicted_test[1] > 0.15:
                y_pred_test[i] = 1
            else:
                y_pred_test[i] = 0
                
        cm = confusion_matrix(y_true_test, y_pred_test, labels=[1,0])

        predictions_train = model.predict(X_train)
        y_true_train = y_train.argmax(axis=-1)    
        y_pred_train = np.zeros(len(y_true_train))
        for i, predicted_train in enumerate(predictions_train):
            if predicted_train[1] > 0.15:
                y_pred_train[i] = 1
            else:
                y_pred_train[i] = 0
       
                
        recall = recall_score(y_true_test, y_pred_test)
        test_accuracy = accuracy_score(y_true_test, y_pred_test)
        train_accuracy = accuracy_score(y_true_train, y_pred_train)

        print('Recall new:', recall)
        print('Test accuracy new:', test_accuracy)
        print('Train accuracy new:', train_accuracy)
        
        print()
        print(cm)
        print()
        print(cm_old)
        print()
        print()
        """

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
    #analyse_classification_results(model, X_test, y_test, indices_test, obsids_test, plot=False)


def plot_recall(model, X_train, y_train, X_test, y_test, save=False):
    cutoff = np.linspace(0,1, 1000)
    
    predictions_test = model.predict(X_test)
    y_true_test = y_test.argmax(axis=-1)
    #y_pred_test = predictions_test.argmax(axis=-1)
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
    #y_pred_train = predictions_train.argmax(axis=-1)
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
    #plt.title('')
    plt.ylabel('Recall / Accuracy')
    plt.xlabel('Cutoff value')
    plt.ylim(0.7, 1.04)
    plt.grid()
    plt.legend(loc='lower center')#'best')
    plt.tight_layout()
    if save:
        plt.savefig('figures/accuracy_recall_BEST.pdf')
    plt.show()
    


if __name__ == '__main__':
    good_samples_folder = 'data/training_data_results/two_means/el_az_good_new/'
    bad_samples_folder = 'data/training_data_results/two_means/el_az_bad_new/' 
    final_good_samples_folder = 'data/training_data_results/final_testing/good_new/'
    final_bad_samples_folder = 'data/training_data_results/final_testing/bad_new/'

    #good_samples_folder =  'data/training_data_results/mixed/good/'
    #bad_samples_folder = 'data/training_data_results/mixed/bad/' 
    #mean_accuracy(good_samples_folder, bad_samples_folder, runs=10) #, mixed_ps=True)
    #grid_search(good_samples_folder, bad_samples_folder, 'filters', 'kernel_size')
    

    X_train, y_train, ps_train, weather_train, X_test, y_test, ps_test, weather_test, indices_test, obsids_test, std = train_test_split(good_samples_folder, bad_samples_folder, seed=2) # Seed = 2
    #model, history, recall = evaluate_CNN_new(X_train, y_train, X_test, y_test, std, save_model=True, best_params=True)
    #model, history, recall = evaluate_NN_ps(X_train, y_train, X_test, y_test, save_model=False)


    #mean_accuracy(good_samples_folder, bad_samples_folder, runs=10, weather=False, ps=False, mixed_weather=False, mixed_ps=True)
    
    std = np.loadtxt('saved_nets/weathernet_BEST_std.txt')
    patience = int(np.loadtxt('saved_nets/weathernet_BEST_patience.txt'))
    model = load_model('saved_nets/weathernet_BEST.h5')
    history = np.load('saved_nets/weathernet_BEST_history.npy', allow_pickle=True).item()

    
    plot_recall(model, X_train, y_train, X_test, y_test, save=False)

    print('Training accuracy:', history['accuracy'][-patience])
    print('Validation accuracy:', history['val_accuracy'][-patience])
    print('Training loss:', history['loss'][-patience])
    print('Validation loss:', history['val_loss'][-patience])
    predictions_train = model.predict(X_train)
    y_true_train = y_train.argmax(axis=-1)
    y_pred_train = predictions_train.argmax(axis=-1)
    print('Training recall:', recall_score(y_true_train, y_pred_train))
    predictions_test = model.predict(X_test)
    y_true_test = y_test.argmax(axis=-1)
    y_pred_test = predictions_test.argmax(axis=-1)
    print('Validation recall:', recall_score(y_true_test, y_pred_test))
    print()

    
    cutoff = 0.23
    predictions_train = model.predict(X_train)
    y_true_train = y_train.argmax(axis=-1)
    y_pred_train = np.zeros(len(y_true_train))
    #y_pred_train = predictions_train.argmax(axis=-1)
    for i, predicted_train in enumerate(predictions_train): 
        if predicted_train[1] > cutoff:         
            y_pred_train[i] = 1 
        else:      
            y_pred_train[i] = 0  
        
    print(accuracy_score(y_true_train, y_pred_train), recall_score(y_true_train, y_pred_train))
    
    predictions_test = model.predict(X_test)
    y_true_test = y_test.argmax(axis=-1)
    y_pred_test = np.zeros(len(y_true_test))
    #y_pred_test = predictions_test.argmax(axis=-1)
    for i, predicted_test in enumerate(predictions_test): 
        if predicted_test[1] > cutoff:         
            y_pred_test[i] = 1 
        else:      
            y_pred_test[i] = 0  
    print(accuracy_score(y_true_test, y_pred_test), recall_score(y_true_test, y_pred_test))
    print()
    
    X_final_test, y_final_test, indices_final_test, obsids_final_test = final_testing_values(final_good_samples_folder, final_bad_samples_folder, std)

    predictions_final = model.predict(X_final_test)
    y_true_final = y_final_test.argmax(axis=-1)
    y_pred_final = np.zeros(len(y_true_final))
    #y_pred_final = predictions_final.argmax(axis=-1)
    for i, predicted_final in enumerate(predictions_final): 
        if predicted_final[1] > cutoff:         
            y_pred_final[i] = 1 
        else:      
            y_pred_final[i] = 0  
    print(accuracy_score(y_true_final, y_pred_final), recall_score(y_true_final, y_pred_final))
    print(confusion_matrix(y_true_final, y_pred_final, labels=[1,0]))
    

    
    #plot_history(history, patience=patience, save=False)
    analyse_classification_results(model, X_final_test, y_final_test, indices_final_test, obsids_final_test, plot=True)
    #plot_recall(model, X_train, y_train, X_test, y_test, save=False)
    
