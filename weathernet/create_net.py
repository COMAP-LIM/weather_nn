import numpy as np 
import glob
import random
import h5py
import matplotlib.pyplot as plt 

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

from keras.layers import Dropout
from keras.layers import MaxPooling1D

import seaborn as sns

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


def evaluate_CNN_new(X_train, y_train, X_test, y_test, std, params = False, save_model=False):
    if params == False:
        params = {'epochs': 150, 'batch_size': 256, 'patience': 20, 'lr': 1e-4, \
                  'filters': 32, 'kernel_size': 6, 'activation': 'relu'} 
        
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    adam = optimizers.Adam(lr=params['lr']) 
    es = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)

    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=24, activation=params['activation'], input_shape=(n_timesteps,n_features)))
    model.add(MaxPooling1D(pool_size = 3))
    model.add(Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], activation=params['activation']))
    model.add(Flatten())
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    plot_model(model, to_file='default_architecture.pdf', show_shapes=False, show_layer_names=True) 
    sys.exit()

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

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) 

    if save_model:
        model.save("saved_nets/weathernet_test.h5")
        with open('saved_nets/weathernet_test_std.txt', 'w') as f:
            f.write(str(std))
            print("Saved model to disk")

    return model, history, recall


def grid_search(good_samples_folder, bad_samples_folder, parameter1, parameter2):
    activation = ['relu', 'tanh', 'sigmoid', 'linear']
    lr = [1e-3, 1e-4, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    bs = [128, 256, 16, 32, 64, 128, 256, 512]
    filters = [64,128]#[16, 32]#, 64, 128]
    kernel_size = [3, 6, 9, 12, 24]

    suggested_params = {'batch_size': bs, 'lr': lr, 'filters': filters, \
                        'kernel_size': kernel_size, 'activation': activation}

    default_params = {'epochs': 150, 'batch_size': 256, 'patience': 20, 'lr': 1e-4, \
                      'filters': 32, 'kernel_size': 6, 'activation': 'relu'} 
    # 150
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

            
            #all_results = np.array(all_results)

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


    np.save('figures/heatmaps/heatmap_test_accuracy_%s_%s_layer2_64_128.npy' %(parameter1, parameter2), all_test_accuracies) 
    np.save('figures/heatmaps/heatmap_train_accuracy_%s_%s_layer2_64_128.npy' %(parameter1, parameter2), all_train_accuracies) 
    np.save('figures/heatmaps/heatmap_test_loss_%s_%s_layer2_64_128.npy' %(parameter1, parameter2), all_test_loss) 
    np.save('figures/heatmaps/heatmap_train_loss_%s_%s_layer2_64_128.npy' %(parameter1, parameter2), all_train_loss) 
    np.save('figures/heatmaps/heatmap_recall_%s_%s_layer2_64_128.npy' %(parameter1, parameter2), all_recall) 



    for i in range(len(results)):
        print(results[i])

    fig, ax = plt.subplots(figsize = (5, 5))
    sns.heatmap(all_test_accuracies, annot=True, fmt=".2f", ax=ax,  cmap="YlGnBu")
    ax.set_title("Validation accuracy")
    ax.set_xlabel(parameter2)
    ax.set_ylabel(parameter1)
    ax.set_xticklabels(suggested_params[parameter2])
    ax.set_yticklabels(suggested_params[parameter1])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('heatmap_accuracy_%s_%s.pdf' %(parameter1, parameter2))
    
    fig, ax = plt.subplots(figsize = (5, 5))
    sns.heatmap(all_recall, annot=True, fmt=".2f", ax=ax,  cmap="YlGnBu")
    ax.set_title("Recall")
    ax.set_xlabel(parameter2)
    ax.set_ylabel(parameter1)
    ax.set_xticklabels(suggested_params[parameter2])
    ax.set_yticklabels(suggested_params[parameter1])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('heatmap_recall_%s_%s.pdf' %(parameter1, parameter2))
    

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
    verbose, epochs, patience, batch_size = 1, 1500, 20, 256
    n_timesteps, n_outputs, n_features = X_train.shape[2], y_train.shape[1], X_train.shape[1]
    adam = optimizers.Adam(lr=1e-3)  
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(24, activation='relu'))
    model.add(Dense(12, activation='relu'))
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


def plot_history(history, save=False):
    plt.figure()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train data', 'Testing data'], loc='upper left')
    plt.grid()
    plt.title('Model accuracy')
    if save:
        plt.savefig('history_accuracies_feb.png')

    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.ylim((-0.1, 2))
    plt.legend(['Training data', 'Testing data'], loc='upper left')
    if save:
        plt.savefig('history_loss_feb.png')
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
            CNN_model, history, recall = evaluate_CNN_new(X_train, y_train, X_test, y_test, std) 
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



if __name__ == '__main__':
    good_samples_folder =  'data/training_data_results/two_means/el_az_good/'
    bad_samples_folder = 'data/training_data_results/two_means/el_az_bad/' 

    #good_samples_folder =  'data/training_data_results/mixed/good/'
    #bad_samples_folder = 'data/training_data_results/mixed/bad/' 

    #mean_accuracy(good_samples_folder, bad_samples_folder, runs=10)#, ps=True)
    #print('Two means, el az')


    X_train, y_train, ps_train, weather_train, X_test, y_test, ps_test, weather_test, indices_test, obsids_test, std = train_test_split(good_samples_folder, bad_samples_folder, seed=2)
    #model, accuracy, history = evaluate_NN_ps(ps_train, y_train, ps_test, y_test, save_model=True)
    #model, history = evaluate_NN_weather(weather_train, y_train, weather_test, y_test, save_model=True)
    model, history, recall = evaluate_CNN_new(X_train, y_train, X_test, y_test, std, save_model=False)
    #acc = evaluate_mixed(X_train, weather_train, y_train, X_test, weather_test, y_test)
    #print(history['accuracy'][-1], history['val_accuracy'][-1], history['loss'][-1], history['val_loss'][-1])

    #evaluate_CNN_new(X_train, y_train, X_test, y_test, std, save_model=False)
    #print(accuracy)

    #grid_search(good_samples_folder, bad_samples_folder, 'filters', 'kernel_size')

    #plot_history(history)
    #acc = evaluate_mixed(X_train, ps_train, y_train, X_test, ps_test, y_test)
