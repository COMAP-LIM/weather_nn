# Weathernet - COMAP 

In this respotory you will find codes for the weathernet algorithm, together with other codes from my master thesis. 

## Description of programs

**preprocessing.py**

Program containing functions used for preprocessing of the data. This file also includes the spike detection algorithm. 


**create_dataset.py**

Creates the dataset used for training the convolutional neural network from files specified in a provided text-file (data/good_subsequences_ALL.txt and data/bad_subsequences_ALL.txt). This includes extracting the relevant data from the level 1 files, preprocessing the data, and writing the preprocessed data into new HDF5-files. 


**create_net.py**

Trains an one dimensional convolutional neural network on a given dataset. This file also includes a function for calculating the mean accuracy, mean recall and mean loss for a model over a given number of runs, and plotting the tradeoff between the recall and accuracy for a given model. 


**load_net.py**

Loads a saved net and writes the probability for bad weather of all level 1 files to a text-file. If the text file already exists, the program will continue with the obsID after the last obsID in the text file. This program can also be used to extract information about spikes in the data for all level 1 files. 
