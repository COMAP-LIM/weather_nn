import keras
import numpy as np
import random
import cv2, os, glob
import matplotlib.pyplot as plt
from keras.models import load_model
from create_dataset import preprocess_data
import sys
import h5py
import julian
from scipy.optimize import curve_fit
import matplotlib
import time 


def subsequencegen(filename):
    # Calculating subsequence length 
    fs = 50
    T = 1/fs
    subseq_length = int(10*60/T)

    # Reading in relevant information from file
    try:
        with h5py.File(filename, 'r') as hdf:
            try:
                tod        = np.array(hdf['spectrometer/band_average'])
                el         = np.array(hdf['/spectrometer/pixel_pointing/pixel_el'])[0]
                az         = np.array(hdf['/spectrometer/pixel_pointing/pixel_az'])[0]
                features   = np.array(hdf['spectrometer/features'])
                MJD_start  = float(hdf['spectrometer/MJD'][0])
                attributes = hdf['comap'].attrs
                target     = attributes['source'].decode()
            except:
                # 'Not sufficient information in the level 1 file'
                return None, None
    except:
        return None, None

    if target[:2] != 'co':
        # 'Target is not a co-field, but %s.' %target
        return None, None

    if np.shape(tod)[2] < subseq_length*3:
        # 'Too short tod.'
        return None, None

    # Removing Tsys measurements 
    boolTsys = (features & 1 << 13 != 8192)
    indexTsys = np.where(boolTsys == False)[0]

    if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
        boolTsys[:np.min(indexTsys)] = False
        boolTsys[np.max(indexTsys):] = False
    else:
        # 'No Tsys measurements / only one measurement.'
        return None, None

    try:
        tod = tod[:,:,boolTsys]
        el  = el[boolTsys]
        az  = az[boolTsys]
    except:
        # 'Not corresponding length of boolTsys and number of samples.'
        return None, None


    # Make subsequences
    sequences = []
    subseq_numb = 0
    while np.shape(tod)[2] > subseq_length*(subseq_numb+1):
        subseq_numb += 1

        subseq = tod[:,:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subel = el[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subaz = az[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]

        subseq = preprocess_data(subseq, subel, subaz, obsid)

        sequences.append(subseq)

    return np.array(sequences), MJD_start



# Load model
model = load_model('weathernet_current.h5')

# Make list with relevant folders
folders = glob.glob('/mn/stornext/d16/cmbco/comap/pathfinder/ovro/20*/')
for el in folders:
    if len(el) > 53:
        folders.remove(el)

# Make list with files
files = []
for el in folders:
    files.extend(glob.glob('%s/*.hd5' %el))
files.sort()



if os.path.exists('weather_list_obsid.txt'):
    last_checked_obsid = np.loadtxt('weather_list_obsid.txt', dtype=int, usecols=(0))[-1]
    last_checked = '%07d' %last_checked_obsid
    last_checked_filename = [f for f in files if last_checked in f][0]
    last_checked_index = files.index(last_checked_filename)
else:
    last_checked_index = 0 


s = 0
for f in files[last_checked_index+1:]:
    s+=1
    obsid = f[59:66]
    sequences, MJD_start = subsequencegen(f)
    if sequences is None:
        continue 

    std = np.loadtxt('weathernet_current_std.txt') 
    sequences = sequences/std
    predictions = model.predict(sequences.reshape(np.shape(sequences)[0], np.shape(sequences)[1], 1)) 

    file_subseq = open('weather_list.txt', 'a')
    for i in range(len(predictions)):
        file_subseq.write('%d    %d    %.4f   %f\n' %(int(obsid), i+1, predictions[i][1], MJD_start))

    file_obsid = open('weather_list_obsid.txt', 'a')
    file_obsid.write('%d    %.4f    %.4f   %f \n' %(int(obsid), max(predictions[:,1]), np.median(predictions[:,1]), MJD_start))

