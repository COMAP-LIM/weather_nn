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

start_time = time.time()

fs = 50
T = 1/fs
subseq_length = int(10*60/T)

def subsequencegen(filename):
    obsid = filename[9:13]
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
                print('Not sufficient information in the level 1 file. ')
                return None, None
    except:
        return None, None

    if target[:2] != 'co':
        print('Target is not a co-field, but %s.' %target)
        return None, None

    if np.shape(tod)[2] < subseq_length*3:
        print('Too short tod.')
        return None, None

    # Removing Tsys measurements 
    boolTsys = (features & 1 << 13 != 8192)
    indexTsys = np.where(boolTsys == False)[0]

    if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
        boolTsys[:np.min(indexTsys)] = False
        boolTsys[np.max(indexTsys):] = False
    else:
        print('No Tsys measurements / only one measurement.')
        return None, None

    try:
        tod = tod[:,:,boolTsys]
        #MJD = MJD[boolTsys]
        el  = el[boolTsys]
        az  = az[boolTsys]
    except:
        print('Not corresponding length of boolTsys and number of samples.')
        return None, None

    # Making time-array for plotting
    #time = []
    #for i in range(len(MJD)):
    #    time.append(julian.from_jd(MJD[i], fmt='mjd'))

    tod_mean = np.nanmean(tod, axis=1)

    # Make subsequences
    sequences = []
    #subtimes = []

    subseq_numb = 0
    while np.shape(tod)[2] > subseq_length*(subseq_numb+1):
        subseq_numb += 1
        print(subseq_numb)

        subseq = tod[:,:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        #subtime = time[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subel = el[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subaz = az[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]

        subseq = preprocess_data(subseq, subel, subaz, obsid)

        sequences.append(subseq)
        #subtimes.append(subtime)

    return np.array(sequences), MJD_start



# load model
model = load_model('CNN_weathernet_new.h5')

folders = ['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07',\
           '2019-08', '2019-09', '2019-10', '2019-11', '2019-12', '2020-01', '2020-02']

files = []
for el in folders:
    files.extend(glob.glob( '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/%s/*.hd5' %el))

files.sort()

s = 0
for f in files:
    print()
    print(s/len(files))
    s+=1
    obsid = f[59:66]
    print('ObsID', obsid)
    sequences, MJD_start = subsequencegen(f)
    if sequences is None:
        continue 
    std = 0.08377675899274586
    sequences = sequences/std
    predictions = model.predict(sequences.reshape(np.shape(sequences)[0], np.shape(sequences)[1], 1)) 

    """
    for i in range(len(predictions)):
        print('Subseq %d    Good: %.4f    Bad: %.4f' %(i+1, predictions[i][0], predictions[i][1]))
    """
   
    file_subseq = open('weather_list.txt', 'a')
    for i in range(len(predictions)):
        file_subseq.write('%d    %d    %.4f   %f\n' %(int(obsid), i+1, predictions[i][1], MJD_start))

    file_obsid = open('weather_list_obsid.txt', 'a')
    file_obsid.write('%d    %.4f    %.4f   %f \n' %(int(obsid), max(predictions[:,1]), np.median(predictions[:,1]), MJD_start))

print("--- %s seconds ---" % (time.time() - start_time))
