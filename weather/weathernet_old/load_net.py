import keras
import numpy as np
import random
import cv2, os, glob
import matplotlib.pyplot as plt
from keras.models import load_model
#from create_dataset import preprocess_data
import sys
import h5py
import julian
from scipy.optimize import curve_fit
import matplotlib
import time 
from CNN_weathernet import create_dataset

def remove_elevation_gain(X, g, a, c, d, e):
    """
    Template for elevation gain. 
    """
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2


def preprocess_data(data, el, az, obsid):
    """
    Preprocesses the data by normalizing, averaging over feeds 
    and sideband, removing elevation gain and azimuth structures
    and subtracting mean.
    Args:
        data (ndarray): 3D array containing sidebandaverage tods.
        el (ndarray): Array containing elevation for each data point. 
        az (ndarray): Array containing azimuth for each data point.
        obsid (str): The obsID of the tod
    Returns:
        data (ndarray): 1D preprocessed data. 
    """

    # Normalizing by dividing each feed on its own mean
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            data[i][j] = data[i][j]/np.nanmean(data[i][j])-1

                 
    # Mean over feeds and sidebands           
    data = np.nanmean(data, axis=0)
    data = np.nanmean(data, axis=0)

    # Removing NaNs   
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])


    # Calculating template for elevation gain and azimuth structue removal 
    part = int(len(el)/4)

    t = np.arange(len(el))
    diff = np.zeros(len(el))
    temp = np.zeros(len(el))
    for i in range(4):
        popt, pcov = curve_fit(remove_elevation_gain, (t[part*i:part*(i+1)],el[part*i:part*(i+1)], az[part*i:part*(i+1)]), data[part*i:part*(i+1)])
        g = popt[0]
        a = popt[1]

        temp[part*i:part*(i+1)] = g/np.sin(el[part*i:part*(i+1)]*np.pi/180) + a*az[part*i:part*(i+1)]
        diff[part*i:part*(i+1)] = (data[part*i-1] - temp[part*i-1]) - (data[part*i] - temp[part*i]) + diff[part*(i-1)]

    # Removing elevation gain and azimuth structures
    data = data - temp + diff
 
    #plt.figure(figsize=(4,3))
    #plt.plot(data)
   
    data = data - np.mean(data)

    return data

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


    if np.shape(tod)[2] < subseq_length:
        # 'Too short tod.'
        return None, None

    # Make subsequences
    sequences = []
    subseq_numb = 0
    while np.shape(tod)[2] > subseq_length*(subseq_numb+1):
        subseq_numb += 1

        subseq = tod[:,:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subel = el[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subaz = az[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]

        # subseq = remove_elevation_structures(subsew, subel, subaz)
        # subseq = remove_spikes(subel)
        subseq = preprocess_data(subseq, subel, subaz, obsid)
        

        sequences.append(subseq)

    return np.array(sequences), MJD_start


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
    
    print(obsid)

    # Legge inn paralellprosessering her 
    sequences, MJD_start = subsequencegen(f)

    if sequences is None:
        continue 

    std = np.loadtxt('weathernet_current_std.txt') 
    sequences = sequences/std
    #if np.shape(sequences)[0]>1:
    predictions = model.predict(sequences.reshape(np.shape(sequences)[0], np.shape(sequences)[1], 1)) 
    #else:
    #    print(sequences)
    #    print(np.shape(sequences))
    #    predictions = model.predict(sequences.reshape(1, len(sequences), 1))
    file_subseq = open('weather_list.txt', 'a')
    for i in range(len(predictions)):
        file_subseq.write('%d    %d    %.4f   %f\n' %(int(obsid), i+1, predictions[i][1], MJD_start))

    file_obsid = open('weather_list_obsid.txt', 'a')
    file_obsid.write('%d    %.4f    %.4f   %f \n' %(int(obsid), max(predictions[:,1]), np.median(predictions[:,1]), MJD_start))


