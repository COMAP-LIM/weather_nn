import keras
import numpy as np
import random
import cv2, os, glob
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
import julian
import matplotlib
import time 
from multiprocessing import Pool
from preprocessing import scale, remove_elevation_azimuth_structures, remove_spikes_parallell


def subsequencegen(filename):
    obsid = int(filename[59:66])

    # Calculating subsequence length 
    fs = 50
    T = 1/fs
    subseq_length = int(10*60/T)

    # Reading in relevant information from file
    try:
        with h5py.File(filename, 'r') as hdf:
            try:
                tod        = np.array(hdf['spectrometer/band_average'])
                el         = np.array(hdf['/spectrometer/pixel_pointing/pixel_el'])
                az         = np.array(hdf['/spectrometer/pixel_pointing/pixel_az'])
                features   = np.array(hdf['spectrometer/features'])
                mjd        = np.array(hdf['spectrometer/MJD'])
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
        num_Tsys_values = np.max(np.where(boolTsys[:int(len(boolTsys)/2)] == False)[0])
    else:
        # 'No Tsys measurements / only one measurement.'
        return None, None

    try:
        tod = tod[:,:,boolTsys]
        el  = el[:,boolTsys]
        az  = az[:,boolTsys]
        mjd = mjd[boolTsys]
    except:
        # 'Not corresponding length of boolTsys and number of samples.'
        return None, None


    if np.shape(tod)[2] < subseq_length:
        # 'Too short tod.'
        return None, None

    # Removing NaNs                                                                                                  
    for feed in range(np.shape(tod)[0]):
        for sideband in range(np.shape(tod)[1]):
            if np.isnan(tod[feed, sideband]).all():
                tod[feed,sideband] = 0

    mask = np.isnan(tod)
    tod[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), tod[~mask])


    # Make subsequences
    sequences = []
    subseq_numb = 0
    file_subseq = open('spike_list.txt', 'a')
    while np.shape(tod)[2] > subseq_length*(subseq_numb+1):
        subseq_numb += 1

        subseq = tod[:,:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subel = el[:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subaz = az[:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]

        subseq = remove_elevation_azimuth_structures(subseq, subel, subaz)
        #subseq = remove_spikes_write_to_file(subel, obsid, subseq_length*subseq_numb, mjd, MJD_start, num_Tsys_values)

        print(subseq_numb)
        #spike_list = remove_spikes_parallell(subseq[1,2])
        #print(spike_list[0])
        
        
        all_feeds = []
        feed_nummeration = []
        for feed in range(np.shape(subseq)[0]):
            for sideband in range(np.shape(subseq)[1]):
                all_feeds.append(subseq[feed][sideband])
                feed_nummeration.append((feed,sideband))
                
        with Pool() as pool:
            spike_list = pool.map(remove_spikes_parallell, all_feeds)
        

        subseq_new = np.zeros(np.shape(subseq))
        for j in range(len(all_feeds)):
            feed, sideband = feed_nummeration[j]
            spike_tops, spike_widths, subseq_feed = spike_list[j]
            subseq_new[feed,sideband] = subseq_feed

            for i in range(len(spike_tops)):
                file_subseq.write('%d    %d    %d    %.4f   %d    %f   %f\n' %(int(obsid), feed, sideband, spike_widths[i], spike_tops[i] + subseq_length*(subseq_numb-1), mjd[spike_tops[i]+subseq_length*(subseq_numb-1)], MJD_start))
        
        subseq = scale(subseq_new)
        
        sequences.append(subseq)

    return np.array(sequences), MJD_start


def update_weatherlist():
    # Load model
    model = load_model('weathernet_current.h5')
    std = np.loadtxt('weathernet_current_std.txt') 

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

    # Check the last obsid written to file
    if os.path.exists('weather_list_obsid_new.txt'):
        last_checked_obsid = np.loadtxt('weather_list_obsid_new.txt', dtype=int, usecols=(0))[-1]
        last_checked = '%07d' %last_checked_obsid
        last_checked_filename = [f for f in files if last_checked in f][0]
        last_checked_index = files.index(last_checked_filename)
    else:
        last_checked_index = 0 

    last_checked_index = 5000
        
    s = 0
    for f in files[last_checked_index+1:]:
        s+=1
        obsid = f[59:66]

        print(obsid)
        sequences, MJD_start = subsequencegen(f)

        if sequences is None:
            print('Passing')
            continue 

        sequences = sequences/std
        predictions = model.predict(sequences.reshape(np.shape(sequences)[0], np.shape(sequences)[1], 1)) 
        file_subseq = open('weather_list.txt', 'a')
        for i in range(len(predictions)):
            file_subseq.write('%d    %d    %.4f   %f\n' %(int(obsid), i+1, predictions[i][1], MJD_start))

        file_obsid = open('weather_list_obsid.txt', 'a')
        file_obsid.write('%d    %.4f    %.4f   %f \n' %(int(obsid), max(predictions[:,1]), np.median(predictions[:,1]), MJD_start))


update_weatherlist()
