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
import sys

def subsequencegen(filename, spikelist_filename, spikes=False):
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
                feeds      = np.array(hdf['spectrometer/feeds'])
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
        num_Tsys_values = np.max(np.where(boolTsys[:int(len(boolTsys)/2)] == False)[0]) + 1
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
    while np.shape(tod)[2] > subseq_length*(subseq_numb+1):
        subseq_numb += 1

        subseq = tod[:,:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subel = el[:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subaz = az[:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]

        # Preprocessing
        subseq = remove_elevation_azimuth_structures(subseq, subel, subaz)

        # Find spikes and write spike data to file
        if spikes == True:
            find_spikes(subseq, spikelist_filename, obsid, mjd, MJD_start, feeds, subseq_length, num_Tsys_values, subseq_numb)

        subseq = scale(subseq)
        sequences.append(subseq)

    return np.array(sequences), MJD_start


def find_spikes(subseq, spikelist_filename, obsid, mjd, MJD_start, feeds, subseq_length, num_Tsys_values, subseq_numb):
    file_subseq = open(spikelist_filename, 'a')
    header = 'obsid     feed   sideband     width          ampl        index            mjd            mjd_start \n'
    
    if os.stat(spikelist_filename).st_size == 0:
        file_subseq.write(header)

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
        spike_tops, spike_widths, spike_ampls, subseq_feed = spike_list[j]
        subseq_new[feed,sideband] = subseq_feed

        for i in range(len(spike_tops)):
            if spike_widths[i] == 0 or spike_widths[i]*2 >= 200:
                pass
            else:
                file_subseq.write('%d        %d        %d        %.4f        %.4f       %d        %f       %f\n' %(int(obsid), feeds[feed], sideband+1, spike_widths[i]*2, spike_ampls[i],  spike_tops[i] + subseq_length*(subseq_numb-1) + num_Tsys_values, mjd[spike_tops[i]+subseq_length*(subseq_numb-1)], MJD_start))


def update_weather_and_spike_list(weatherlist_filename, spikelist_filename, weathernet):
    # Load model 
    std = np.loadtxt(weathernet[:-3] + '_std.txt')
    model = load_model(weathernet)

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
    if os.path.exists(weatherlist_filename):
        last_checked_obsid = np.loadtxt(weatherlist_filename, dtype=int, usecols=(0))[-1]
        last_checked = '%07d' %last_checked_obsid
        last_checked_filename = [f for f in files if last_checked in f][0]
        last_checked_index = files.index(last_checked_filename)
    else:
        last_checked_index = -1 

    s = 0
    for f in files[last_checked_index+1:]:
        s+=1
        obsid = f[59:66]

        print(obsid)
        sequences, MJD_start = subsequencegen(f, spikelist_filename, spikes=True)

        if sequences is None:
            print('Passing')
            continue 

        sequences = sequences/std
        predictions = model.predict(sequences.reshape(np.shape(sequences)[0], np.shape(sequences)[1], 1)) 
        file_subseq = open(weatherlist_filename, 'a')
        for i in range(len(predictions)):
            file_subseq.write('%d    %d    %.4f   %f\n' %(int(obsid), i+1, predictions[i][1], MJD_start))

        file_obsid = open(weatherlist_filename, 'a')
        file_obsid.write('%d    %.4f    %.4f   %f \n' %(int(obsid), max(predictions[:,1]), np.median(predictions[:,1]), MJD_start))



def update_spike_list_only(spikelist_filename):
    # Load model 
    std = np.loadtxt(weathernet[:-3] + '_std.txt')
    model = load_model(weathernet)

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
    if os.path.exists(weatherlist_filename):
        last_checked_obsid = np.loadtxt(spikelist_filename, dtype=int, usecols=(0), skiprows=1)[-1]
        last_checked = '%07d' %last_checked_obsid
        last_checked_filename = [f for f in files if last_checked in f][0]
        last_checked_index = files.index(last_checked_filename)
    else:
        last_checked_index = -1 

    #files = ['/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2019-07/comap-0006801-2019-07-09-005158.hd5', '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2019-06/comap-0006557-2019-06-17-172637.hd5', '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2019-07/comap-0006801-2019-07-09-005158.hd5']
    #last_checked_index = 0

    s = 0
    for f in files[last_checked_index+1:]:
        s+=1
        obsid = f[59:66]

        print(obsid)
        sequences, MJD_start = subsequencegen(f, spikelist_filename, spikes=True)


def update_weather_list_only(weatherlist_filename, weathernet):
    # Load model 
    std = np.loadtxt(weathernet[:-3] + '_std.txt')
    model = load_model(weathernet)

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
    if os.path.exists(weatherlist_filename):
        last_checked_obsid = np.loadtxt(weatherlist_filename, dtype=int, usecols=(0))[-1]
        last_checked = '%07d' %last_checked_obsid
        last_checked_filename = [f for f in files if last_checked in f][0]
        last_checked_index = files.index(last_checked_filename)
    else:
        last_checked_index = -1 

    s = 0
    for f in files[last_checked_index+1:]:
        s+=1
        obsid = f[59:66]

        print(obsid)
        sequences, MJD_start = subsequencegen(f, None, spikes=False)

        if sequences is None:
            print('Passing')
            continue 

        sequences = sequences/std
        predictions = model.predict(sequences.reshape(np.shape(sequences)[0], np.shape(sequences)[1], 1)) 
        file_subseq = open(weatherlist_filename, 'a')
        for i in range(len(predictions)):
            file_subseq.write('%d    %d    %.4f   %f\n' %(int(obsid), i+1, predictions[i][1], MJD_start))

        file_obsid = open(weatherlist_filename, 'a')
        file_obsid.write('%d    %.4f    %.4f   %f \n' %(int(obsid), max(predictions[:,1]), np.median(predictions[:,1]), MJD_start))




weatherlist_filename = '/mn/stornext/d16/cmbco/comap/marenras/master/weathernet/data/weather_data/weather_list_obsid_TEST.txt'
spikelist_filename = '/mn/stornext/d16/cmbco/comap/marenras/master/weathernet/data/spike_data/spike_list_TEST.txt'
weathernet = '/mn/stornext/d16/cmbco/comap/marenras/master/weathernet/saved_nets/weathernet_current.h5'

import time
start_time = time.time()

#update_weather_list_only(weatherlist_filename, weathernet)
update_spike_list_only(spikelist_filename)
#update_weather_and_spike_list(weatherlist_filename, spikelist_filename, weathernet)

print("--- %s seconds ---" % (time.time() - start_time))
