import keras
import numpy as np
import random
import os, glob
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
import matplotlib
import time 
from multiprocessing import Pool
from preprocessing import scale_two_mean, remove_elevation_azimuth_structures, remove_spikes_parallell
import sys
from functools import partial

def subsequencegen(filename):
    obsid = int(filename[-29:-22])

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
                return None, None, None, None, None, None
    except:
        return None, None, None, None, None, None

    if target[:2] != 'co':
        # 'Target is not a co-field, but %s.' %target
        return None, None, None, None, None, None
        
    # Removing Tsys measurements 
    boolTsys = (features & 1 << 13 != 8192)
    indexTsys = np.where(boolTsys == False)[0]

    if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
        boolTsys[:np.min(indexTsys)] = False
        boolTsys[np.max(indexTsys):] = False
        num_Tsys_values = np.max(np.where(boolTsys[:int(len(boolTsys)/2)] == False)[0]) + 1
    else:
        # 'No Tsys measurements / only one measurement.'
        return None, None, None, None, None, None

    try:
        tod = tod[:,:,boolTsys]
        el  = el[:,boolTsys]
        az  = az[:,boolTsys]
        mjd = mjd[boolTsys]
    except:
        # 'Not corresponding length of boolTsys and number of samples.'
        return None, None, None, None, None, None


    if np.shape(tod)[2] < subseq_length:
        # 'Too short tod.'
        return None, None, None, None, None, None

    # Removing NaNs     
    for feed in range(np.shape(tod)[0]):
        for sideband in range(np.shape(tod)[1]):
            if np.isnan(tod[feed, sideband]).all():
                tod[feed,sideband] = 0

            mask = np.isnan(tod[feed, sideband])
            if np.isnan(tod[feed, sideband,-1]):
                tod[feed, sideband, -1] = tod[feed, sideband, ~mask][-1]

            tod[feed, sideband, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), tod[feed, sideband,~mask])


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
        sequences.append(subseq)

    return np.array(sequences), feeds, mjd, MJD_start, subseq_length, num_Tsys_values


def find_spikes(filename, spikelist_filename):
    obsid = int(filename[-29:-22])
    sequences, feeds, mjd, MJD_start, subseq_length, num_Tsys_values = subsequencegen(filename)
    print(obsid)

    if MJD_start != None:
        file_subseq = open(spikelist_filename, 'a')
        header = 'obsid     feed   sideband     width          ampl        index            mjd            mjd_start \n'
        if os.stat(spikelist_filename).st_size == 0:
            file_subseq.write(header)
            
        subseq_numb = 0 
        for subseq in sequences:
            subseq_numb += 1
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

    else:
        print('Passing')


def find_weather(model, std, filename):
    obsid = int(filename[-29:-22])
    sequences, feeds, mjd, MJD_start, subseq_length, num_Tsys_values = subsequencegen(filename)
    
    if sequences is None:
        return [obsid, None, None]
 
    else:
        sequences_scaled = []
        for subseq in sequences:
            sequences_scaled.append(scale_two_mean(subseq))
        sequences_scaled = sequences_scaled/std
        predictions = model.predict(sequences_scaled)

        return [obsid, predictions, MJD_start]


def update_weatherlist(weatherlist_filename, weathernet):
    # Load model 
    std = np.loadtxt(weathernet[:-3] + '_std.txt')
    model = load_model(weathernet)

    # Make list with relevant folders
    folders = glob.glob('/mn/stornext/d22/cmbco/comap/protodir/level1/20*/')
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


    file_subseq = open(weatherlist_filename, 'a')
    file_obsid = open(weatherlist_filename[:-4]+'_obsid.txt', 'a')

    for f in files[last_checked_index+1:]:
        obsid, predictions, MJD_start = find_weather(model, std, f)
        print(obsid)
        if MJD_start == None:
            print('passing')
            pass
        else:
            for j in range(len(predictions)):
                file_subseq.write('%d    %d    %.4f   %f\n' %(int(obsid), j+1, predictions[j][1], MJD_start))
            file_obsid.write('%d    %.4f    %.4f   %f \n' %(int(obsid), max(predictions[:,1]), np.median(predictions[:,1]), MJD_start))
            

        


def update_spikelist(spikelist_filename):
    # Make list with relevant folders
    folders = glob.glob('/mn/stornext/d22/cmbco/comap/protodir/level1/20*/')
    for el in folders:
        if len(el) > 53:
            folders.remove(el)

    # Make list with files
    files = []
    for el in folders:
        files.extend(glob.glob('%s/*.hd5' %el))
    files.sort()

    # Check the last obsid written to file
    if os.path.exists(spikelist_filename):
        last_checked_obsid = np.loadtxt(spikelist_filename, dtype=int, usecols=(0), skiprows=1)[-1]
        last_checked = '%07d' %last_checked_obsid
        last_checked_filename = [f for f in files if last_checked in f][0]
        last_checked_index = files.index(last_checked_filename)
    else:
        last_checked_index = -1 

    print(len(files))
    remaining_files = files[7200:]#[last_checked_index+1:]
    #print(remaining_files)

    s = 0
    for f in remaining_files:
        print(s/len(remaining_files)*100)
        find_spikes(f, spikelist_filename)
        s+=1
    


weatherlist_filename = '/mn/stornext/d22/cmbco/comap/protodir/auxiliary/weather_list.txt'
spikelist_filename = '/mn/stornext/d22/cmbco/comap/protodir/master/weathernet/data/spike_data/spike_list_TEST.txt'
weathernet = '/mn/stornext/d22/cmbco/comap/protodir/COMAP_weather_nn/weathernet/saved_nets/weathernet_BEST.h5'

import time
start_time = time.time()

update_weatherlist(weatherlist_filename, weathernet)
#update_spikelist(spikelist_filename)

print("--- %s seconds ---" % (time.time() - start_time))
