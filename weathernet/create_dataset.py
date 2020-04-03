import numpy as np 
from preprocessing import scale, remove_elevation_azimuth_structures, remove_spikes
import os, glob, sys
import h5py
from functools import partial 
from multiprocessing import Pool
import matplotlib.pyplot as plt


def read_file(output_folder, n, line):
    """
    Reads in the subsequence and file specified in line. 
    """
    filename = line.split()[0]
    index1 = int(line.split()[1])
    index2 = int(line.split()[2])
    month = filename[14:21]
    obsid = int(filename[9:13])
    index = (index1, index2)
    subseq = int(index2/30000)

    path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/' + month + '/'
    with h5py.File(path + filename, 'r') as hdf:
        tod       = np.array(hdf['spectrometer/band_average'])
        el        = np.array(hdf['spectrometer/pixel_pointing/pixel_el'])
        az        = np.array(hdf['spectrometer/pixel_pointing/pixel_az'])
        features  = np.array(hdf['spectrometer/features'])
        
    # Removing Tsys measurements                                                                   
    boolTsys = (features & 1 << 13 != 8192)
    indexTsys = np.where(boolTsys == False)[0]

    if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
        boolTsys[:np.min(indexTsys)] = False
        boolTsys[np.max(indexTsys):] = False

    tod       = tod[:,:,boolTsys]
    el        = el[:,boolTsys]
    az        = az[:,boolTsys]

    # Extracting subsequence  
    tod       = tod[:,:,index1:index2]
    el        = el[:,index1:index2]
    az        = az[:,index1:index2]
 
    # Removing NaNs  
    for feed in range(np.shape(tod)[0]):
        for sideband in range(np.shape(tod)[1]):
            if np.isnan(tod[feed, sideband]).all():
                tod[feed,sideband] = 0

    mask = np.isnan(tod)
    tod[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), tod[~mask])

    # Preprocessing
    tod = remove_elevation_azimuth_structures(tod, el, az)    
    tod = remove_spikes(tod)                                                                     
    tod = scale(tod)

    # Calculating power spectrum  
    ps = power_spectrum(tod)


    for i in range(n):
        # Generate more data                                         
        simulated_data = generate_data(tod)

        # Calculating power spectrum for generated data  
        simulated_ps = power_spectrum(simulated_data)

        # Write generated data to file                                                         
        with h5py.File(output_folder + filename[:-4] + '_%d_simulated_%d.hd5' %(subseq,i), 'w') as hdf:
            hdf.create_dataset('tod', data=simulated_data)
            hdf.create_dataset('ps', data=simulated_ps)
            hdf.create_dataset('index', data=index)
            hdf.create_dataset('obsid', data=obsid*10)
                
    # Write to file   
    with h5py.File(output_folder + filename[:-4] + '_%d.hd5' %subseq, 'w') as hdf:
        hdf.create_dataset('tod', data=tod)
        hdf.create_dataset('ps', data=ps)
        hdf.create_dataset('index', data=index)
        hdf.create_dataset('obsid', data=obsid)



def generate_data(data):
    """    
    Generates data from power spectrum. 
    """
    ps = np.abs(np.fft.rfft(data))**2
    std = np.sqrt(ps)
    fourier_coeffs = np.sqrt(2)*(np.random.normal(loc=0, scale=std) + 1j*np.random.normal(loc=0, scale\
=std))
    new_data = np.fft.irfft(fourier_coeffs)

    return new_data


def power_spectrum(data):
    """  
    Calculating power spectrum. 
    Args: 
        data (ndarray): 1D array containing preprocessed tod.
    Returns: 
        ps (ndarray): Scaled power spectrum. 
    """
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(len(data))

    logbins = np.logspace(-5, -0.2, 10)
    ps_binned, bins = np.histogram(freqs, bins=logbins, weights=ps)
    ps_binned_2, bins = np.histogram(freqs, bins=logbins)

    return ps_binned/ps_binned_2/1e6

def create_dataset_parallel():
    textfile_bad = open('data/bad_subsequences_ALL_updated.txt', 'r')
    lines_bad = textfile_bad.readlines()

    textfile_good = open('data/good_subsequences_ALL_updated.txt', 'r')
    lines_good = textfile_good.readlines()

    read_file_bad = partial(read_file, 'data/bad_test_update/', 5)
    read_file_good = partial(read_file, 'data/good_test_update/', 0)


    with Pool() as pool:
        pool.map(read_file_bad, lines_bad)
    
    with Pool() as pool:
        pool.map(read_file_good, lines_good)
    

if __name__ == '__main__':
    import time
    start_time = time.time()
    create_dataset_parallel()
    print("--- %s seconds ---" % (time.time() - start_time))
    #print('Parallell')
