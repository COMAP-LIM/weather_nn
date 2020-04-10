import numpy as np 
from preprocessing import scale, remove_elevation_azimuth_structures, remove_spikes, scale_two_mean, scale_all_feeds
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
    #tod = remove_spikes(tod)                                                                     
    #tod = scale(tod)
    tod = scale_two_mean(tod)
    #tod = scale_all_feeds(tod)

    # Calculating power spectrum  
    ps = 0#power_spectrum(tod)

    
    for i in range(n):
        # Generate more data                                         
        #simulated_data = generate_data(tod)
        simulated_data = generate_multidimensional_data(tod)
    
    """
        # Calculating power spectrum for generated data  
        simulated_ps = 0#power_spectrum(simulated_data)

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
    """
    

def generate_data(data):
    """    
    Generates data from power spectrum. 
    """
    ps = np.abs(np.fft.rfft(data))**2
    std = np.sqrt(ps)
    fourier_coeffs = np.sqrt(2)*(np.random.normal(loc=0, scale=std) + 1j*np.random.normal(loc=0, scale=std))
    new_data = np.fft.irfft(fourier_coeffs)

    plt.plot(power_spectrum(data))
    plt.plot(power_spectrum(new_data))
    plt.show()

    return new_data

def generate_multidimensional_data(data):
    """    
    Generates data from power spectrum for multidimensional data. 
    """
    fourier_coeffs1 = np.fft.fft(data[:,0])
    fourier_coeffs2 = np.fft.fft(data[:,1])
    
    ps1 = np.abs(np.fft.fft(data[:,0]))**2
    std1 = np.sqrt(ps1)
    ps2 = np.abs(np.fft.fft(data[:,1]))**2
    std2 = np.sqrt(ps2)

    freqs = np.fft.fftfreq(np.shape(data)[0])
    logbins = np.logspace(-5, -0.2, 10)  # -5, -0.2, 10 
    indices = np.digitize(freqs, logbins)
    
    corr_matrices = []
    for j in range(len(logbins)):
        fourier_coeffs1_binned = fourier_coeffs1[indices==j]
        fourier_coeffs2_binned = fourier_coeffs2[indices==j]

        corr = np.corrcoef(fourier_coeffs1_binned, fourier_coeffs2_binned)
        #corr_matrices.append(corr)
        
        if np.isnan(corr).any() and len(corr_matrices) > 0:
            corr_matrices.append(corr_matrices[-1])
            print('nan')
        #elif np.isnan(corr).any() and len(corr_matrices) == 0:
        #    corr_matrices.append(np.ones((2,2)))           
        else:
            corr_matrices.append(corr)
        

    fourier_coeffs1_new = []
    fourier_coeffs2_new = []
    for i in range(len(fourier_coeffs1)):
        bin_nr = indices[i]
        cov = corr_matrices[bin_nr]*np.sqrt(std1[i]**2*std2[i]**2)
        
        cov[0,0] = std1[i]**2
        cov[1,1] = std2[i]**2

        r = np.sqrt(2)*np.random.multivariate_normal([0,0], cov) 
        im = np.sqrt(2)*np.random.multivariate_normal([0,0], cov)

        #fourier_coeffs1_new.append(np.sqrt(2)*(np.random.normal(loc=0, scale=std1[i]) + 1j*np.random.normal(loc=0, scale=std1[i])))
        #fourier_coeffs2_new.append(np.sqrt(2)*(np.random.normal(loc=0, scale=std2[i]) + 1j*np.random.normal(loc=0, scale=std2[i])))

        fourier_coeffs1_new.append(r[0] + 1j*im[0])
        fourier_coeffs2_new.append(r[1] + 1j*im[1])


    new_data1 = np.fft.ifft(fourier_coeffs1_new)
    new_data2 = np.fft.ifft(fourier_coeffs2_new)

    plt.figure(figsize=(4,3))
    plt.plot(data[:,0])
    plt.plot(data[:,1])

    plt.figure(figsize=(4,3))
    plt.plot(new_data1)
    plt.plot(new_data2)
    plt.show()

    plt.figure(figsize=(4,3))
    plt.plot(power_spectrum(data[:,0]))
    plt.plot(power_spectrum(data[:,1]))

    plt.figure(figsize=(4,3))
    plt.plot(power_spectrum(new_data1))
    plt.plot(power_spectrum(new_data2))
    plt.show()
    
    """
    new_data = np.zeros(np.shape(data))
    for i in range(np.shape(data)[1]):
        np.random.seed(24)
        ps = np.abs(np.fft.rfft(data[:,i]))**2
        std = np.sqrt(ps)
        fourier_coeffs = np.sqrt(2)*(np.random.normal(loc=0, scale=std) + 1j*np.random.normal(loc=0, scale=std))
        new_data[:,i] = np.fft.irfft(fourier_coeffs)

    """

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

    return ps_binned/ps_binned_2#/1e6


def create_dataset_parallel():
    textfile_bad = open('data/training_data_preprocess/bad_subsequences_ALL.txt', 'r')
    lines_bad = textfile_bad.readlines()

    textfile_good = open('data/training_data_preprocess/good_subsequences_ALL.txt', 'r')
    lines_good = textfile_good.readlines()

    read_file_bad = partial(read_file, 'data/training_data_preprocess/two_means/bad_generated/', 5)
    read_file_good = partial(read_file, 'data/training_data_preprocess/two_means/good_more_good/', 0)


    with Pool() as pool:
        pool.map(read_file_bad, lines_bad)
    
    #with Pool() as pool:
    #    pool.map(read_file_good, lines_good)
    

if __name__ == '__main__':
    
    textfile_bad = open('data/training_data_preprocess/bad_subsequences_ALL.txt', 'r')
    lines_bad = textfile_bad.readlines()

    for j in range(30):
        read_file('test/', 1, lines_bad[j+2])
    """
    
    import time
    start_time = time.time()
    create_dataset_parallel()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    """
