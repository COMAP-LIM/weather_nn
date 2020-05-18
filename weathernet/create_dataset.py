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
    obsid = int(filename[6:13])
    index = (index1, index2)
    subseq = int(index2/30000)

    path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/' + month + '/'
    with h5py.File(path + filename, 'r') as hdf:
        tod       = np.array(hdf['spectrometer/band_average'])
        el        = np.array(hdf['spectrometer/pixel_pointing/pixel_el'])
        az        = np.array(hdf['spectrometer/pixel_pointing/pixel_az'])
        features  = np.array(hdf['spectrometer/features'])
        temp      = np.mean(np.array(hdf['hk/array/weather/airTemperature']))
        dewpoint  = np.mean(np.array(hdf['hk/array/weather/dewPointTemp']))
        pressure  = np.mean(np.array(hdf['hk/array/weather/pressure']))
        rain      = np.mean(np.array(hdf['hk/array/weather/rainToday']))
        humidity  = np.mean(np.array(hdf['hk/array/weather/relativeHumidity']))
        status    = np.mean(np.array(hdf['hk/array/weather/status']))
        winddeg   = np.mean(np.array(hdf['hk/array/weather/windDirection']))
        windspeed = np.mean(np.array(hdf['hk/array/weather/windSpeed'])) 


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
            
            mask = np.isnan(tod[feed, sideband])            
            if np.isnan(tod[feed, sideband,-1]):
                tod[feed, sideband, -1] = tod[feed, sideband, ~mask][-1]

            tod[feed, sideband, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), tod[feed, sideband,~mask])


    # Preprocessing
    tod = remove_elevation_azimuth_structures(tod, el, az)    
    #tod = remove_spikes(tod)                   
    #tod = scale(tod)
    tod = scale_two_mean(tod)
    #tod = scale_all_feeds(tod)
    # Calculating power spectrum  

    ps = power_spectrum(tod)

    # Weatherdata
    weatherdata = [temp, dewpoint, pressure, rain, humidity, status, winddeg, windspeed]

    
    for i in range(n):
        # Generate more data      
        if len(np.shape(tod)) < 2:
            simulated_data = generate_data(tod)
        else:
            simulated_data = generate_multidimensional_data(tod)
    
    
        # Calculating power spectrum for generated data  
        simulated_ps = power_spectrum(simulated_data)

    
        # Write generated data to file                                                         
        with h5py.File(output_folder + filename[:-4] + '_%d_simulated_%d.hd5' %(subseq,i), 'w') as hdf:
            hdf.create_dataset('tod', data=simulated_data)
            hdf.create_dataset('ps', data=simulated_ps)
            hdf.create_dataset('index', data=index)
            hdf.create_dataset('obsid', data=obsid*10)
            hdf.create_dataset('weather', data=weatherdata)
                
    
    # Write to file   
    with h5py.File(output_folder + filename[:-4] + '_%d.hd5' %subseq, 'w') as hdf:
        hdf.create_dataset('tod', data=tod)
        hdf.create_dataset('ps', data=ps)
        hdf.create_dataset('index', data=index)
        hdf.create_dataset('obsid', data=obsid)
        hdf.create_dataset('weather', data=weatherdata)
    


def generate_data(data):
    """    
    Generates data from power spectrum. 
    """
    ps = np.abs(np.fft.rfft(data))**2
    std = np.sqrt(ps)
    fourier_coeffs = np.sqrt(2)*(np.random.normal(loc=0, scale=std) + 1j*np.random.normal(loc=0, scale=std))
    new_data = np.fft.irfft(fourier_coeffs)

    #plt.figure(figsize=(4,3))
    #plt.plot(data)

    #plt.figure(figsize=(4,3))
    #plt.plot(new_data)
    #plt.show()

    return new_data


def generate_multidimensional_data(data):
    """    
    Generates data from power spectrum for multidimensional data. 
    """
    fourier_coeffs = np.fft.fft(data, axis=0)
    ps = np.abs(fourier_coeffs)**2
    std = np.sqrt(ps)

    freqs = np.fft.fftfreq(np.shape(data)[0])
    logbins = np.logspace(-5, -0.2, 10) 
    indices = np.digitize(freqs, logbins)

    corr_matrices = []
    for j in range(len(logbins)):
        fourier_coeffs_binned = fourier_coeffs[indices==j]
        corr = np.corrcoef(fourier_coeffs_binned, rowvar=False)
        
        if np.isnan(corr).any() and len(corr_matrices) > 0:
            corr_matrices.append(corr_matrices[-1])
        elif np.shape(fourier_coeffs_binned)[0] < 2 and len(corr_matrices) > 0:
            corr_matrices.append(corr_matrices[-1])
        else:
            corr_matrices.append(corr)
  
    fourier_coeffs_new = []
    for i in range(np.shape(fourier_coeffs)[0]):
        bin_nr = indices[i]
        corr = corr_matrices[bin_nr]
        cov = np.zeros(np.shape(corr))

        for l in range(np.shape(fourier_coeffs)[1]):
            for m in range(np.shape(fourier_coeffs)[1]):
                cov[l,m] = corr[l,m]*np.sqrt(std[i,l]**2*std[i,m]**2)
        mean = np.zeros(np.shape(fourier_coeffs)[1])
        r = np.sqrt(2)*np.random.multivariate_normal(mean, cov) 
        im = np.sqrt(2)*np.random.multivariate_normal(mean, cov)

        fourier_coeffs_new.append(r + 1j*im)


    new_data = np.fft.ifft(fourier_coeffs_new, axis=0)

    return new_data



def power_spectrum(data):
    """  
    Calculating power spectrum. 
    Args: 
        data (ndarray): 1D array containing preprocessed tod.
    Returns: 
        ps (ndarray): Scaled power spectrum. 
    """
    ps = np.abs(np.fft.fft(data,axis=0))**2
    freqs = np.fft.fftfreq(len(data))
    logbins = np.logspace(-5, -0.2, 10)
    power_spectra = []
    if len(np.shape(ps)) > 1:
        for j in range(np.shape(ps)[1]):
            ps_binned, bins = np.histogram(freqs, bins=logbins, weights=ps[:,j])
            ps_binned_2, bins = np.histogram(freqs, bins=logbins)
            power_spectra.append(ps_binned/ps_binned_2/1e6)
    else:
        ps_binned, bins = np.histogram(freqs, bins=logbins, weights=ps)
        ps_binned_2, bins = np.histogram(freqs, bins=logbins)
        power_spectra = ps_binned/ps_binned_2/1e6

    return power_spectra


def create_dataset_parallel():
    textfile_bad = open('data/bad_subsequences_ALL.txt', 'r')
    textfile_good = open('data/good_subsequences_ALL.txt', 'r')

    lines_good = textfile_good.readlines()[:400]
    lines_bad = textfile_bad.readlines()[:400]

    read_file_bad = partial(read_file, 'data/training_data_results/two_means/el_az_bad_new/', 0)
    read_file_good = partial(read_file, 'data/training_data_results/two_means/el_az_good_new/', 0)
        
    #for line in lines_bad[73:]:
    #    read_file_bad(line)
    
    with Pool() as pool:
        pool.map(read_file_bad, lines_bad)
    
    with Pool() as pool:
        pool.map(read_file_good, lines_good)
    

if __name__ == '__main__':
    import time
    start_time = time.time()
    create_dataset_parallel()
    print("--- %s seconds ---" % (time.time() - start_time))

