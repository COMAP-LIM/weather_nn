import numpy as np 
import os, glob, sys
import h5py 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit


def extract_data_to_file(textfile, output_folder, generate_more_data=False, n=10):
    f = open(textfile, 'r')
    lines = f.readlines()

    for line in lines:
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
            el        = np.array(hdf['spectrometer/pixel_pointing/pixel_el'][0])
            az        = np.array(hdf['spectrometer/pixel_pointing/pixel_az'][0])
            features  = np.array(hdf['spectrometer/features'])

        # Removing Tsys measurements  
        boolTsys = (features != 8192)
        indexTsys = np.where(boolTsys==False)[0]

        if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
            boolTsys[:np.min(indexTsys)] = False
            boolTsys[np.max(indexTsys):] = False

        tod       = tod[:,:,boolTsys]
        el        = el[boolTsys]
        az        = az[boolTsys]

        # Extracting subsequence 
        tod       = tod[:,:,index1:index2]
        el        = el[index1:index2]
        az        = az[index1:index2]

        # Preprocessing     
        tod = preprocess_data(tod, el, az, obsid, index)

        if generate_more_data:
            for i in range(n):
                simulated_data = generate_data(tod)

                # Calculating power spectrum for generated data 
                simulated_ps = power_spectrum(simulated_data)

                # Normalizing generated data  
                # Should the normalization happen in another way??
                simulated_data = normalize(simulated_data)

                with h5py.File(output_folder + filename[:-4] + '_%d_simulated_%d.hd5' %(subseq,i), 'w') as hdf:
                    hdf.create_dataset('tod', data=simulated_data)
                    hdf.create_dataset('ps', data=simulated_ps)
                    hdf.create_dataset('index', data=index)
                    hdf.create_dataset('obsid', data=obsid*10)

        # Calculating power spectrum       
        ps = power_spectrum(tod)
        
        # Normalizing               
        tod = normalize(tod)

        with h5py.File(output_folder + filename[:-4] + '_%d.hd5' %subseq, 'w') as hdf:
            hdf.create_dataset('tod', data=tod)
            hdf.create_dataset('ps', data=ps)
            hdf.create_dataset('index', data=index)
            hdf.create_dataset('obsid', data=obsid)


def remove_elevation_gain(X, g, a, c, d, e):
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2


def preprocess_data(data, el, az, obsid, index):
    # Normalizing by dividing each feed on its own mean                                                                           
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            data[i][j] = data[i][j]/np.nanmean(data[i][j])

    #print(np.shape(data))                                                                                                        
    # Mean over feeds and sidebands                                                                                               
    data = np.nanmean(data, axis=0)
    data = np.nanmean(data, axis=0)


    part = int(len(el)/4)

    t = np.arange(len(el))
    g = np.zeros(len(el))
    a = np.zeros(len(el))
    std = np.zeros(len(el))
    diff = np.zeros(len(el))
    for i in range(4):
        popt, pcov = curve_fit(remove_elevation_gain, (t[part*i:part*(i+1)],el[part*i:part*(i+1)], az[part*i:part*(i+1)]), data[part*i:part*(i+1)])
        g[part*i:part*(i+1)] = popt[0]
        a[part*i:part*(i+1)] = popt[1]
        std[part*i:part*(i+1)] = np.std(data[part*i:part*(i+1)])
        diff[part*i:part*(i+1)] = (data[part*i-1] - g[part*i-1]/np.sin(el[part*i-1]*np.pi/180) - a[part*i-1]*az[part*i-1]) - (data[part*i] - g[part*i]/np.sin(el[part*i]*np.pi/180) - a[part*i]*az[part*i]) + diff[part*(i-1)]


    # Removing elevation gain                                  
    data = data - g/np.sin(el*np.pi/180) - a*az + diff

    #data = data[::10]                                                                                                            
    return data

def normalize(data):
    data = (data - np.mean(data))/np.std(data)

    return data


def power_spectrum(data):
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(len(data))

    logbins = np.logspace(-5, -0.2, 10)
    ps_binned, bins = np.histogram(freqs, bins=logbins, weights=ps)
    ps_binned_2, bins = np.histogram(freqs, bins=logbins)

    #plt.plot(ps_binned/ps_binned_2)                                                                                              
    #plt.show()                                                                                                                   

    return ps_binned/ps_binned_2/1e6#1e18

def generate_data(data):
    ps = np.abs(np.fft.rfft(data))**2
    std = np.sqrt(ps)
    fourier_coeffs = np.sqrt(2)*(np.random.normal(loc=0, scale=std) + 1j*np.random.normal(loc=0, scale=std))
    new_data = np.fft.irfft(fourier_coeffs)

    return new_data


if __name__ == '__main__':
    textfile = 'data/bad_subsequences_ALL.txt'
    output_folder = 'bad_samples_no_gen/'
    extract_data_to_file(textfile, output_folder, generate_more_data=False, n=5)

