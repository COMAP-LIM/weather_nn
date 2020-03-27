import numpy as np 
import os, glob, sys
import h5py 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import getopt


def extract_data_to_file(textfile, output_folder, generate_more_data=False, n=5):
    """
    Opens files listed in textfile, reads in relevant values
    and writes only relevant, preprocessed information to 
    new files in output_folder. 

    Args:
        textfile (str): The file location and filename of the textfile
        containing a list of files to preprocess.
        output_folder (str): Folder where new files should be saved.
        generate_more_data (bool): Generate more tods from power spectrum
        or not (default is False).
        n (int): Number of generated tods per file (default is 10).
    
    Returns:
        None
    """

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

        """
        print(obsid)
        print(subseq)
        plt.figure(figsize=(3,2))
        for i in range(np.shape(tod)[0]):
            plt.plot(np.mean(tod, axis=1)[i])
        """
    
        # Preprocessing   
        where_are_NaNs = np.isnan(tod)
        tod[where_are_NaNs] = 0
        tod = remove_elevation_azimuth_structures(tod, el, az)
        #tod = remove_spikes(tod)
        tod = preprocess_data(tod)

        """
        plt.figure(figsize=(3,2))
        plt.grid('Power')
        plt.xlabel('Sample')
        plt.plot(tod)
        """
        
        if generate_more_data:
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
        
        # Calculating power spectrum       
        ps = power_spectrum(tod)
        
        """
        # Normalizing               
        #tod = normalize(tod)

        
        plt.figure(figsize=(4,3))
        plt.plot(tod)
        plt.grid()
        plt.ylabel('Power')
        plt.xlabel('Sample')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig('7368_preprocessed_new.png')
        plt.show()
        """
        
        # Write to file 
        with h5py.File(output_folder + filename[:-4] + '_%d.hd5' %subseq, 'w') as hdf:
            hdf.create_dataset('tod', data=tod)
            hdf.create_dataset('ps', data=ps)
            hdf.create_dataset('index', data=index)
            hdf.create_dataset('obsid', data=obsid)
        

def elevation_azimuth_template(X, g, a, c, d, e):
    """
    Template for elevation gain and azimuth correlations. 
    """
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2


def remove_elevation_azimuth_structures(tod, el, az):
    for feed in range(np.shape(tod)[0]):
        for sideband in range(np.shape(tod)[1]):
            num_parts = 4
            part = int(np.shape(el)[1]/num_parts)

            # Calculating template for elevation gain and azimuth structue removal 
            t = np.arange(np.shape(el)[1])
            diff = np.zeros(np.shape(el)[1])
            temp = np.zeros(np.shape(el)[1])
            for i in range(num_parts):
                if np.all(tod[feed, sideband, part*i:]==0):
                    continue
                else:
                    if i == num_parts-1:
                        popt, pcov = curve_fit(elevation_azimuth_template, (t[part*i:],el[feed, \
                                part*i:], az[feed, part*i:]), tod[feed, sideband, part*i:])
                        g = popt[0]
                        a = popt[1]

                        temp[part*i:] = g/np.sin(el[feed, part*i:]*np.pi/180) + \
                                    a*az[feed, part*i:]
                        diff[part*i:] = (tod[feed, sideband, part*i-1] - temp[part*i-1]) - \
                                    (tod[feed, sideband, part*i] - temp[part*i]) + diff[part*(i-1)]

                    else:
                        popt, pcov = curve_fit(elevation_azimuth_template, (t[part*i:part*(i+1)], \
                                el[feed, part*i:part*(i+1)], az[feed,  \
                                part*i:part*(i+1)]), tod[feed, sideband, part*i:part*(i+1)])
                        g = popt[0]
                        a = popt[1]
                    
                        temp[part*i:part*(i+1)] = g/np.sin(el[feed, part*i:part*(i+1)] \
                                        *np.pi/180) + a*az[feed, part*i:part*(i+1)]
                        diff[part*i:part*(i+1)] = (tod[feed, sideband, part*i-1] - temp[part*i-1]) \
                            - (tod[feed, sideband, part*i]- temp[part*i]) + diff[part*(i-1)]

            # Removing elevation gain and azimuth structures
            tod[feed, sideband] = tod[feed, sideband] - temp + diff

    return tod

def preprocess_data(data):
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
    for feed in range(np.shape(data)[0]):
        for sideband in range(np.shape(data)[1]):
            data[feed][sideband] = data[feed][sideband]/np.nanmean(data[feed][sideband])-1

                 
    # Mean over feeds and sidebands           
    data = np.nanmean(data, axis=0)
    data = np.nanmean(data, axis=0)

    # Zero-center data 
    data = data - np.mean(data)

    return data


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

    #plt.plot(ps_binned/ps_binned_2)          
    #plt.show() 

    return ps_binned/ps_binned_2/1e6#1e18


def generate_data(data):
    """
    Generates data from power spectrum.

    Args:
        data (ndarray): 1D array containing preprocessed tods.
    Returns:
        new_data (ndarray): 1D array containing new generated tod. 
    """
    ps = np.abs(np.fft.rfft(data))**2
    std = np.sqrt(ps)
    fourier_coeffs = np.sqrt(2)*(np.random.normal(loc=0, scale=std) + 1j*np.random.normal(loc=0, scale=std))
    new_data = np.fft.irfft(fourier_coeffs)

    return new_data

def usage():
    print('No')
    sys.exit(2)

if __name__ == '__main__':
    textfile = 'data/bad_subsequences_ALL.txt'
    output_folder = 'bad_test2/'
    extract_data_to_file(textfile, output_folder, generate_more_data=True, n=5)

    """
    if len(sys.argv) > 3:
        try:
            opts, args = getopt.getopt(sys.argv[1:], 't:o:g:n:', ['textfile=', 'output_folder=", "generate_data=', 'number='])
        except getopt.GetoptError:
            usage()
            sys.exit(2)

        for opt, arg in opts:
            if opt in ('-t','--textfile'):
                textfile = arg
            if opt in ('-o', '--output_file'):
                output_file = arg
            if opt in ('-g', '--generate_data'):
                generate_data = arg
            if opt in ('-n', '--number'):
                n = arg
            else:
                usage()
                sys.exit(2)
    else:
        usage()
        sys.exit(2)
        
    if n and generate_more_data:
        extract_data_to_file(textfile, output_folder, generate_more_data, n)
    elif generate_more_data:
        extract_data_to_file(textfile, output_folder, generate_more_data)
    else:
        extract_data_to_file(textfile, output_folder)
    """
    #textfile = 'data/bad_subsequences_ALL.txt'
    #output_folder = 'bad_test/'
    #extract_data_to_file(textfile, output_folder, generate_more_data=True, n=5)

