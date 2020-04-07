import numpy as np 
import matplotlib.pyplot as plt
import h5py
from preprocessing import scale, remove_elevation_azimuth_structures, remove_spikes
from multiprocessing import Pool

def read_file(line):
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
        mjd = np.array(hdf['spectrometer/MJD'])

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

    for i in range(np.shape(tod)[0]):
        plt.figure()
        plt.plot(tod[i,0])
        #plt.plot([20394,25283,25403], [tod[i,0][20394],tod[i,0][25283],tod[i,0][25403]], 'ro')
        plt.title('Feed: %d, Sideband: 0' %i) 
        """
        std_noise = np.std(tod[i,0][1:]-tod[i,0][:-1])/np.sqrt(2)
        print(i, 0)
        print('std:', std_noise)
        print('25400:', (tod[i,0][25400]-np.mean(tod[i,0]))/std_noise, tod[i,0][25400])
        print('20394:', (tod[i,0][20394]-np.mean(tod[i,0]))/std_noise, tod[i,0][20394])
        print('25283:', (tod[i,0][25283]-np.mean(tod[i,0]))/std_noise, tod[i,0][25283])
        print('25403:', (tod[i,0][25403]-np.mean(tod[i,0]))/std_noise, tod[i,0][25403])
        """
        plt.show()


    plt.figure()
    for feed in range(np.shape(tod)[0]-1):
        for sideband in range(np.shape(tod)[1]):
            plt.plot(np.arange(index[0], index[1]), tod[feed,sideband])
    plt.title('Obsid: %d, subseq: %d' %(obsid, int(index[1]/30000)) )
    #plt.savefig('figures_good_test/%d_%d_%d.png' %(obsid, index[0], index[1]))
    plt.show()


textfile_bad = open('data/training_data/bad_subsequences_ALL.txt', 'r')
lines_bad = textfile_bad.readlines()

textfile_good = open('data/training_data/good_subsequences_ALL.txt', 'r')
lines_good = textfile_good.readlines()


lines = ['comap-0006801-2019-07-09-005158.hd5   0   30000', 'comap-0006557-2019-06-17-172637.hd5  0  30000']

for line in lines:
    read_file(line)

#with Pool() as pool:
#    pool.map(read_file, lines_good)
