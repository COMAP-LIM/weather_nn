import keras
import numpy as np
import random
import cv2, os, glob
import matplotlib.pyplot as plt
from keras.models import load_model
import sys
import h5py
import julian
from scipy.optimize import curve_fit
import matplotlib


fs = 50
T = 1/fs
subseq_length = int(10*60/T)

def subsequencegen(filename):
    obsid = filename[9:13]
    with h5py.File(filename, 'r') as hdf:
        try:
            tod      = np.array(hdf['spectrometer/band_average'])
            MJD      = np.array(hdf['spectrometer/MJD'])
            el       = np.array(hdf['/spectrometer/pixel_pointing/pixel_el'])[0]
            az       = np.array(hdf['/spectrometer/pixel_pointing/pixel_az'])[0]
            features = np.array(hdf['spectrometer/features'])
        except:
            print('Not sufficient information in the level 1 file. ')
            sys.exit()

    if np.shape(tod)[2] < subseq_length*3:
        print('Too short tod.')
        sys.exit()

    tod = np.nanmean(tod, axis=1)
    #tod = np.nanmean(tod_feeds, axis=0)

    # Removing Tsys measurements 
    boolTsys = (features != 8192)
    indexTsys = np.where(boolTsys==False)[0]

    if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
        boolTsys[:np.min(indexTsys)] = False
        boolTsys[np.max(indexTsys):] = False
    else:
        print('No Tsys measurements / only one measurement.')
        sys.exit()

    try:
        tod = tod[:,boolTsys]
        MJD = MJD[boolTsys]
        el  = el[boolTsys]
        az  = az[boolTsys]
    except:
        print('Not corresponding length of boolTsys and number of samples.')
        sys.exit()

    tod_new = np.nanmean(tod, axis=0)

    # Making time-array for plotting
    time = []
    for i in range(len(MJD)):
        time.append(julian.from_jd(MJD[i], fmt='mjd'))

    # Make subsequences
    sequences = []
    subtimes = []

    subseq_numb = 0
    while len(tod_new) > subseq_length*(subseq_numb+1):
        subseq_numb += 1
        print(subseq_numb)

        subseq = tod_new[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subtime = time[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subel = el[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subaz = az[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]

        subseq = preprocess_data(subseq, subel, subaz, obsid)

        sequences.append(subseq)
        subtimes.append(subtime)
    
    return np.array(sequences), subtimes, tod, time

def remove_elevation_gain(X, g, a, c, d, e):
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2

def preprocess_data(data, el, az, obsid):
    part = int(len(el)/4)

    t = np.arange(len(el))
    popt1, pcov1 = curve_fit(remove_elevation_gain, (t[:part],el[:part], az[:part]), data[:part])
    popt2, pcov2 = curve_fit(remove_elevation_gain, (t[part:part*2],el[part:part*2], az[part:part*2]), data[part:part*2])
    popt3, pcov3 = curve_fit(remove_elevation_gain, (t[part*2:part*3],el[part*2:part*3], az[part*2:part*3]), data[part*2:part*3])
    popt4, pcov4 = curve_fit(remove_elevation_gain, (t[-part:],el[-part:],az[-part:]), data[-part:])

    g = np.zeros(len(el))
    g[:part] = popt1[0]
    g[part:part*2] = popt2[0]
    g[part*2:part*3] = popt3[0]
    g[-part:] = popt4[0]

    a = np.zeros(len(el))
    a[:part] = popt1[1]
    a[part:part*2] = popt2[1]
    a[part*2:part*3] = popt3[1]
    a[-part:] = popt4[1]

    diff = np.zeros(len(el))
    diff[part:part*2] = (data[part-1] - g[part-1]/np.sin(el[part-1]*np.pi/180) - a[part-1]*az[part-1]) - (data[part] - g[part]/np.sin(el[part]*np.pi/180) - a[part]*az[part])
    diff[part*2:part*3] = (data[part*2-1] - g[part*2-1]/np.sin(el[part*2-1]*np.pi/180) - a[part*2-1]*az[part*2-1]) - (data[part*2] - g[part*2]/np.sin(el[part*2]*np.pi/180) - a[part*2]*az[part*2]) + diff[part]
    diff[part*3:part*4] = (data[part*3-1] - g[part*3-1]/np.sin(el[part*3-1]*np.pi/180) - a[part*3-1]*az[part*3-1]) - (data[part*3] - g[part*3]/np.sin(el[part*3]*np.pi/180) - a[part*3]*az[part*3]) + diff[part*2]

    # Removing elevation gain                                                                                  
    data = data - g/np.sin(el*np.pi/180) - a*az + diff

    # Normalizing                                                                                              
    data = (data - np.mean(data))/np.std(data)

    return data


# load model
model = load_model('weathernet_az.h5')


#files = glob.glob( '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2019-11/*.hd5')

files = ['/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2019-11/comap-0009470-2019-11-23-145820.hd5', '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2019-11/comap-0009469-2019-11-23-135046.hd5']

for f in files:
    obsid = f[62:66]
    print('ObsID', obsid)
    print(f)
    with h5py.File(f, 'r') as hdf:
        attributes = hdf['comap'].attrs
        target = attributes['source'].decode()
        if target[:2] != 'co':
            print('Target is not a co-field, but %s.' %target)
            continue
    sequences, subtimes, tod, time = subsequencegen(f)
    predictions = model.predict(sequences.reshape(np.shape(sequences)[0], len(sequences[0]), 1))

    for i in range(len(predictions)):
        print('Subseq %d    Good: %.4f    Bad: %.4f' %(i+1, predictions[i][0], predictions[i][1]))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    hours = matplotlib.dates.MinuteLocator(interval = 10)
    h_fmt = matplotlib.dates.DateFormatter('%H:%M:%S')
    for i in range(np.shape(tod)[0]-1):
        plt.plot(time, tod[i])

    for i in range(len(sequences)+1):
        plt.axvline(time[i*subseq_length], color='black', alpha=0.5)
        if i != len(sequences):
            plt.text(time[i*subseq_length+int(subseq_length/2)], 150000, '%d' %(i+1), alpha=0.5)


    #plt.plot(time, tod)
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(h_fmt)
    fig.autofmt_xdate()
    plt.show()



"""
for i in range(len(obsids)):
    plt.imshow(x_test[i][:,:,0],cmap='gray')
    plt.title('ObsID: ' + obsids[i])
    plt.text(105, 15, 'Bad weather: %.2f \nGood weather: %.2f'  %(predictions[i][0], predictions[i][1]) ,
        bbox={'facecolor': 'lightblue', 'alpha': 0.7, 'pad': 10})
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.show()
"""
