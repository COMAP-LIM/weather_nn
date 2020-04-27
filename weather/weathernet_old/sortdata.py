import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob, os, os.path
import random
import julian
import datetime
from scipy.optimize import curve_fit


def elevation_azimuth_template(X, g, a, c, d, e):
    """  
    Template for elevation gain and azimuth correlations.                                
    """
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2




def remove_elevation_azimuth_structures(data, el, az, plot=False):
    """   
    Removes elevation gain and azimuth structures for each feed and sideband.                           
    """
    for feed in range(np.shape(data)[0]):
        for sideband in range(np.shape(data)[1]):
            num_parts = 4
            part = int(np.shape(el)[1]/num_parts)

            # Calculating template for elevation gain and azimuth structue removal
            t = np.arange(np.shape(el)[1])
            diff = np.zeros(np.shape(el)[1])
            temp = np.zeros(np.shape(el)[1])
            fitted = np.zeros(np.shape(el)[1])

            for i in range(num_parts):
                if np.all(data[feed, sideband, part*i:part*(i+1)]==0):
                    continue
                else:
                    if i == num_parts-1:
                        popt, pcov = curve_fit(elevation_azimuth_template, (t[part*i:],el[feed, \
                                        part*i:], az[feed, part*i:]), data[feed, sideband, part*i:])
                        g = popt[0]
                        a = popt[1]

                        temp[part*i:] = g/np.sin(el[feed, part*i:]*np.pi/180) + \
                                    a*az[feed, part*i:]
                        diff[part*i:] = (data[feed, sideband, part*i-1] - temp[part*i-1]) - \
                                    (data[feed, sideband, part*i] - temp[part*i]) + diff[part*(i-1)]

                        fitted[part*i:] = elevation_azimuth_template((t[part*i:],el[feed, part*i:], az[feed, part*i:]), *popt)
                    else:
                        popt, pcov = curve_fit(elevation_azimuth_template, (t[part*i:part*(i+1)], \
                                el[feed, part*i:part*(i+1)], az[feed,  \
                                part*i:part*(i+1)]), data[feed, sideband, part*i:part*(i+1)])
                        g = popt[0]
                        a = popt[1]

                        temp[part*i:part*(i+1)] = g/np.sin(el[feed, part*i:part*(i+1)] \
                                        *np.pi/180) + a*az[feed, part*i:part*(i+1)]
                        diff[part*i:part*(i+1)] = (data[feed, sideband, part*i-1] - temp[part*i-1]) \
                            - (data[feed, sideband, part*i]- temp[part*i]) + diff[part*(i-1)]
                        fitted[part*i:part*(i+1)] = elevation_azimuth_template((t[part*i:part*(i+1)],el[feed, part*i:part*(i+1)], az[feed, part*i:part*(i+1)]), *popt)

            if plot:
                if feed == 0 and sideband == 0:
                    plt.plot(fitted, 'r', alpha=0.7, linewidth=1, label='Fitted template')

            # Removing elevation gain and azimuth structures
            data[feed, sideband] = data[feed, sideband] - temp + diff

    return data






fs = 50 
T = 1/fs 
subseq_length = int(10*60/T)

save_choices = {'g': 'data/good_subsequences_TESTING2.txt', 'b': 'data/bad_subsequences_TESTING2.txt'}
path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2020-02/'
data_path = os.path.join(path, '*.hd5') 
lines = glob.glob(data_path)                                                                           

last_count = 20  #117
count = last_count

for f in lines[last_count:]:
    f = f.replace(" ", "")
    f = f[len(path):]
    
    print('--------------------------')
    print('%d' %count)
    count = count + 1
    month = f[14:21]
    obsid = int(f[7:13])

    with h5py.File(path + f, 'r') as hdf:
        print('ObsID:', obsid)
        try:
            attributes = hdf['comap'].attrs 
            target = attributes['source'].decode()
            if target[:2] != 'co':
                print('Target is not a co-field, but %s.' %target)
                continue

            tod = np.array(hdf['spectrometer/band_average'])
            el  = np.array(hdf['spectrometer/pixel_pointing/pixel_el'])
            az  = np.array(hdf['spectrometer/pixel_pointing/pixel_az'])
            MJD = np.array(hdf['spectrometer/MJD'])
            features = np.array(hdf['spectrometer/features'])

        except:
            print('No band average/MJD/features')
            continue
        
    # Removing T-sys measurements                                                                   
    boolTsys = (features != 8192)
    indexTsys = np.where(boolTsys==False)[0]

    if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
        boolTsys[:np.min(indexTsys)] = False
        boolTsys[np.max(indexTsys):] = False
    else:
        print('No Tsys measurements, or only one measurement.')
        continue

    try:
        tod_new = tod[:,:,boolTsys]
        el = el[:,boolTsys]
        az = az[:,boolTsys]
        MJD = MJD[boolTsys]
    except:
        print('Not corresponding length of boolTsys and number of samples.')
        continue

    # Check length of tod
    if np.shape(tod_new)[-1] < subseq_length*3:
        print('Too short tod')
        continue

    for feed in range(np.shape(tod_new)[0]):
        for sideband in range(np.shape(tod_new)[1]):
            if np.isnan(tod_new[feed, sideband]).all():
                tod_new[feed,sideband] = 0

    mask = np.isnan(tod_new)
    tod_new[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), tod_new[~mask])


    time = []
    for i in range(len(MJD)):
        time.append(julian.from_jd(MJD[i], fmt='mjd'))

    # Remove elevation and azimuth structures
    tod_new = remove_elevation_azimuth_structures(tod_new, el, az)
    

    # Plot whole tod (without feed 20)
    fig = plt.figure(figsize=(7,3))
    ax = fig.add_subplot(111)
    hours = matplotlib.dates.MinuteLocator(interval = 10)
    h_fmt = matplotlib.dates.DateFormatter('%H:%M:%S')
                                                                  
    for sideband in range(np.shape(tod_new)[1]):
        for feed in range(np.shape(tod_new)[0]-1):
            plt.plot(time, tod_new[feed,sideband,:], linewidth=0.8, label='Feed: %d' %(i+1))

    for i in range(int(np.shape(tod_new)[-1])//30000 + 1):
        try:
            plt.axvline(time[i*subseq_length], color='black', alpha=0.5)
            if i != int(np.shape(tod_new)[-1])//30000:
                plt.text(time[i*subseq_length+int(subseq_length/2)], max_y*0.85, '%d' %(i+1), alpha=0.5)
        except:
            pass

    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(h_fmt)
    plt.xlabel('UTC (hours)')
    plt.ylabel('Power')
    fig.autofmt_xdate()
    plt.show(block=False)


    c = input("Do you want to label all subsequences in obsid %s as the same category?  (y/n) \n" %obsid)

    if c == 'y':
        label = input("Chose category for all subsequences in obsid %s: \n1) Good (g) \n2) Bad (b) \n3) Pass (p) \n" %obsid)
        if label == 'p':
            pass
        else:
            file1 = open(save_choices[label], 'r')
            lines = file1.readlines()
            subseq_numb = 0
            while np.shape(tod_new)[-1] > subseq_length*(subseq_numb+1):
                subseq_numb += 1
                new_line = '%s   %d   %d   \n' %(f,subseq_length*(subseq_numb-1),subseq_length*subseq_numb)
                if new_line not in lines:
                    file1 = open(save_choices[label], 'a')
                    file1.write(new_line)
                    print('Written to file.')
                else:
                    print('Subsequence %d is already in file' %subseq_numb)

    elif c == 'n':
        labels = input("Input labels for each subsequences separated by space for obsid %s: \n1) Good (g) \n2) Bad (b) \n3) Pass (p) \n" %obsid)
        labels = labels.split()
        subseq_numb = 0
        while np.shape(tod_new)[-1] > subseq_length*(subseq_numb+1):
            subseq_numb += 1
            if labels[subseq_numb-1] == 'p':
                pass
            else:
                file1 = open(save_choices[labels[subseq_numb-1]], 'r')
                lines = file1.readlines()
                new_line = '%s   %d   %d   \n' %(f,subseq_length*(subseq_numb-1),subseq_length*subseq_numb)
                if new_line not in lines:
                    file1 = open(save_choices[labels[subseq_numb-1]], 'a')
                    file1.write(new_line)
                    print('Written to file.')
                else:
                    print('Subsequence %d is already in file' %subseq_numb)
  
    plt.show()

