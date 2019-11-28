import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob, os, os.path
import random
import julian
import datetime

fs = 50 
T = 1/fs 
subseq_length = int(10*60/T)

save_choices = {'g': 'data/good_subsequences_test.txt', 'b': 'data/bad_subsequences_test.txt'}
path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2019-11/'
data_path = os.path.join(path, '*.hd5') 
lines = glob.glob(data_path)                                                                           

last_count = 334
count = last_count

for f in lines[last_count:]:
    f = f.replace(" ", "")
    f = f[len(path):]
    
    print('--------------------------')
    print('%d' %count)
    count = count + 1
    month = f[14:21]
    obsid = f[9:13]

    with h5py.File(path + f, 'r') as hdf:
        print('ObsID:', obsid)
        try:
            attributes = hdf['comap'].attrs 
            target = attributes['source'].decode()
            if target[:2] != 'co':
                print('Target is not a co-field, but %s.' %target)
                continue

            tod = np.array(hdf['spectrometer/band_average'])
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
        MJD = MJD[boolTsys]
    except:
        print('Not corresponding length of boolTsys and number of samples.')
        continue


    time = []
    for i in range(len(MJD)):
        time.append(julian.from_jd(MJD[i], fmt='mjd'))

    # Finding average over bands only                                                               
    tod_new = np.nanmean(tod_new, axis=1)

    # Check length of tod
    if np.shape(tod_new)[1] < subseq_length*3:
        print('Too short tod')
        continue

    # Plot whole tod (without feed 20)
    fig = plt.figure(figsize=(7,3))
    ax = fig.add_subplot(111)
    hours = matplotlib.dates.MinuteLocator(interval = 10)
    h_fmt = matplotlib.dates.DateFormatter('%H:%M:%S')
                                                                                    
    for i in range(np.shape(tod_new)[0]-1):
        plt.plot(time, tod_new[i,:], linewidth=0.8, label='Feed: %d' %(i+1))

    for i in range(7):
        plt.axvline(time[i*subseq_length], color='black', alpha=0.5)
        if i != 6:
            plt.text(time[i*subseq_length+int(subseq_length/2)], 400000, '%d' %(i+1), alpha=0.5)
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(h_fmt)
    plt.xlabel('UTC (hours)')
    plt.ylabel('Power')
    fig.autofmt_xdate()
    plt.show(block=False)


    c = input("Do you want to label all subsequences in obsid %s as the same category? (y/n) \n" %obsid)

    if c == 'y':
        label = input("Chose category for all subsequences in obsid %s: \n1) Good (g) \n2) Bad (b) \n3) Pass (p) \n" %obsid)
        if label == 'p':
            pass
        else:
            file1 = open(save_choices[label], 'r')
            lines = file1.readlines()
            subseq_numb = 0
            while np.shape(tod_new)[1] > subseq_length*(subseq_numb+1):
                subseq_numb += 1
                new_line = '%s   %d   %d   \n' %(f,subseq_length*(subseq_numb-1),subseq_length*subseq_numb)
                if new_line not in lines:
                    file1 = open(save_choices[label], 'a')
                    file1.write(new_line)
                else:
                    print('Subsequence %d is already in file' %subseq_numb)

    elif c == 'n':
        labels = input("Input labels for each subsequences separated by space for obsid %s: \n1) Good (g) \n2) Bad (b) \n3) Pass (p) \n" %obsid)
        labels = labels.split()
        subseq_numb = 0
        while np.shape(tod_new)[1] > subseq_length*(subseq_numb+1):
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
                else:
                    print('Subsequence %d is already in file' %subseq_numb)
    plt.show()
