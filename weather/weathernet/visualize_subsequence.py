import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob, os, os.path, sys
import random
import julian
import datetime


if len(sys.argv) > 1:
    lines = sys.argv[1:]  

else:
    f = open('data/testing_data.txt', 'r')
    lines = f.read().splitlines()

print('lines:', lines)

fs = 50 
T = 1/fs 
subseq_length = int(10*60/T)

last_count = 0
count = last_count

for f in lines[last_count:]:
    f = f.replace(" ", "")
    print('--------------------------')
    print('%d' %count)
    count = count + 1
    month = f[14:21]
    obsid = f[9:13]

    path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/' + month + '/'
    with h5py.File(path + f, 'r') as hdf:
        print('ObsID:', obsid)
        try:
            tod = np.array(hdf['spectrometer/band_average'])
            MJD = np.array(hdf['spectrometer/MJD'])
            feeds = np.array(hdf['spectrometer/feeds'])
        except:
            print('No band average')
            continue

        # Removing T-sys measurements                                                           
        try:
            features = np.array(hdf['spectrometer/features'])
        except:
            print('No features')
            continue

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

        max_y = np.max(tod_new[:-1])*1.2 #+ (np.max(tod_new[:-1])-np.min(tod_new[:-1]))*0.2
        min_y = np.min(tod_new[:-1])*0.8 #- (np.max(tod_new[:-1])-np.min(tod_new[:-1]))*0.1

        subseq_numb = 0 
        last_subseq = False

        if np.shape(tod_new)[1] < subseq_length*3:
            print('Too short tod')
            continue

        fig = plt.figure(figsize=(11,5))
        ax = fig.add_subplot(111)
        hours = matplotlib.dates.MinuteLocator(interval = 10)
        h_fmt = matplotlib.dates.DateFormatter('%H:%M:%S')
        #plt.plot(subtime, subseq)                                                                        
        for i in range(np.shape(tod_new)[0]):
            if feeds[i] == 20:
                continue
            try:
                plt.plot(time, tod_new[i,:], linewidth=0.8, label='Feed: %d' %(i+1))
            except:
                pass

        for i in range(9):
            try:
                plt.axvline(time[i*subseq_length], color='black', alpha=0.5)
                if i != 8:
                    plt.text(time[i*subseq_length+int(subseq_length/2)], max_y*0.85, '%d' %(i+1), alpha=0.5) 
            except:
                pass
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt)
        #plt.grid()
        plt.xlabel('UTC (hours)')
        plt.ylabel('Power')
        plt.ylim(min_y,max_y)
        fig.autofmt_xdate()
        #plt.tight_layout()
        plt.title('ObsID: %s' %obsid)
        plt.savefig('figures/'+obsid+'_whole_tod.png')
        plt.show()

        """
        while np.shape(tod_new)[1] > subseq_length*(subseq_numb):
            subseq_numb += 1
            print(subseq_numb)

            # if os.path.exists('bad/'+ f[:-4] + '_%d.hd5' %(subseq_numb)) or  os.path.exists('good/'+ f[:-4] + '_%d.hd5' %(subseq_numb)):
            #    print('Obsid %s, subsequence %d, is already written to file' %(obsid, subseq_numb))
            #    continue

            if np.shape(tod_new)[1] > subseq_length*subseq_numb:
                subseq = tod_new[:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
                subtime = time[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
            else:
                print('Last')
                break
                
            fig = plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            hours = matplotlib.dates.MinuteLocator(interval = 2)
            h_fmt = matplotlib.dates.DateFormatter('%H:%M:%S')
            #plt.plot(subtime, subseq)                                                                        
            for i in range(np.shape(subseq)[0]):
                if feeds[i] == 20:
                    continue
                try:
                    plt.plot(subtime, subseq[i,:], linewidth=0.8, label='Feed: %d' %(i+1))
                except:
                    pass
                    
            ax.xaxis.set_major_locator(hours)
            ax.xaxis.set_major_formatter(h_fmt)
            plt.grid()
            fig.autofmt_xdate()
            plt.xlabel('UTC (hours)')
            plt.ylabel('Power')
            plt.tight_layout()
            plt.ylim(min_y,max_y)
            plt.savefig(obsid+'_subsequence_%d.png' %(subseq_numb))
            plt.show()
        """
