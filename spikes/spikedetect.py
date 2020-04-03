import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import scipy.signal
import sys
from scipy.optimize import curve_fit
import scipy.stats
import matplotlib.transforms as mtransforms
import time

def read_file(filename, feed, sideband):
    # Calculating subsequence length      
    fs = 50
    T = 1/fs
    subseq_length = int(10*60/T)

    month = filename[14:21]
    obsid = int(filename[9:13])

    print('Obsid:', obsid)

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

    # Preprocessing          
    where_are_NaNs = np.isnan(tod)
    tod[where_are_NaNs] = 0

    # Make subsequences 
    sequences = []
    subseq_numb = 0
    while np.shape(tod)[2] > subseq_length*(subseq_numb+1):
        subseq_numb += 1
        subseq = tod[:,:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subel = el[:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subaz = az[:,subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
        subseq = remove_elevation_azimuth_structures(subseq, subel, subaz)
        sequences.append(subseq)

    return sequences

def preprocess_data(data):
    new_data = np.copy(data)
    # Normalizing by dividing each feed on its own mean                                          
    for feed in range(np.shape(data)[0]):
        for sideband in range(np.shape(data)[1]):
            new_data[feed][sideband] = data[feed][sideband]/np.nanmean(data[feed][sideband])-1


    # Mean over feeds and sidebands                                                              
    new_data = np.nanmean(new_data, axis=0)
    new_data = np.nanmean(new_data, axis=0)

    # Zero-center data                                                                           
    new_data = new_data - np.mean(new_data)

    return new_data



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
                if np.all(tod[feed, sideband, part*i:part*(i+1)]==0):
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




def highpass_filter(data, fc=0.001, b=0.01):
    """
    fc : cutoff frequency as a fraction of the sampling rate, (0, 0.5).
    b  : tramsition band as a fraction of the sampling rate, (0, 0.5).
    """
    
    # Adding flipped array before and after array to get a periodic function
    data_mirror = np.copy(data)[::-1]
    data_total = np.append(data_mirror, data)
    data_total = np.append(data_total, data_mirror)

    N = int(np.ceil((4/b)))
    if not N % 2: N += 1  # Make sure that N is an odd number 
    n = np.arange(N)

    # Compute sinc filter
    h = np.sinc(2 * fc * (n - (N-1)/2))

    # Compute the Blackman window
    w = np.blackman(N)

    # Compute the windowed-sinc filter
    h = h * w
    h = h / np.sum(h)

    # Turn the low-pass filter into a high-pass filter through spectral inversion
    h = -h
    h[(N-1) // 2] += 1

    # Apply high-pass filter by convolving over the signal
    data_total = np.convolve(data_total, h, mode='same')

    return data_total[len(data):len(data)*2]


def g(X, b, d, c):
    x, a = X
    return (a-d)*np.exp(-(x-b)**2/(2*c**2)) + d 


def spike_detect(y, y_highpass, lag=5, threshold=10, influence=0, lim=3):

    signal = np.zeros(len(y))
    y_filtered = np.copy(y_highpass)
    average = np.zeros(len(y))
    std = np.zeros(len(y))
    average[lag-1] = np.mean(y_highpass[:lag])
    std[lag-1] = np.std(y_highpass[:lag])

    # Find all data points detected as spikes
    for i in range(lag, len(y)):
        if (y_highpass[i] - average[i-1]) > threshold*std[i-1]:
            signal[i] = 1
            y_filtered[i] = influence*y_highpass[i] + (1-influence)*y_filtered[i-1]
        else:
            signal[i] = 0
            y_filtered[i] = y_highpass[i]

        average[i] = np.mean(y_filtered[i-lag+1:i+1])
        std[i] = np.std(y_filtered[i-lag+1:i+1])
    
    peak_indices = np.nonzero(signal)[0]
    cut = []
    for i in range(1, len(peak_indices)):
        if (peak_indices[i] - peak_indices[i-1] > 1):
            cut.append(i)


    # Find the top of each spike
    peak_indices = np.split(peak_indices, cut)
    peak_tops = []
    std_before_top = []
    if len(peak_indices[0])>0:
        for i in range(len(peak_indices)):
            peak_tops.append(peak_indices[i][np.argmax(abs(y[peak_indices[i]]))])
            std_before_top.append(std[peak_tops[-1]])
            #print(peak_tops[-1])

    # Estimate the width of each spike
    peak_widths = []
    fitted_peak_tops = []
    for j in range(len(peak_tops)):
        subseq = y[peak_tops[j]-100 : peak_tops[j]+100]
        x = np.arange(len(subseq))
        
        try:
            popt, pcov = curve_fit(g, (x, np.ones(len(subseq))*subseq[100]), subseq, bounds=((100-lim,-1e10, 0),(100+lim,1e10, 200)))               
            fitted = g((x, subseq[100]), *popt)
            half_width = popt[-1]*3 # 3 standard deviations from the peak top in each direction
            fitted_peak_tops.append(peak_tops[j]-100+popt[0])
        
            #plt.figure()
            #plt.plot(subseq)
            #plt.plot(fitted, alpha=0.7)
            #plt.plot(100, subseq[100], 'ro')
            #plt.show()

        except:
            print('Could not find optimal values for this index:', peak_tops[j])
            half_width = 0
            fitted_peak_tops.append(peak_tops[j])
        
        peak_widths.append(half_width)
    
    return peak_tops, peak_widths, fitted_peak_tops, std_before_top, signal


def spike_replace(data, peak_tops, peak_widths, subseq_num):
    new_data = np.copy(data)
    x1_list = [0]
    x2_list = [0]
    std_list = [0]

    for j in range(len(peak_tops)):
        peak_width = np.ceil(peak_widths[j])
        x1 = int(peak_tops[j] - peak_width)
        x2 = int(peak_tops[j] + peak_width)

        if abs(x2 - x1) > 200:
            continue

        if x1 < x2_list[-1]:
            x2_list[-1] = x2
        else:
            x1_list.append(x1)
            x2_list.append(x2)
            #std_list.append(std_before_top[j])

    for j in range(1,len(x1_list)):
        x1 = x1_list[j]
        x2 = x2_list[j]
        
        if x2 >= len(data):
            x2 = len(data)-1        

        else:
            print(x1 + np.argmax(data[x1:x2]))

            y1 = data[x1]
            y2 = data[x2]
        
            m = (y2-y1)/(x2-x1)
            b = (x2*y1 - x1*y2)/(x2-x1)
            
            x = np.arange(x1,x2+1)
            y = m*x + b

            std =np.std(data[1:]-data[:-1])/np.sqrt(2)
            noise = np.random.normal(y, std)#std_list[j])
            noise[0] = y1
            noise[-1] = y2

            plt.plot(subseq_num*30000 + x, noise, 'r', alpha=0.7)
            new_data[x1:x2+1] = noise
    
    return new_data


def remove_spikes(data, subseq_num):
    fc = 0.001 #0.001
    b = 0.01 #0.01
    
    
    data_highpass = highpass_filter(data, fc=fc, b=b)
    peak_tops, peak_widths, fitted_peak_tops, std_before_top, signal = spike_detect(data, data_highpass, lag=300, threshold=5, influence=0)
    data_clean = spike_replace(data, fitted_peak_tops, peak_widths, subseq_num)

    #fitted_peak_tops = [ int(x) for x in fitted_peak_tops ]
    #if len(peak_tops)>0: 
    #    plt.plot(np.arange(len(data))[peak_tops], data[peak_tops], 'ro')
    #    plt.plot(np.arange(len(data))[fitted_peak_tops], data[fitted_peak_tops], 'bo', alpha=0.3)
    
    data_highpass = highpass_filter(data_clean, fc=fc, b=b)
    peak_tops, peak_widths, fitted_peak_tops, std_before_top, signal = spike_detect(data_clean[::-1], data_highpass[::-1], lag=500, threshold=5, influence=0)
    if len(fitted_peak_tops) > 0:
        fitted_peak_tops = [ len(data)-x-1 for x in fitted_peak_tops]
        peak_tops = [ len(data)-x-1 for x in peak_tops]
        fitted_peak_tops, peak_widths = zip(*sorted(zip(fitted_peak_tops, peak_widths)))
        peak_tops = np.sort(peak_tops)

    #fitted_peak_tops = [ int(x) for x in fitted_peak_tops ]
    #if len(peak_tops)>0: 
    #    plt.plot(np.arange(len(data))[peak_tops], data[peak_tops], 'ro')
    #    plt.plot(np.arange(len(data))[fitted_peak_tops], data[fitted_peak_tops], 'bo', alpha=0.3)
   
            
    data_final_full = spike_replace(data_clean, fitted_peak_tops, peak_widths,subseq_num)
    
    """
    data_final_full = np.zeros(np.shape(data))
    for feed in range(np.shape(data)[0]):
        for sideband in range(np.shape(data)[1]):

            #plt.figure()
            #plt.plot(data[feed,sideband])

            data_highpass = highpass_filter(data[feed, sideband], fc=fc, b=b)
            peak_tops, peak_widths, fitted_peak_tops, std_before_top, signal = spike_detect(data[feed,sideband], data_highpass, lag=300, threshold=5, influence=0)
            data_clean = spike_replace(data[feed,sideband], peak_tops, peak_widths, std_before_top)
  

            
            
            #fitted_peak_tops = [ int(x) for x in fitted_peak_tops ]
            #if len(peak_tops)>0: 
            #    plt.plot(np.arange(len(data[feed, sideband]))[peak_tops], data[feed, sideband][peak_tops], 'ro')
            #    plt.plot(np.arange(len(data[feed, sideband]))[fitted_peak_tops], data[feed, sideband][fitted_peak_tops], 'bo', alpha=0.3)
            
            
            data_highpass = highpass_filter(data_clean, fc=fc, b=b)
            peak_tops, peak_widths, fitted_peak_tops, std_before_top, signal = spike_detect(data_clean[::-1], data_highpass[::-1], lag=500, threshold=5, influence=0)
            if len(fitted_peak_tops) > 0:
                fitted_peak_tops = [ len(data[feed, sideband])-x-1 for x in fitted_peak_tops]
                peak_tops = [ len(data[feed, sideband])-x-1 for x in peak_tops]
                fitted_peak_tops, peak_widths = zip(*sorted(zip(fitted_peak_tops, peak_widths)))
                peak_tops = np.sort(peak_tops)
            
            
            #fitted_peak_tops = [ int(x) for x in fitted_peak_tops ]
            #if len(peak_tops)>0: 
            #    plt.plot(np.arange(len(data[feed, sideband]))[peak_tops], data[feed, sideband][peak_tops], 'ro')
            #    plt.plot(np.arange(len(data[feed, sideband]))[fitted_peak_tops], data[feed, sideband][fitted_peak_tops], 'bo', alpha=0.3)
            
            data_final = spike_replace(data_clean, fitted_peak_tops, peak_widths, std_before_top)
            
            data_final_full[feed, sideband, :] = data_final
            #plt.show()
    """
    return data_final_full

#f = open('spikes.txt', 'r')
#lines = f.readlines()
#filename = lines[4].split()[0] #9
#filename = 'comap-0006944-2019-07-17-174905.hd5'
#filename = 'comap-0007613-2019-09-10-183037.hd5'
#filename = 'comap-0011507-2020-02-21-174416.hd5' # weather 
#filename = 'comap-0011510-2020-02-21-200733.hd5' # weather
#filename = 'comap-0011419-2020-02-12-182147.hd5'
#filename = 'comap-0008229-2019-10-09-050335.hd5' # broad spike
#filename = 'comap-0008312-2019-10-13-011517.hd5' # broad spike
#filename = 'comap-0011480-2020-02-19-004954.hd5' # weather
#filename = 'comap-0010676-2020-01-22-023457.hd5' # weather
#filename = 'comap-0006541-2019-06-16-232518.hd5' # spike storm
#filename = 'comap-0006653-2019-06-27-000128.hd5' # spike storm 
#filename = 'comap-0006800-2019-07-08-232544.hd5' # spike storm 
#filename = 'comap-0006801-2019-07-09-005158.hd5' # spike strom
#filename = 'comap-0008173-2019-10-06-211355.hd5'
#filename = 'comap-0008356-2019-10-14-185407.hd5'
#filename = 'comap-0008357-2019-10-14-200315.hd5'
#filename = 'comap-0009493-2019-11-24-145131.hd5'

filename = 'comap-0008403-2019-10-16-235302.hd5'


from keras.models import load_model

model = load_model('../weather/weathernet/weathernet_current.h5')
std = np.loadtxt('../weather/weathernet/weathernet_current_std.txt')


sequences = read_file(filename, 10, 2)


full_tod = np.zeros((np.shape(sequences)[1], np.shape(sequences)[2], np.shape(sequences)[3]*len(sequences)))
for i in range(len(sequences)):
    full_tod[:,:,np.shape(sequences)[3]*i:np.shape(sequences)[3]*(i+1)] = sequences[i]

feed = 0
sideband = 0

for feed in range(np.shape(full_tod)[0]):
    for sideband in range(np.shape(full_tod)[1]):
        print(feed, sideband)
        plt.figure()
        plt.plot(full_tod[feed,sideband])
        plt.title('Feed: %d, Sideband: %d' %(feed, sideband))

        start_time = time.time()
        sequences1 = []
        for i in range(len(sequences)):
            print(i)
            sequences1.append(remove_spikes(sequences[i][feed,sideband],i))
        print("--- %s seconds ---" % (time.time() - start_time))
        plt.show()
        
"""
full_tod1 = np.zeros((np.shape(sequences)[3]*len(sequences)))
for i in range(len(sequences)):
    full_tod1[np.shape(sequences)[3]*i:np.shape(sequences)[3]*(i+1)] = sequences1[i]
plt.show()

plt.figure()
plt.plot(full_tod1)
plt.show()

"""

"""
prep_seq = prep_seq/std
predictions = model.predict(np.array(prep_seq).reshape(1,len(prep_seq), 1))

seq = preprocess_data(sequence)/std
predictions2 = model.predict(np.array(seq).reshape(1,len(seq),1))

print('After spike removal: %.4f ' %(predictions[0][1]))
print('Before spike removal: %.4f ' %(predictions2[0][1]))

"""
