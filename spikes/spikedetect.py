import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import scipy.signal
import sys
from scipy.optimize import curve_fit
import scipy.stats
import matplotlib.transforms as mtransforms
import time


class SpikeDetect:
    def __init__(self, data):
        self.data = data


def remove_elevation_gain(X, g, a, c, d, e):
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2


def read_file(filename, feed, sb):
    month = filename[14:21]
    obsid = int(filename[9:13])
    print(obsid)

    path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/' + month + '/'

    with h5py.File(path + filename, 'r') as hdf:
        tod       = np.array(hdf['spectrometer/band_average'])
        el        = np.array(hdf['spectrometer/pixel_pointing/pixel_el'][0])
        az        = np.array(hdf['spectrometer/pixel_pointing/pixel_az'][0])
        features  = np.array(hdf['spectrometer/features'])

    # Removing Tsys measurements     
    boolTsys = (features & 1 << 13 != 8192)
    indexTsys = np.where(boolTsys==False)[0]

    if len(indexTsys) > 0 and (np.max(indexTsys) - np.min(indexTsys)) > 5000:
        boolTsys[:np.min(indexTsys)] = False
        boolTsys[np.max(indexTsys):] = False

    tod       = tod[:,:,boolTsys]
    el        = el[boolTsys]
    az        = az[boolTsys]

    tod = tod[feed,sb]
    where_are_NaNs = np.isnan(tod)
    tod[where_are_NaNs] = 0

    #plt.figure(figsize=(5,4))
    #plt.plot(tod)
    
    num_parts = 24
    part = int(len(el)/num_parts)

    t = np.arange(len(el))
    diff = np.zeros(len(el))
    temp = np.zeros(len(el))
    for i in range(num_parts):
        if i == num_parts-1:
            popt, pcov = curve_fit(remove_elevation_gain, (t[part*i:],el[part\
*i:], az[part*i:]), tod[part*i:])
            g = popt[0]
            a = popt[1]

            temp[part*i:] = g/np.sin(el[part*i:]*np.pi/180) + a*az[part*i:]
            diff[part*i:] = (tod[part*i-1] - temp[part*i-1]) - (tod[part*i]\
                                        - temp[part*i]) + diff[part*(i-1)]
            
        else:
            popt, pcov = curve_fit(remove_elevation_gain, (t[part*i:part*(i+1)],el[part\
                        *i:part*(i+1)], az[part*i:part*(i+1)]), tod[part*i:part*(i+1)])
            g = popt[0]
            a = popt[1]

            temp[part*i:part*(i+1)] = g/np.sin(el[part*i:part*(i+1)]*np.pi/180) + a*az[\
                                                                part*i:part*(i+1)]
            diff[part*i:part*(i+1)] = (tod[part*i-1] - temp[part*i-1]) - (tod[part*i]\
                                                - temp[part*i]) + diff[part*(i-1)]

    # Removing elevation gain                                                       
    tod = tod - temp + diff 
    tod = tod - np.mean(tod)
    
    return tod 
    

def highpass_filter(data, fc=0.1, b=0.08):
    """
    fc : cutoff frequency as a fraction of the sampling rate, (0, 0.5).
    b  : tramsition band as a fraction of the sampling rate, (0, 0.5).
    """

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
    data = np.convolve(data, h, mode='same')

    return data 


def g(X, b, d, c):#, e, f, g, c):
    x, a = X
    #return (a-d-e*x-f*x**2-g*x**3)*np.exp(-(x-b)**2/(2*c**2)) + d + e*x + f*x**2 + g*x**3 
    return (a-d)*np.exp(-(x-b)**2/(2*c**2)) + d 


def peak_detect(y, y_highpass, lag=5, threshold=10, influence=0, lim=3):
    signal = np.zeros(len(y))
    y_filtered = np.copy(y_highpass)
    average = np.zeros(len(y))
    std = np.zeros(len(y))
    average[lag-1] = np.mean(y[:lag])
    std[lag-1] = np.std(y[:lag])

    for i in range(lag, len(y)):
        if (y_highpass[i] - average[i-1]) > threshold*std[i-1]: # Detects only positive spikes
        #if np.abs(y[i] - average[i-1]) > threshold*std[i-1]:
                if y_highpass[i] > average[i-1]:
                    signal[i] = 1
                else:
                    signal[i] = -1
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

    peak_indices = np.split(peak_indices, cut)
    peak_tops = []
            
    if len(peak_indices[0])>0:
        for i in range(len(peak_indices)):
            #peak_tops.append(peak_indices[i][0] + int(len(peak_indices[i])/2))
            peak_tops.append(peak_indices[i][np.argmax(abs(y_highpass[peak_indices[i]]))])


    peak_widths = []
    fitted_peak_tops = []
    for j in range(len(peak_tops)):
        subseq = y[peak_tops[j]-100 : peak_tops[j]+100]
        #subseq = subseq-np.mean(y)
        x = np.arange(len(subseq))
        
        #tod_subseq = tod[peak_tops[j]-100 : peak_tops[j]+100]
        #tod_subseq = tod_subseq - np.mean(tod_subseq)
        
        try:
            popt, pcov = curve_fit(g, (x, np.ones(len(subseq))*subseq[100]), subseq, bounds=((100-lim,-1e4, 0),(100+lim,1e4, 200)))
            #popt2, pcov2 = curve_fit(g, (x, np.ones(len(tod_subseq))*tod_subseq[100]), tod_subseq, bounds=((100-lim, -1e4, -1e4,-1e4, -1e4, 0),(100+lim,1e4,1e4,1e4,1e4,200)))
                
            fitted = g((x, subseq[100]), *popt)
            #fitted2 = g((x, tod_subseq[100]), *popt2)
            half_width = popt[-1]*3 # 3 standard deviations from the peak top in each direction
            fitted_peak_tops.append(peak_tops[j]-100+popt[0])
        
        except:
            print(peak_tops[j])
            half_width = 0
            fitted_peak_tops.append(peak_tops[j])

        #plt.figure()
        #plt.plot(tod_subseq)
        #plt.figure()
        #plt.plot(subseq)
        #plt.plot(fitted)
        #plt.plot(fitted2, '--', label='Normal')
        #plt.show()
            
        peak_widths.append(half_width)
      
    return np.sort(peak_tops), peak_widths, np.sort(fitted_peak_tops), signal


def peak_replace(data, peak_tops, peak_widths):
    new_data = np.copy(data)
    x1_list = [0]
    x2_list = [0]
    for j in range(len(peak_tops)):
        peak_width = np.ceil(peak_widths[j])
        x1 = int(peak_tops[j] - peak_width)
        x2 = int(peak_tops[j] + peak_width)

        if x1 < x2_list[-1]:
            x2_list[-1] = x2
        else:
            x1_list.append(x1)
            x2_list.append(x2)
            
    for j in range(1,len(x1_list)):
        x1 = x1_list[j]
        x2 = x2_list[j]
        
        
        if x2 >= len(data):
            x2 = len(data)-1
        
        if abs(x2 - x1) > 200:
            continue
        
        else:
            y1 = data[x1]
            y2 = data[x2]
        
            m = (y2-y1)/(x2-x1)
            b = (x2*y1 - x1*y2)/(x2-x1)
            
            x = np.arange(x1,x2+1)
            y = m*x + b

            std = np.std(data[1:]-data[:-1])/np.sqrt(2)
            noise = np.random.normal(y, std)
            noise[0] = y1
            noise[-1] = y2

            #plt.plot(x, noise, 'r', alpha=0.7)
            new_data[x1:x2+1] = noise
    
    return new_data

f = open('spikes.txt', 'r')
lines = f.readlines()
#filename = lines[4].split()[0] #9
#filename = 'comap-0006944-2019-07-17-174905.hd5'
#filename = 'comap-0007613-2019-09-10-183037.hd5'
#filename = 'comap-0011507-2020-02-21-174416.hd5' # weather 
#filename = 'comap-0011510-2020-02-21-200733.hd5'
#filename = 'comap-0011419-2020-02-12-182147.hd5'
#filename = 'comap-0008229-2019-10-09-050335.hd5' # broad spike
#filename = 'comap-0008312-2019-10-13-011517.hd5' # broad spike
#filename = 'comap-0011480-2020-02-19-004954.hd5' # weather
#filename = 'comap-0010676-2020-01-22-023457.hd5' # weather
#filename = 'comap-0006541-2019-06-16-232518.hd5' # spike storm
#filename = 'comap-0006653-2019-06-27-000128.hd5' # spike storm 
#filename = 'comap-0006800-2019-07-08-232544.hd5' # spike storm 
filename = 'comap-0006801-2019-07-09-005158.hd5' # spike strom

tod = read_file(filename, 10, 2)


"""
from keras.models import load_model
from create_dataset_copy import preprocess_data

fs = 50
T = 1/fs
subseq_length = int(10*60/T)

model = load_model('../weather/weathernet/weathernet_current.h5')
std = np.loadtxt('../weather/weathernet/weathernet_current_std.txt')

# Make subsequences                                                                                     
sequences = []
subseq_numb = 0
while len(tod) > subseq_length*(subseq_numb+1):
    subseq_numb += 1
    subseq = tod[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
    sequences.append(subseq)

sequences = sequences/std
predictions = model.predict(sequences.reshape(np.shape(sequences)[0], np.shape(sequences)[1], 1))

print('Before spike removal: %.4f %.4f ' %(max(predictions[:,1]), np.median(predictions[:,1])))
"""

fc = 0.001
b = 0.01#0.1#0.001   # If bad weather: 0.1

tod_new = highpass_filter(tod, fc=fc, b=b)

peak_tops, peak_widths, fitted_peak_tops, signal = peak_detect(tod, tod_new, lag=300, threshold=5, influence=0)


plt.figure()
plt.plot(tod)

if len(peak_tops)>0:
    plt.plot(np.arange(len(tod))[peak_tops], tod[peak_tops], 'ro')
"""
plt.figure()
plt.plot(tod_new)
if len(peak_tops)>0:
    plt.plot(np.arange(len(tod_new))[peak_tops], tod_new[peak_tops], 'ro')
"""

tod_final = peak_replace(tod, fitted_peak_tops, peak_widths)

plt.figure()
#plt.plot(tod)
plt.plot(tod_final)#, 'r', alpha=0.7)


tod_final_new = highpass_filter(tod_final, fc=fc, b=0.001)

peak_tops, peak_widths, fitted_peak_tops, signal = peak_detect(tod_final[::-1], tod_final_new[::-1], lag=300, threshold=5, influence=0, lim=3)

fitted_peak_tops = [ len(tod)-x-1 for x in fitted_peak_tops]
peak_tops = [ len(tod)-x-1 for x in peak_tops]
if len(fitted_peak_tops) > 0:
    fitted_peak_tops, peak_widths = zip(*sorted(zip(fitted_peak_tops, peak_widths)))
    peak_tops = np.sort(peak_tops)

tod_final_final = peak_replace(tod_final, fitted_peak_tops, peak_widths)

plt.figure()
plt.plot(tod_final_new)
if len(peak_tops)>0:
    plt.plot(np.arange(len(tod_new))[peak_tops], tod_final_new[peak_tops], 'ro')

"""
plt.figure()
plt.plot(tod_final)
if len(peak_tops)>0:
    plt.plot(np.arange(len(tod_new))[peak_tops], tod_final[peak_tops], 'ro')
"""
plt.figure()
plt.plot(tod_final_final)#, 'r', alpha=0.7)
#if len(peak_tops)>0:
#    plt.plot(np.arange(len(tod_new))[peak_tops], tod_final_final[peak_tops], 'ro')
plt.show()




"""
# Make subsequences                                                                                     
sequences = []
subseq_numb = 0
while len(tod_final_final) > subseq_length*(subseq_numb+1):
    subseq_numb += 1
    subseq = tod_final_final[subseq_length*(subseq_numb-1):subseq_length*subseq_numb]
    sequences.append(subseq)

sequences = sequences/std
predictions = model.predict(sequences.reshape(np.shape(sequences)[0], np.shape(sequences)[1], 1))

print('After spike removal: %.4f %.4f ' %(max(predictions[:,1]), np.median(predictions[:,1])))
  

"""


"""
tod_final_final_new = highpass_filter(tod_final_final, fc=fc, b=0.1)

peak_tops, peak_widths, fitted_peak_tops, signal = peak_detect(tod_final_final, tod_final_final_new, lag=300, threshold=5, influence=0, lim=1)
tod_final_final_final = peak_replace(tod_final_final, fitted_peak_tops, peak_widths)

plt.figure()
plt.plot(tod_final_final_new)
if len(peak_tops)>0:
    plt.plot(np.arange(len(tod_new))[peak_tops], tod_final_final_new[peak_tops], 'ro')

plt.figure()
plt.plot(tod_final_final_final)#, 'r', alpha=0.7)
#if len(peak_tops)>0:
#    plt.plot(np.arange(len(tod_new))[peak_tops], tod_final_final[peak_tops], 'ro')
plt.show()
"""

"""
plt.figure()
plt.plot(tod_new)
plt.plot(np.arange(len(tod_new))[peak_tops_default], tod_new[peak_tops_default], 'ro')

plt.figure()
plt.plot(tod)
plt.plot(np.arange(len(tod))[peak_tops_default], tod[peak_tops_default], 'ro')

plt.show()



print(peak_tops_default)
"""




"""
#make a fitting function that takes x number of peak widths
def makeFunction(indices, data):
    def fitFunction(x, *args):
        #sum of gaussian functions with centers at peak_indices and heights at data[peak_indices] plus a constant for background noise (args[-1])
        return sum([data[indices[i]]*np.exp(-((x-indices[i])**2)/(2*args[i]**2)) for i in range(len(indices))])+args[-1]       
    return fitFunction

x = np.arange(len(tod_new))

plt.figure()
plt.plot(tod-np.mean(tod))
plt.grid()
plt.plot(np.arange(len(tod_new))[peak_tops_default], tod_new[peak_tops_default], 'ro')

f = makeFunction(peak_tops_default, tod_new)

popt, pcov = curve_fit(f, np.arange(len(tod_new)), tod_new, np.ones(len(peak_tops_default)+1))

#standard deviations (widths) of each gaussian peak and the average of the background noise
stdevs, background = popt[:-1], popt[-1]
#covariance of result variables
stdevcov, bgcov = pcov[:-1], pcov[-1]

plt.plot(x, f(x, *popt))


for j in range(len(peak_tops_default)):
    #g = gaussian(x, peak_tops_default[j], stdevs[j])
    #g = g/np.max(g)
    #print(j, np.sum(g>0.001))
    print(j, stdevs[j]*6)

    #print(g[peak_tops_default[j]-50:peak_tops_default[j]+50])

plt.show()

"""
"""
    plt.plot(g*tod_new[peak_tops_default[j]], alpha=0.7)

plt.figure()
plt.plot(tod)
plt.plot(np.arange(len(tod))[peak_tops_default], tod[peak_tops_default], 'ro')
plt.grid()
plt.show()
"""
