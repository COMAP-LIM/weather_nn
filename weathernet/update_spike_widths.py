import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
from plot_tod import read_file
from scipy.optimize import curve_fit
from scipy import interpolate
import glob, os


def gaussian(X, b, d, c):
    """
    Gaussian function.
    """
    x, a = X
    return (a-d)*np.exp(-(x-b)**2/(2*c**2)) + d


def plot_spike_width(obsid, index, feed, sideband, width, files):
    obsid_str = '%07d' %obsid    
    filename = [s for s in files if obsid_str in s][0]
    
    tod, mjd, el, az, feed_names, obsid = read_file(filename, keepTsys=True)

    n_original = 100
    subseq_old = tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1),int(index)-n_original:int(index)+n_original]
    x_old = np.arange(len(subseq_old))
    n = 1000
    f = interpolate.interp1d(x_old, subseq_old)
    x = np.linspace(x_old[0],x_old[-1], n*2)
    subseq = f(x)

    try:
        popt, pcov = curve_fit(gaussian, (x, np.ones(len(subseq))*subseq[n]), subseq, bounds=((n_original-3, -1e10, 0),(n_original+3, 1e10, 2*n_original)))
        fitted = gaussian((x, subseq[n]), *popt)
        half_width = popt[-1]*2 # 3 standard deviations from the spike in each direction
    except:
        half_width = 0    
    return half_width*2


folders = glob.glob('/mn/stornext/d16/cmbco/comap/pathfinder/ovro/20*/')
for el in folders:
    if len(el) > 53:
        folders.remove(el)
    
# Make list with files                                                                                         
files = []
for el in folders:
    #files.extend(glob.glob('%s/*.hd5' %el))
    files.extend(glob.glob1(el, '*.hd5'))
files.sort()

obsid_spike, feed, sideband, width, ampl, index, mjd_spike, mjd_start = np.loadtxt('data/spike_data/spike_list_ALL_v4.txt', skiprows=1, unpack=True)

for i in range(len(obsid_spike)):
    new_width = plot_spike_width(obsid_spike[i], index[i], feed[i], sideband[i], width[i], files)


"""
obsid_7422 = obsid_spike == 9851
#plot_spikes('comap-0009851-2019-12-20-002322.hd5', index[obsid_7422], feed[obsid_7422], sideband[obsid_7422], 20,\
 2, width[obsid_7422])                                                                                             
plot_spike_width('comap-0009851-2019-12-20-002322.hd5', index[obsid_7422], feed[obsid_7422], sideband[obsid_7422],\
 20, 2, width[obsid_7422])
"""
