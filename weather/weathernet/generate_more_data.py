import numpy as np 
import h5py 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_data(textfile):
    f = open(textfile, 'r')
    lines = f.readlines()
    data = []
    power_spectrum = []
    obsids = []
    labels = []
    index = []

    for line in lines[:5]:
        print(line)
        filename = line.split()[0]
        index1 = int(line.split()[1])
        index2 = int(line.split()[2])
        month = filename[14:21]
        obsid = int(filename[9:13])

        obsids.append(obsid)
        index.append((index1, index2))
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

        tod = preprocess_data(tod, el, az, obsids[-1], index[-1])
        data.append(tod)

    return np.array(data)


def remove_elevation_gain(X, g, a, c, d, e):
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2




def preprocess_data(data, el, az, obsid, index):
    # Normalizing by dividing each feed on its own mean
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            data[i][j] = data[i][j]/np.nanmean(data[i][j])
    
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

    return data

def generate_data(data):
    ps = np.abs(np.fft.rfft(data))**2
    std = np.sqrt(ps)
    fourier_coeffs = np.sqrt(2)*(np.random.normal(loc=0, scale=std) + 1j*np.random.normal(loc=0, scale=std))
    new_data = np.fft.irfft(fourier_coeffs)

    new_data2 = np.fft.irfft(np.fft.rfft(data))
    print(np.fft.rfft(data))
    
    print(len(new_data2))
    plt.figure()
    plt.plot(new_data)
    plt.figure()
    plt.plot(new_data2)
    plt.show()

    return new_data


        
data = read_data('data/bad_subsequences_ALL.txt')


plt.figure(figsize=(4,3))
plt.title('Data')
plt.plot(data[2])

for i in range(10):
    new_data = generate_data(data[2])
    plt.figure(figsize=(4,3))
    plt.title('Generated data')
    plt.plot(new_data)
    plt.show()


