import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import interpolate

def scale(data):
    """
    Scales and zero-center the data 
    """
    new_data = np.copy(data)
    # Normalizing by dividing each feed on its own mean 
    for feed in range(np.shape(data)[0]):
        for sideband in range(np.shape(data)[1]):
            if np.all(data[feed, sideband]==0):
                continue
            new_data[feed][sideband] = data[feed][sideband]/np.nanmean(data[feed][sideband])-1


    # Mean over feeds and sidebands            
    new_data = np.nanmean(new_data, axis=0)
    new_data = np.nanmean(new_data, axis=0)

    # Zero-center data 
    new_data = new_data - np.mean(new_data)

    return new_data


def scale_two_mean(data):
    """
    Scales and zero-center the data 
    """
    new_data = np.copy(data)
    # Normalizing by dividing each feed on its own mean 
    for feed in range(np.shape(data)[0]):
        for sideband in range(np.shape(data)[1]):
            if np.all(data[feed, sideband]==0):
                continue
            new_data[feed][sideband] = data[feed][sideband]/np.nanmean(data[feed][sideband])-1

    # Mean over feeds and sidebands            
    new_data = np.nanmean(new_data, axis=1)
    new_data1 = np.nanmean(new_data[:int(np.shape(new_data)[0]/2)], axis=0)
    new_data2 = np.nanmean(new_data[int(np.shape(new_data)[0]/2):], axis=0)

    # Zero-center data 
    new_data1 = new_data1 - np.mean(new_data1)
    new_data2 = new_data2 - np.mean(new_data2)

    new_data = np.vstack([new_data1, new_data2]).T

    return new_data

def scale_all_feeds(data):
    """
    Scales and zero-center the data 
    """
    new_data = np.copy(data)
    # Normalizing by dividing each feed on its own mean 
    for feed in range(np.shape(data)[0]):
        for sideband in range(np.shape(data)[1]):
            if np.all(data[feed, sideband]==0):
                continue
            new_data[feed][sideband] = data[feed][sideband]/np.nanmean(data[feed][sideband])-1


    # Mean over feeds and sidebands            
    new_data = np.nanmean(new_data, axis=1)

    # Chose first 18 feeds to get same dimensions of each sample
    new_data = new_data[:16].T

    return new_data



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


def highpass_filter(data, fc=0.001, b=0.01):
    """ 
    High pass filters the input data. 
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


def gaussian(X, b, d, c):
    """
    Gaussian function. 
    """
    x, a = X
    return (a-d)*np.exp(-(x-b)**2/(2*c**2)) + d


def spike_detect(data, lag=5, threshold=10, influence=0):
    """
    Detects the spikes in the data, and estimates the width of the spikes. 
    """
    data_highpass = highpass_filter(data, fc = 0.001, b = 0.01)

    data_filtered = np.copy(data_highpass)
    signal = np.zeros(len(data))
    average = np.zeros(len(data))
    std = np.zeros(len(data))

    average[lag-1] = np.mean(data_highpass[:lag])
    std[lag-1] = np.std(data_highpass[:lag])

    # Find all data points detected as spikes 
    for i in range(lag, len(data)):
        if data_highpass[i] - average[i-1] > threshold*std[i-1]:
            signal[i] = 1
            data_filtered[i] = influence*data_highpass[i] + (1-influence)*data_filtered[i-1]
        elif data_highpass[i] - average[i-1] < - threshold*std[i-1]:
            signal[i] = -1
            data_filtered[i] = influence*data_highpass[i] + (1-influence)*data_filtered[i-1]
        else:
            signal[i] = 0
            data_filtered[i] = data_highpass[i]

        average[i] = np.mean(data_filtered[i-lag+1:i+1])
        std[i] = np.std(data_filtered[i-lag+1:i+1])

    # Cuts each detected spike area in separate segments 
    spike_indices = np.nonzero(signal)[0]
    cut = []
    for i in range(1, len(spike_indices)):
        if (spike_indices[i] - spike_indices[i-1] > 1):
            cut.append(i)
    spike_indices = np.split(spike_indices, cut)

    # Find the top point of each spike 
    spike_tops = []
    if len(spike_indices[0])>0:
        for i in range(len(spike_indices)):
            if signal[spike_indices[i][0]] == 1:
                spike_tops.append(spike_indices[i][np.argmax(abs(data[spike_indices[i]]))])
            elif signal[spike_indices[i][0]] == -1:
                spike_tops.append(spike_indices[i][np.argmin(abs(data[spike_indices[i]]))])

    # Estimate the width of each spike and calculate signal to noise
    spike_widths = []
    fitted_spike_tops = []
    spike_ampls = []
    std_noise = np.std(data[1:]-data[:-1])/np.sqrt(2)
    for j in range(len(spike_tops)):
        n_original = 100
        subseq_old = data[spike_tops[j]-n_original:spike_tops[j]+n_original]
        subseq_highpass = data_highpass[spike_tops[j]-n_original:spike_tops[j]+n_original]
        x_old = np.arange(len(subseq_old))
        n = 1000
        f = interpolate.interp1d(x_old, subseq_old)
        x = np.linspace(x_old[0],x_old[-1], n*2)
        subseq = f(x)

        try:
            popt, pcov = curve_fit(gaussian, (x, np.ones(len(subseq))*subseq[n]), subseq, bounds=((n_original-3, -1e10, 0),(n_original+3, 1e10, 2*n_original)))
            fitted = gaussian((x, subseq[n]), *popt)
            half_width = popt[-1]*2 # 3 standard deviations from the spike top in each direction
            fitted_spike_tops.append(spike_tops[j]-100+popt[0])
            #spike_ampls.append((subseq[n] - popt[1])/std_noise)
            spike_ampls.append((data_highpass[spike_tops[j]] - average[spike_tops[j]-1])/std[spike_tops[j]-1])
        except:
            # Skip spike, could not find optimal values
            half_width = 0
            fitted_spike_tops.append(spike_tops[j])
            spike_ampls.append(0)

        spike_widths.append(half_width)
    
    return spike_tops, fitted_spike_tops, spike_widths, spike_ampls


def spike_replace(data, spike_tops, spike_widths, plot=False):
    """
    Replaces the spikes in the data with noise.
    """
    new_data = np.copy(data)
    
    x1_list = [0]
    x2_list = [0]
   
    # Ensures that no spikes overlaps 
    for j in range(len(spike_tops)):
        spike_width = np.ceil(spike_widths[j]/2*3)
        x1 = int(spike_tops[j] - spike_width)
        x2 = int(spike_tops[j] + spike_width)

        # Skip spikes broader than 200 data points
        if abs(x2 - x1) > 200:
            continue

        if x1 < x2_list[-1]:
            x2_list[-1] = x2
        else:
            x1_list.append(x1)
            x2_list.append(x2)

    # Replaces spikes with noise
    for j in range(1,len(x1_list)):
        x1 = x1_list[j]
        x2 = x2_list[j]

        if x2 >= len(data):
            x2 = len(data)-1

        else:
            y1 = data[x1]
            y2 = data[x2]

            # Fit linear function between y1 and y2
            m = (y2-y1)/(x2-x1)
            b = (x2*y1 - x1*y2)/(x2-x1)

            x = np.arange(x1,x2+1)
            y = m*x + b

            std = np.std(data[1:]-data[:-1])/np.sqrt(2)            
            noise = np.random.normal(y, std)

            try:
                noise[0] = y1
                noise[-1] = y2
            
                if plot:
                    plt.plot(x, noise, 'r')

                new_data[x1:x2+1] = noise
                
            except:
                pass

    return new_data

def remove_spikes(data):
    """
    Detects and removes spikes for each feed and sideband within data. 
    """
    data_final_full = np.zeros(np.shape(data))
    for feed in range(np.shape(data)[0]): 
        for sideband in range(np.shape(data)[1]):                                         
            spike_tops, fitted_spike_tops, spike_widths, spike_ampls = spike_detect(data[feed,sideband], lag=300, threshold=5, influence=0)
            data_clean = spike_replace(data[feed,sideband], spike_tops, spike_widths)   

            # Repeat for reversed data to detect remaining spikes
            spike_tops, fitted_spike_tops, spike_widths, spike_ampls = spike_detect(data_clean[::-1], lag=500, threshold=5, influence=0)
            if len(fitted_spike_tops) > 0:     
                fitted_spike_tops = [ len(data[feed, sideband])-x-1 for x in fitted_spike_tops]
                spike_tops = [ len(data[feed, sideband])-x-1 for x in spike_tops]
                fitted_spike_tops, spike_tops, spike_widths, spike_ampls = zip(*sorted(zip(fitted_spike_tops, spike_tops, spike_widths, spike_ampls)))

            data_final = spike_replace(data_clean, fitted_spike_tops, spike_widths)
            data_final_full[feed, sideband, :] = data_final           

    return data_final_full


def remove_spikes_parallell(data, plot=False):
    spike_tops1, fitted_spike_tops1, spike_widths1, spike_ampls1 = spike_detect(data, lag=300, threshold=5, influence=0)
    data_clean = spike_replace(data, fitted_spike_tops1, spike_widths1, plot)          


    # Repeat for reversed data to detect remaining spikes
    spike_tops2, fitted_spike_tops2, spike_widths2, spike_ampls2 = spike_detect(data_clean[::-1], lag=500, threshold=5, influence=0)

    if len(fitted_spike_tops2) > 0: 
        fitted_spike_tops2 = [ len(data)-x-1 for x in fitted_spike_tops2] 
        spike_tops2 = [ len(data)-x-1 for x in spike_tops2]    
        fitted_spike_tops2, spike_tops2, spike_widths2, spike_ampls2 = zip(*sorted(zip(fitted_spike_tops2, spike_tops2, spike_widths2, spike_ampls2)))

    data_final = spike_replace(data_clean, fitted_spike_tops2, spike_widths2, plot)        


    all_spikes = spike_tops1 + list(spike_tops2)
    all_widths = spike_widths1 + list(spike_widths2)
    all_ampls = spike_ampls1 + list(spike_ampls2)

    output_list = [all_spikes, all_widths, all_ampls, data_final]
    return output_list
            
