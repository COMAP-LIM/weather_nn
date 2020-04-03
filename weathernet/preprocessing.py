import numpy as np 
from scipy.optimize import curve_fit


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

def elevation_azimuth_template(X, g, a, c, d, e):
    """
    Template for elevation gain and azimuth correlations.                    
    """
    t, el, az = X
    return  g/np.sin(el*np.pi/180) + az*a + c + d*t + e*t**2


def remove_elevation_azimuth_structures(data, el, az):
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
        if (data_highpass[i] - average[i-1]) > threshold*std[i-1]:
            signal[i] = 1
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
            spike_tops.append(spike_indices[i][np.argmax(abs(data[spike_indices[i]]))])
    
    # Estimate the width of each spike    
    spike_widths = []
    fitted_spike_tops = []
    for j in range(len(spike_tops)):
        subseq = data[spike_tops[j]-100:spike_tops[j]+100]
        x = np.arange(len(subseq))

        try:
            popt, pcov = curve_fit(gaussian, (x, np.ones(len(subseq))*subseq[100]), subseq, bounds=((97,-1e10, 0),(103,1e10, 200)))
            fitted = gaussian((x, subseq[100]), *popt)
            half_width = popt[-1]*3 # 3 standard deviations from the spike top in each direction 
            fitted_spike_tops.append(spike_tops[j]-100+popt[0])
        except:
            # Skip spike, could not find optimal values
            half_width = 0
            fitted_spike_tops.append(spike_tops[j])

        spike_widths.append(half_width)

    return spike_tops, fitted_spike_tops, spike_widths


def spike_replace(data, spike_tops, spike_widths):
    """
    Replaces the spikes in the data with noise.
    """
    new_data = np.copy(data)
    
    x1_list = [0]
    x2_list = [0]
    final_peaks = []
    final_widths = []
    
    # Ensures that no spikes overlaps 
    for j in range(len(spike_tops)):
        spike_width = np.ceil(spike_widths[j])
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
            final_peaks.append(x1 + np.argmax(data[x1:x2]))
            final_widths.append(x2-x1)
            
            y1 = data[x1]
            y2 = data[x2]

            # Fit linear function between y1 and y2
            m = (y2-y1)/(x2-x1)
            b = (x2*y1 - x1*y2)/(x2-x1)

            x = np.arange(x1,x2+1)
            y = m*x + b

            std =np.std(data[1:]-data[:-1])/np.sqrt(2)
            noise = np.random.normal(y, std)                                                
            noise[0] = y1
            noise[-1] = y2

            new_data[x1:x2+1] = noise

    return new_data, final_peaks, final_widths

def remove_spikes(data):
    """
    Detects and removes spikes for each feed and sideband within data. 
    """
    data_final_full = np.zeros(np.shape(data))
    for feed in range(np.shape(data)[0]): 
        for sideband in range(np.shape(data)[1]):                                         
            spike_tops, fitted_spike_tops, spike_widths = spike_detect(data[feed,sideband], lag=300, threshold=5, influence=0)
            data_clean = spike_replace(data[feed,sideband], spike_tops, spike_widths)   

            # Repeat for reversed data to detect remaining spikes
            spike_tops, fitted_spike_tops, spike_widths = spike_detect(data_clean[::-1], lag=500, threshold=5, influence=0)
            if len(fitted_spike_tops) > 0:     
                fitted_spike_tops = [ len(data[feed, sideband])-x-1 for x in fitted_spike_tops]
                spike_tops = [ len(data[feed, sideband])-x-1 for x in spike_tops]
                fitted_spike_tops, spike_widths = zip(*sorted(zip(fitted_spike_tops, spike_widths)))
                spike_tops = np.sort(spike_tops)
                fitted_spike_tops = np.sort(fitted_spike_tops)

            data_final = spike_replace(data_clean, fitted_spike_tops, spike_widths)
            data_final_full[feed, sideband, :] = data_final           

    return data_final_full


def remove_spikes_parallell(data):
    spike_tops1, fitted_spike_tops, spike_widths = spike_detect(data, lag=300, threshold=5, influence=0)
    data_clean, final_spikes1, final_widths1 = spike_replace(data, fitted_spike_tops, spike_widths)      

    # Repeat for reversed data to detect remaining spikes
    spike_tops2, fitted_spike_tops, spike_widths = spike_detect(data_clean[::-1], lag=500, threshold=5, influence=0)
    if len(fitted_spike_tops) > 0: 
        fitted_spike_tops = [ len(data)-x-1 for x in fitted_spike_tops] 
        spike_tops2 = [ len(data)-x-1 for x in spike_tops2]    
        fitted_spike_tops, spike_tops2, spike_widths = zip(*sorted(zip(fitted_spike_tops, spike_tops2, spike_widths)))


    data_final, final_spikes2, final_widths2 = spike_replace(data_clean, fitted_spike_tops, spike_widths)        
    all_spikes = final_spikes1 + final_spikes2
    all_widths = final_widths1 + final_widths2
    # all_spikes = spike_tops1 + list(spike_tops2)

    output_list = [all_spikes, all_widths, data_final]
    return output_list
            

def remove_spikes_write_to_file(data, obsid, subseq_start, mjd, mjd_start, num_Tsys_values):
    """
    Detects and removes spikes for each feed and sideband within data. 
    """
    data_final_full = np.zeros(np.shape(data))
    for feed in range(np.shape(data)[0]): 
        for sideband in range(np.shape(data)[1]):                                         
            spike_tops, fitted_spike_tops, spike_widths = spike_detect(data[feed,sideband], lag=300, threshold=5, influence=0)
            data_clean = spike_replace(data[feed,sideband], spike_tops, spike_widths)   

            file_subseq = open('spike_list.txt', 'a')
            for i in range(len(spike_tops)):
                file_subseq.write('%d    %d    %d    %d    %f   %f\n' %(int(obsid), feed, sideband, spike_tops[i] + num_Tsys_values, mjd[subseq_start+spike_tops[i]], mjd_start))

            # Repeat for reversed data to detect remaining spikes
            spike_tops, fitted_spike_tops, spike_widths = spike_detect(data_clean[::-1], lag=500, threshold=5, influence=0)
            if len(fitted_spike_tops) > 0:     
                fitted_spike_tops = [ len(data[feed, sideband])-x-1 for x in fitted_spike_tops]
                spike_tops = [ len(data[feed, sideband])-x-1 for x in spike_tops]
                fitted_spike_tops, spike_widths = zip(*sorted(zip(fitted_spike_tops, spike_widths)))
                spike_tops = np.sort(spike_tops)  
            data_final = spike_replace(data_clean, fitted_spike_tops, spike_widths)
            data_final_full[feed, sideband, :] = data_final           

            for i in range(len(spike_tops)):
                file_subseq.write('%d    %d    %d    %d    %f   %f\n' %(int(obsid), feed, sideband, spike_tops[i] + num_Tsys_values, mjd[subseq_start+spike_tops[i]], mjd_start))


    return data_final_full
