import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import h5py 
from plot_tod import read_file
import collections
from scipy.optimize import curve_fit
from scipy import interpolate
from matplotlib import gridspec
import glob
import random
import julian
from dateutil import tz

colors_master = ['#173F5F', '#7ea3be', '#20639b', '#b1cce1', '#3caea3', '#a1d8d2', '#f6d55c', '#faedc0','#ed553b', '#f2c1ba']
font = {'size'   : 13}#, 'family':'serif'}                                      
matplotlib.rc('font', **font)

"""
obsid_spike, feed, sideband, width, ampl, index, mjd_spike, mjd_start = np.loadtxt('data/spike_data/spike_list_v6_beehive34.txt', skiprows=1, unpack=True)

file_subseq = open('data/spike_data/spike_list_ALL_v6.txt', 'a')
for i in range(len(obsid_spike)):
    file_subseq.write('%d        %d        %d        %.4f        %.4f       %d        %f       %f\n' %(obsid_spike[i], feed[i], sideband[i], width[i], ampl[i],  index[i], mjd_spike[i], mjd_start[i]))

"""

def gaussian(X, b, d, c):
    x, a = X
    return (a-d)*np.exp(-(x-b)**2/(2*c**2)) + d

def plot_spike_types(obsid_spikes, index_spikes, feed_spikes, sideband_spikes, width_spikes, which_spikes):
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

    plt.figure(figsize=(13,14))
    i = 0
    for s in which_spikes:
        i+=1
        obsid = obsid_spikes[s]
        index = index_spikes[s]
        feed = feed_spikes[s]
        sideband = sideband_spikes[s]
        width = width_spikes[s]

        obsid_str = '%07d' %obsid
        filename = [s for s in files if obsid_str in s][0]
        tod, mjd, el, az, feed_names, obsid = read_file(filename, keepTsys=True)

        n = 6
        subseq = tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1),int(index)-n:int(index)+n]
        plt.subplot(6,3,i)
        plt.plot(np.arange(index-n, index+n), subseq)
        plt.title('Obsid: %d, feed: %d, sideband: %d' %(obsid, feed, sideband))
        plt.xticks(rotation=30)
        plt.grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.suptitle('Width: 0-3 samples', fontsize=20)
    plt.savefig('figures/spike_typeA.pdf')
    plt.show()


def plot_spikes(filename, index, feeds, sidebands, feed, sideband, width, save=False):
    tod, mjd, el, az, feed_names, obsid = read_file(filename, keepTsys=True)
    feeds_bolean = feeds == feed
    sidebands_bolean = sidebands == sideband
    final_bolean = np.array(feeds_bolean * sidebands_bolean).astype(bool)
    chosen_indices = index[final_bolean]
    chosen_width = width[final_bolean]

    plt.figure(figsize=(6,3))
    plt.plot(tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1)])
    j = 0
    for i in chosen_indices:
        plt.plot(i, tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1),int(i)], 'ro')
        j+=1

    #plt.figure()
    #plt.subplot(121)
    fig = plt.figure(figsize=(12, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 2]) 
    ax0 = plt.subplot(gs[0])
    ax0.plot(np.arange(128800,129800), tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1), 128800:129800])
    plt.xlabel('Sample')
    plt.ylabel('Power')
    plt.grid()
    ax1 = plt.subplot(gs[1])
    ax1.plot(np.arange(128961,129001), tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1), 128961:129001])
    plt.xlabel('Sample')
    plt.ylabel('Power')
    plt.grid()
    plt.suptitle('ObsID: %d, feed: %d, sideband: %d' %(obsid, feed, sideband))
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    if save:
        plt.savefig('figures/spikes_%d_feed%d.pdf' %(obsid, feed))
    plt.show()

def plot_spike_width(filename, index, feeds, sidebands, feed, sideband, width):
    tod, mjd, el, az, feed_names, obsid = read_file(filename, keepTsys=True)
    feeds_bolean = feeds == feed
    sidebands_bolean = sidebands == sideband
    final_bolean = np.array(feeds_bolean * sidebands_bolean).astype(bool)
    chosen_indices = index[final_bolean]
    chosen_width = width[final_bolean]

    plt.figure(figsize=(6,3))
    plt.plot(tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1)])
    j = 0
    for i in chosen_indices:
        n_original = 100
        subseq_old = tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1),int(i)-n_original:int(i)+n_original]
        x_old = np.arange(len(subseq_old))
        n = 1000
        f = interpolate.interp1d(x_old, subseq_old)
        x = np.linspace(x_old[0],x_old[-1], n*2)
        subseq = f(x)
        
        plt.figure()
        plt.plot(x, subseq)
        try:
            popt, pcov = curve_fit(gaussian, (x, np.ones(len(subseq))*subseq[n]), subseq, bounds=((n_original-3, -1e10, 0),(n_original+3, 1e10, 2*n_original)))
            fitted = gaussian((x, subseq[n]), *popt)
            half_width = popt[-1]
            print()
            print('Old width  :', chosen_width[j])
            #print('New width 1:', half_width*2*2)
            #print('New width 2:', half_width*3*2)

            plt.plot(x,fitted)
            plt.show()
            
            #plt.text(i+1, tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1),int(i)], '%.2f' %width[j], horizontalalignment='center', verticalalignment='center', fontsize=6)    

            #if chosen_width[j] < 1:
            #    plt.plot(i, tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1),int(i)], 'ro')
            #    plt.plot(x-100+i, fitted, 'r', alpha=0.7)
            #elif chosen_width[j] > 2.3:
            #    plt.plot(i, tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1),int(i)], 'bo')
            #    plt.plot(x-100+i, fitted, 'b', alpha=0.7)
            
        except:
            pass
        
        j+=1
    plt.show()


def plot_pie(total_obsids, total_weather_obsids, total_spike_obsids, save=False):
    fig1, ax1 = plt.subplots()
    ax1.pie([total_obsids - total_weather_obsids, total_weather_obsids], explode=(0,0), \
            labels=['Good weather', 'Bad weather'], autopct='%.2f%%', textprops=dict(color="w"), \
            shadow=False,wedgeprops = {'linewidth': 2, 'edgecolor':'w'}, startangle=90, \
            colors=[colors_master[4], colors_master[2]])
    ax1.legend(['Good weather', 'Bad weather'], loc='center left', bbox_to_anchor=(0.9, 0, 0.5, 1)) 
    ax1.axis('equal') 
    #plt.subplots_adjust(right=0.5)
    if save:
        plt.savefig('figures/weather_pie.pdf', bbox_inches = "tight")
    
    fig1, ax1 = plt.subplots()
    ax1.pie([total_obsids - total_spike_obsids, total_spike_obsids], explode=(0,0), \
            labels=['Number of obsIDs not containing spikes', 'Number of obsIDs containing spikes'], \
            autopct='%.2f%%', textprops=dict(color="w"), shadow=False, \
            wedgeprops = {'linewidth': 2, 'edgecolor':'w'}, startangle=90, \
            colors=[colors_master[4], colors_master[2]])
    ax1.legend(['Number of obsIDs not containing spikes', 'Number of obsIDs containing spikes'], loc='center left', bbox_to_anchor=(0.9, 0, 0.5, 1)) 
    ax1.axis('equal') 
    #plt.subplots_adjust(right=0.5)
    if save:
        plt.savefig('figures/spike_pie.pdf', bbox_inches = "tight")
    plt.show()

def plot_hist_weather(weather, max_weather=False, median_weather=False, save=False):
    plt.figure(figsize=(6,5))#(6,3))
    plt.hist(weather, bins=100, color=colors_master[3], edgecolor=colors_master[2])
    plt.axvline(x=0.23, color= colors_master[8], linestyle='--')
    plt.text(0.215, 1000, 'Cutoff', horizontalalignment='center', verticalalignment='center', rotation='vertical', color = colors_master[8])
    plt.title('Probability of bad weather')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    #plt.yscale('log', nonposy='clip')
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig('figures/weather_histogram_no_log.pdf')
    
    if max_weather[0]:
        plt.figure(figsize=(5,3))
        plt.hist(max_weather, bins=100, color=colors_master[3], edgecolor=colors_master[2])
        plt.title('Maximum probability of bad weather')
        plt.xlabel('Maximum probability')
        plt.ylabel('Frequency')
        plt.yscale('log', nonposy='clip')
        plt.grid()
        plt.tight_layout()
        if save:
            plt.savefig('figures/max_weather_histogram.pdf')


    if median_weather[0]:
        plt.figure(figsize=(5,3))
        plt.hist(median_weather, bins=100, color=colors_master[3], edgecolor=colors_master[2])
        plt.title('Median probability of bad weather')
        plt.xlabel('Median probability')
        plt.ylabel('Frequency')
        plt.grid()
        plt.yscale('log', nonposy='clip')
        plt.tight_layout()
        if save:
            plt.savefig('figures/median_weather_histogram.pdf')
    
    plt.show()

def plot_hist_spikes(width, ampl, feed=False, save=False):
    if feed[0]:
       feed_20 = feed == 20
       no_feed_20 = feed != 20

       width_only_20 = width[feed_20]
       width_no_20 = width[no_feed_20]
       ampl_only_20 = ampl[feed_20]
       ampl_no_20 = ampl[no_feed_20]
       
       plt.figure(figsize=(5,5))
       _, bins, patches = plt.hist([width_no_20, width_only_20], bins=np.logspace(np.log10(0.1), np.log10(200), 50), label=['Feed 1-19', 'Feed 20'], stacked=True, color=[colors_master[3], colors_master[9]])
       plt.setp(patches[0], edgecolor=colors_master[2])
       plt.setp(patches[1], edgecolor=colors_master[8])
       plt.title('Width of spikes')
       plt.xlabel('Spike width')
       plt.ylabel('Frequency')
       plt.grid()
       plt.legend()
       plt.gca().set_xscale("log")
       plt.tight_layout()
       if save:
            plt.savefig('figures/spike_width_histogram_feed20_10sigma.pdf')


       plt.figure(figsize=(5,5))
       _, bins, patches = plt.hist([ampl_no_20, ampl_only_20], bins=np.logspace(np.log10(5), np.log10(1700), 50), label=['Feed 1-19', 'Feed 20'], stacked=True, color=[colors_master[3], colors_master[9]])
       plt.setp(patches[0], edgecolor=colors_master[2])
       plt.setp(patches[1], edgecolor=colors_master[8])
       plt.title('Absolute value of signal \n to noise of spikes')
       plt.xlabel('|signal to noise|')
       plt.ylabel('Frequency')
       plt.grid()
       plt.legend()
       plt.gca().set_xscale("log")
       plt.tight_layout()
       if save:
            plt.savefig('figures/spike_ampl_histogram_feed20_10sigma.pdf')


    else:
        plt.figure(figsize=(5,5)) 
        plt.hist(width, bins=np.logspace(np.log10(0.1), np.log10(200), 50), color=colors_master[9], edgecolor=colors_master[8])
        #plt.hist(width1, bins=np.logspace(np.log10(0.1), np.log10(200), 50), color=colors_master[3], edgecolor=colors_master[2])
        #plt.hist(width, bins=10, color=colors_master[3], edgecolor=colors_master[2])
        plt.title('Width of spikes')
        plt.xlabel('Spike width')
        plt.ylabel('Frequency')
        plt.grid()
        plt.gca().set_xscale("log")
        #plt.yscale('log', nonposy='clip')
        plt.tight_layout()
        if save:
            plt.savefig('figures/spike_width_histogram_no_log.pdf')

        plt.figure(figsize=(5,5)) 
        plt.hist(abs(ampl), bins=np.logspace(np.log10(5), np.log10(1700), 50), color=colors_master[3], edgecolor=colors_master[2])
        #plt.hist(ampl, bins=2000, color=colors_master[3], edgecolor=colors_master[2])
        plt.title('Absolute value of signal \n to noise of spikes')
        plt.xlabel('|signal to noise|')
        plt.ylabel('Frequency')
        plt.grid()
        plt.gca().set_xscale("log")
        #plt.yscale('log', nonposy='clip')
        plt.tight_layout()
        if save:
            plt.savefig('figures/spike_ampl_histogram_no_log.pdf')
    plt.show()

def plot_number_of_spikes_hist(obsid_spike, feed=False, save=False):
    counter = collections.Counter(obsid_spike)
    #values_to_plot = np.array(list(counter.values()))[np.array(list(counter.values())) < 50]

    if feed[0]:
        feed_20 = feed == 20
        no_feed_20 = feed != 20  
        counter_only_20 = collections.Counter(obsid_spike[feed_20])
        counter_no_20 = collections.Counter(obsid_spike[no_feed_20])
        values_to_plot_only_20 = [2030 if i > 2029 else i for i in counter_only_20.values()]
        values_to_plot_no_20 = [2030 if i > 2029 else i for i in counter_no_20.values()]
        #_, bins, patches = plt.hist(values_to_plot_20, bins=110, color=colors_master[9], edgecolor=colors_master[8], alpha = 0.7)
        _, bins, patches = plt.hist([values_to_plot_no_20, values_to_plot_only_20], bins=110, label=['Feed 1-19', 'Feed 20'], stacked=True, color=[colors_master[3], colors_master[9]])
        plt.text(np.max(values_to_plot_no_20)+140, -75, '+', horizontalalignment='center', verticalalignment='center')   # np.max(values_to_plot_no_20)+400
        plt.setp(patches[0], edgecolor=colors_master[2])
        plt.setp(patches[1], edgecolor=colors_master[8])
        plt.title('Number of spikes per observation')
        plt.xlabel('Number of spikes')
        plt.ylabel('Frequency')
        #plt.yscale('log', nonposy='clip')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if save:
            plt.savefig('figures/number_of_spikes_histogram_feed20_v2.pdf')

        plt.show()

    else:
        values_to_plot = [6030 if i > 6029 else i for i in counter.values()]
        fig, ax = plt.subplots(figsize=(6, 5)) #110
        _, bins, patches = plt.hist(values_to_plot, bins=4100, color=colors_master[3], edgecolor=colors_master[2])
        plt.text(np.max(values_to_plot)+285, -65, '+', horizontalalignment='center', verticalalignment='center')
        plt.title('Number of spikes per observation')
        plt.xlabel('Number of spikes')
        plt.ylabel('Frequency')
        plt.grid()
        plt.tight_layout()
        if save:
            plt.savefig('figures/number_of_spikes_histogram_no_log.pdf')

    plt.show()

def plot_feed_fractions(feed, save=False):
    plt.figure(figsize=(8,3))
    labels, counts = np.unique(feed, return_counts=True)
    plt.bar(labels, counts/len(feed), width=0.7, color=colors_master[3], edgecolor=colors_master[2])
    plt.xticks(np.arange(1,20,1))
    plt.grid()
    plt.tight_layout()
    plt.title('Fraction of spikes in different feeds')
    plt.xlabel('Feed')
    plt.ylabel('Fraction of spikes')
    if save:
        plt.savefig('figures/spike_feed_fraction.pdf', bbox_inches = "tight")
    plt.show()



def cut_spike_values(obsid_spike, feed, sideband, width, ampl, index, mjd_spike, mjd_start, cutoff=5):
    inf_boolean = abs(ampl) < 10000
    obsid_spike = obsid_spike[inf_boolean]
    feed = feed[inf_boolean]
    sideband = sideband[inf_boolean]
    width = width[inf_boolean]
    ampl = ampl[inf_boolean]
    index = index[inf_boolean]
    mjd_spike = mjd_spike[inf_boolean]
    mjd_start = mjd_start[inf_boolean]

    cutoff_boolean = abs(ampl) > cutoff
    obsid_spike = obsid_spike[cutoff_boolean]
    feed = feed[cutoff_boolean]
    sideband = sideband[cutoff_boolean]
    width = width[cutoff_boolean]
    ampl = ampl[cutoff_boolean]
    index = index[cutoff_boolean]
    mjd_spike = mjd_spike[cutoff_boolean]
    mjd_start = mjd_start[cutoff_boolean]

    return obsid_spike, feed, sideband, width, ampl, index, mjd_spike, mjd_start

obsid_spike, feed, sideband, width, ampl, index, mjd_spike, mjd_start = np.loadtxt('data/spike_data/spike_list_ALL_v6.txt', skiprows=1, unpack=True)
obsid, max_weather, median_weather, mjd = np.loadtxt('data/weather_data/weather_list_BEST_copy_obsid.txt', unpack=True)
obsid_subseq, subseq, weather, mjd_subseq = np.loadtxt('data/weather_data/weather_list_BEST_copy.txt', unpack=True)


# Work with same obsIDs for spikes and weather
last_spike_index = np.where(obsid == obsid_spike[-1])[0][0] + 1
obsid = obsid[:last_spike_index]
max_weather = max_weather[:last_spike_index]
median_weather = median_weather[:last_spike_index]
mjd = mjd[:last_spike_index]

print('Last spike obsid:', obsid_spike[-1])
print('Last weather obsid:', obsid[-1])


obsid_spike, feed, sideband, width, ampl, index, mjd_spike, mjd_start = cut_spike_values(obsid_spike, feed, sideband, width, ampl, index, mjd_spike, mjd_start, cutoff=7)

cutoff = 0.23
total_obsids = len(obsid)
total_spike_obsids = len(np.unique(obsid_spike[feed != 20]))
total_weather_obsids = np.sum(max_weather > cutoff)

total_spikes = len(ampl)
positive_spikes = sum(ampl > 0)
negative_spikes = sum(ampl < 0)


obsid_7983 = obsid_spike == 13701
#plot_spikes('comap-0013701-2020-05-26-141228.hd5', index[obsid_7983], feed[obsid_7983], sideband[obsid_7983], 5, 1, width[obsid_7983], save=False)   
plot_spike_width('comap-0013701-2020-05-26-141228.hd5', index[obsid_7983], feed[obsid_7983], sideband[obsid_7983], 12, 1, width[obsid_7983]) 


from_zone = tz.gettz('UTC')
to_zone = tz.gettz('America/Los_Angeles')


month_weather = []
time_weather = []
for i in range(len(obsid)):
    dt_weather = julian.from_jd(mjd[i], fmt='mjd')
    dt_weather = dt_weather.replace(tzinfo=from_zone)
    dt_weather = dt_weather.astimezone(to_zone)
    month_weather.append(dt_weather.month)
    time_weather.append(dt_weather.hour)

datetime_spike = []
month_spike = []
time_spike = []
mjd_spike_no20 = mjd_spike[(feed != 20)]
obsid_no20 = obsid_spike[(feed != 20)]

for i in range(len(mjd_spike_no20)):
    dt_spike = julian.from_jd(mjd_spike_no20[i], fmt='mjd')
    dt_spike = dt_spike.replace(tzinfo=from_zone)
    datetime_spike.append(dt_spike.astimezone(to_zone))
    month_spike.append(datetime_spike[-1].month)
    time_spike.append(datetime_spike[-1].hour)
    #print(obsid_no20[i], datetime[-1], time[-1])
    #print(datetime[-1], month[-1])


#plt.hist(month, bins=5, rwidth=0.8)
labels, counts = np.unique(month_spike, return_counts=True)

n_obsids = []
for m in labels:
    n_obsids.append(len(np.unique(obsid[month_weather == m])))

print(labels)
print(n_obsids)
print(counts)

fig, ax1 = plt.subplots()
ax1.bar(labels, counts/n_obsids, width=0.7, color=colors_master[3], edgecolor=colors_master[2])
ax1.set_ylabel('Number of spikes per obsID')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(labels)
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],  rotation=30)
ax1.set_xlabel('Month')
plt.title('Number of spikes per obsID for \ndifferent months of the year')

from matplotlib.ticker import StrMethodFormatter
ax2 = ax1.twinx()
ax2.plot(np.arange(1,13,1), np.array([3.2, 5.8, 8.2, 11.9, 16.8, 21.6, 24.8, 23.7, 19.3, 13.8, 7.3, 3.2]), 'ro-', alpha = 0.5)
ax2.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°C"))
ax2.set_ylabel('Average temperature at OVRO')
ax2.tick_params(axis='y', labelcolor='black')

#a.xticks(np.arange(1,13,1))
plt.grid()
#plt.xlabel('Month')
plt.tight_layout()
plt.savefig('spikes_month.pdf', bbox_inches = "tight")

plt.figure(figsize = (8,5))
labels, counts = np.unique(time_spike, return_counts=True)
n_obsids = []
for m in labels:
    n_obsids.append(len(np.unique(obsid[time_weather == m])))
plt.bar(labels, counts/n_obsids, width=0.7, color=colors_master[3], edgecolor=colors_master[2])
#plt.xticks(np.arange(0,24,1))
plt.grid()
plt.xlabel('Time')
plt.ylabel('Number of spikes per obsID')
plt.title('Number of spikes per obsID for \ndifferent times of the day')
plt.tight_layout()
xticks = np.arange(0,24,1)
plt.xticks(xticks, ['{0:02d}:00'.format(x) for x in xticks], rotation=45, ha='right')
plt.savefig('spikes_hour.pdf', bbox_inches = "tight")
plt.show()

#plot_hist_spikes(width[feed != 20], ampl[feed != 20], feed=[False, False], save=False)
"""
area1 = (width > 0) & (width < 3) & (feed != 20) 
area1_indices = np.where(area1)[0]

random.seed(24)
plot_spike_types(obsid_spike, index, feed, sideband, width, random.sample(list(area1_indices), k=15))
"""

"""
print('----------------------------------------------------------')
print('Total number of spikes:', total_spikes)
print('Percentage of positive spikes:', positive_spikes/total_spikes*100)
print('Percentage of negative spikes:', negative_spikes/total_spikes*100)
print('Percentage of spikes in feed 20:', sum(feed == 20)/total_spikes*100)
print('Percentage of obsIDs containing spikes:', total_spike_obsids/total_obsids)
print('----------------------------------------------------------')



area1 = (width > 12) & (width < 17) & (feed != 20)
area2 = (width > 1.9) & (width < 3) & (feed != 20)

counter = collections.Counter(obsid_spike[area1])
print(counter.most_common(50))
"""

#obs_list = [12521, 11771, 12520, 11773, 12523, 9941, 9936, 12555, 12547, 12552, 12468, 11764, 11772, 9952, 12553, 9963, 12551, 12554, 11906, 12528, 12742, 12768, 9942, 11907, 9962, 11905]


#obsid_7983 = obsid_spike == 12468
#plot_feed_fractions(feed[obsid_7983], save=False)
#plot_spikes('comap-0012468-2020-04-06-082145.hd5', index[obsid_7983], feed[obsid_7983], sideband[obsid_7983], 5, 1, width[obsid_7983], save=False)
#plot_spike_width('comap-0012521-2020-04-08-064050.hd5', index[obsid_7983], feed[obsid_7983], sideband[obsid_7983], 12, 1, width[obsid_7983])

#counter = collections.Counter(obsid_spike[only11])
#for i in range(len(counter)):
#    if list(counter.values())[i] > 100 and list(counter.values())[i] < 500:
#        print(list(counter.items())[i], list(counter.values())[i])

#for i in range(len(width11)):
#    print(obsid11[i], feed11[i], sideband11[i], ampl11[i], width11[i])

#plot_feed_fractions(feed, save=False)
#plot_number_of_spikes_hist(obsid_spike[feed == 11], feed=[False, False], save=False)
#plot_hist_spikes(width, ampl, feed=[False, False], save=False)
#plot_hist_weather(weather, max_weather, median_weather, save=True)
#plot_pie(total_obsids, total_weather_obsids, total_spike_obsids, save=True)


#obsid_7983 = obsid_spike == 7983
#plot_spikes('comap-0007983-2019-09-28-054657.hd5', index[obsid_7983], feed[obsid_7983], sideband[obsid_7983], 17, 1, width[obsid_7983], save=True)
#plot_spike_width('comap-0007983-2019-09-28-054657.hd5', index[obsid_7983], feed[obsid_7983], sideband[obsid_7983], 17, 1, width[obsid_7983])
#obsid_7422 = obsid_spike == 9851
#plot_spikes('comap-0009851-2019-12-20-002322.hd5', index[obsid_7422], feed[obsid_7422], sideband[obsid_7422], 20, 2, width[obsid_7422])
#plot_spike_width('comap-0009851-2019-12-20-002322.hd5', index[obsid_7422], feed[obsid_7422], sideband[obsid_7422], 20, 2, width[obsid_7422])
"""
obsid_7422 = obsid_spike == 7422
plot_spikes('comap-0007422-2019-08-11-114532.hd5', index[obsid_7422], feed[obsid_7422], sideband[obsid_7422], 20, 2, width[obsid_7422])

obsid_7422 = obsid_spike == 9739
plot_spikes('comap-0009739-2019-12-14-112015.hd5', index[obsid_7422], feed[obsid_7422], sideband[obsid_7422], 20, 2, width[obsid_7422])
plt.show()
"""
"""
obsid_12924 = obsid_spike == 7663
for i in range(np.sum(obsid_12924)):
    print(feed[obsid_12924][i], sideband[obsid_12924][i], width[obsid_12924][i], ampl[obsid_12924][i], index[obsid_12924][i])

f = 'comap-0007663-2019-09-12-161342.hd5'
#plot_spikes(f, index[obsid_12924], feed[obsid_12924], sideband[obsid_12924], 11, 2)
"""
