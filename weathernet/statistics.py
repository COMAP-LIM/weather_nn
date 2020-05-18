import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import h5py 
from plot_tod import read_file
import collections
from scipy.optimize import curve_fit
from scipy import interpolate

colors_master = ['#173F5F', '#7ea3be', '#20639b', '#b1cce1', '#3caea3', '#a1d8d2', '#f6d55c', '#faedc0','#ed553b', '#f2c1ba']
font = {'size'   : 15}#, 'family':'serif'}                                      
matplotlib.rc('font', **font)

"""
obsid_spike, feed, sideband, width, ampl, index, mjd_spike, mjd_start = np.loadtxt('data/spike_data/spike_list_v5_beehive30.txt', skiprows=1, unpack=True)

file_subseq = open('data/spike_data/spike_list_ALL_v5.txt', 'a')
for i in range(len(obsid_spike)):
    file_subseq.write('%d        %d        %d        %.4f        %.4f       %d        %f       %f\n' %(obsid_spike[i], feed[i], sideband[i], width[i], ampl[i],  index[i], mjd_spike[i], mjd_start[i]))

"""

def gaussian(X, b, d, c):
    x, a = X
    return (a-d)*np.exp(-(x-b)**2/(2*c**2)) + d


def plot_spikes(filename, index, feeds, sidebands, feed, sideband, width):
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
        if chosen_width[j] < 1:
            plt.plot(i, tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1),int(i)], 'ro')
        if chosen_width[j] > 2.3:
            plt.plot(i, tod[int(np.where(feed_names == feed)[0][0]), int(sideband-1),int(i)], 'bo')
        #print(tod[int(np.where(feed_names == feed)[0][0]),int(sideband-1),int(i)])
        #plt.show()
        j+=1

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
            print('New width 1:', half_width*2*2)
            print('New width 2:', half_width*3*2)

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
            plt.savefig('figures/spike_width_histogram_feed20.pdf')


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
            plt.savefig('figures/spike_ampl_histogram_feed20.pdf')


    else:
        plt.figure(figsize=(5,5))
        #plt.hist(width, bins=np.logspace(np.log10(0.1), np.log10(200), 50), color=colors_master[3], edgecolor=colors_master[2])
        plt.hist(width, bins=2000, color=colors_master[3], edgecolor=colors_master[2])
        plt.title('Width of spikes')
        plt.xlabel('Spike width')
        plt.ylabel('Frequency')
        plt.grid()
        #plt.gca().set_xscale("log")
        #plt.yscale('log', nonposy='clip')
        plt.tight_layout()
        if save:
            plt.savefig('figures/spike_width_histogram_no_log.pdf')

        plt.figure(figsize=(5,5))
        #plt.hist(abs(ampl), bins=np.logspace(np.log10(5), np.log10(1700), 50), color=colors_master[3], edgecolor=colors_master[2])
        plt.hist(ampl, bins=2000, color=colors_master[3], edgecolor=colors_master[2])
        plt.title('Absolute value of signal \n to noise of spikes')
        plt.xlabel('|signal to noise|')
        plt.ylabel('Frequency')
        plt.grid()
        #plt.gca().set_xscale("log")
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
        values_to_plot_only_20 = [6030 if i > 6029 else i for i in counter_only_20.values()]
        values_to_plot_no_20 = [6030 if i > 6029 else i for i in counter_no_20.values()]
        #_, bins, patches = plt.hist(values_to_plot_20, bins=110, color=colors_master[9], edgecolor=colors_master[8], alpha = 0.7)
        _, bins, patches = plt.hist([values_to_plot_no_20, values_to_plot_only_20], bins=110, label=['Feed 1-19', 'Feed 20'], stacked=True, color=[colors_master[3], colors_master[9]])
        plt.setp(patches[0], edgecolor=colors_master[2])
        plt.setp(patches[1], edgecolor=colors_master[8])
        plt.title('Number of spikes per observation')
        plt.xlabel('Number of spikes')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if save:
            plt.savefig('figures/number_of_spikes_histogram_feed20.pdf')

        plt.show()

    else:
        values_to_plot = [6030 if i > 6029 else i for i in counter.values()]
        fig, ax = plt.subplots(figsize=(6, 5)) 
        _, bins, patches = plt.hist(values_to_plot, bins=110, color=colors_master[3], edgecolor=colors_master[2])
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
    plt.xticks(np.arange(1,21,1))
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

obsid_spike, feed, sideband, width, ampl, index, mjd_spike, mjd_start = np.loadtxt('data/spike_data/spike_list_ALL_v4.txt', skiprows=1, unpack=True)
obsid, max_weather, median_weather, mjd = np.loadtxt('data/weather_data/weather_list_BEST_copy_obsid.txt', unpack=True)
obsid_subseq, subseq, weather, mjd = np.loadtxt('data/weather_data/weather_list_BEST_copy.txt', unpack=True)


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
total_spike_obsids = len(np.unique(obsid_spike))
total_weather_obsids = np.sum(max_weather > cutoff)

total_spikes = len(ampl)
positive_spikes = sum(ampl > 0)
negative_spikes = sum(ampl < 0)

print('Total number of spikes:', total_spikes)
print('Percentage of positive spikes:', positive_spikes/total_spikes*100)
print('Percentage of negative spikes:', negative_spikes/total_spikes*100)
print('Percentage of obsIDs containing spikes:', total_spike_obsids/total_obsids)




#plot_number_of_spikes_hist(obsid_spike, feed=feed, save=False)
#plot_hist_spikes(width, ampl, feed=feed, save=False)
#plot_hist_weather(weather, max_weather, median_weather, save=True)
#plot_pie(total_obsids, total_weather_obsids, total_spike_obsids, save=True)



"""
spikes_top1 = np.where(np.logical_and(width>=0.6, width<=1.2))[0]#(width>=0.9, width<=1.0))[0]
spikes_top2 = np.where(np.logical_and(width>=2.5, width<=2.7))[0]

ampls_top1 = ampl[spikes_top1]
print(len(spikes_top1)/len(ampl))
print(len(spikes_top2)/len(ampl))
print('Negatives:', sum(ampls_top1 < 0)/len(ampls_top1))
print('Positives:', sum(ampls_top1 > 0)/len(ampls_top1))

count = 0
count_20 = 0
for i in range(len(spikes_top2)):
    if feed[spikes_top2[i]] == 20:
        #print(obsid_spike[spikes_top1[i]])
        count_20 += 1

        if width[spikes_top2[i]] > 1:
            print(obsid_spike[spikes_top2[i]], feed[spikes_top2[i]], sideband[spikes_top2[i]], width[spikes_top2[i]], ampl[spikes_top2[i]])
            count += 1

   
#print('Precentage:', count/len(spikes_top1))
print('Total feed 20:', count_20)
print('Total:', len(spikes_top1))
print('Precentage feed 20:', count_20/len(spikes_top1))


print('Precentage of feed 20 in all data', sum(feed == 20)/len(ampl))
print('Total number of spikes:', len(ampl))




counter = collections.Counter(obsid_spike[feed == 20])
for i in range(len(counter.keys())):
    print(list(counter.keys())[i], list(counter.values())[i])

#obsid_7422 = obsid_spike == 9851
#plot_spikes('comap-0009851-2019-12-20-002322.hd5', index[obsid_7422], feed[obsid_7422], sideband[obsid_7422], 20, 2, width[obsid_7422])
#plot_spike_width('comap-0009851-2019-12-20-002322.hd5', index[obsid_7422], feed[obsid_7422], sideband[obsid_7422], 20, 2, width[obsid_7422])

obsid_7422 = obsid_spike == 7422
plot_spikes('comap-0007422-2019-08-11-114532.hd5', index[obsid_7422], feed[obsid_7422], sideband[obsid_7422], 20, 2, width[obsid_7422])

obsid_7422 = obsid_spike == 9739
plot_spikes('comap-0009739-2019-12-14-112015.hd5', index[obsid_7422], feed[obsid_7422], sideband[obsid_7422], 20, 2, width[obsid_7422])
plt.show()
"""

"""
cutoff = 0.23
total_obsids = len(obsid)
total_spike_obsids = len(np.unique(obsid_spike))
total_weather_obsids = np.sum(max_weather > cutoff)

total_spikes = len(ampl)
positive_spikes = sum(ampl > 0)
negative_spikes = sum(ampl < 0)

print('Total number of spikes:', total_spikes)
print('Percentage of positive spikes:', positive_spikes/total_spikes*100)
print('Percentage of negative spikes:', negative_spikes/total_spikes*100)
print('Percentage of obsIDs containing spikes:', total_spike_obsids/total_obsids)



#plot_number_of_spikes_hist(obsid_spike, save=False)
#plot_hist_spikes(width, ampl, save=False)
#plot_hist_weather(weather, max_weather, median_weather, save=True)
#plot_pie(total_obsids, total_weather_obsids, total_spike_obsids, save=True)
"""

"""
obsid_12924 = obsid_spike == 7663
for i in range(np.sum(obsid_12924)):
    print(feed[obsid_12924][i], sideband[obsid_12924][i], width[obsid_12924][i], ampl[obsid_12924][i], index[obsid_12924][i])

f = 'comap-0007663-2019-09-12-161342.hd5'
#plot_spikes(f, index[obsid_12924], feed[obsid_12924], sideband[obsid_12924], 11, 2)
"""
