import h5py
import numpy as np 
import matplotlib.pyplot as plt
import glob 
import sys

np.set_printoptions(threshold=sys.maxsize)

path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2019-08/'   
files = glob.glob(path+'*.hd5')
#f = 'comap-0006564-2019-06-18-002334.hd5'

for f in files:
   with h5py.File(f, 'r') as hdf:
      #el  = np.array(hdf['pointing/elActual'])     
      #az  = np.array(hdf['pointing/azActual'])
      #plt.plot(az, el)    
      #plt.ylabel('el')   
      #plt.xlabel('az')
      #plt.tight_layout()   
      #plt.show()

      tod = np.array(hdf['/spectrometer/band_average'])
      #print(np.shape(tod))
      #vane = np.array(hdf['/hk/antenna0/vane/state'])
      #print(vane)

      plt.plot(tod[1,1])
      plt.show()

      #for i in range(np.shape(tod)[0]):
      #   plt.subplot(5,4,i+1)

      #   for j in range(np.shape(tod)[1]):
      #      plt.plot(tod[i,j])
      #
      #plt.tight_layout()
      #plt.show()

      
      # tod = np.array(hdf['spectrometer/tod'][1,1,3, 10000:-5000])
      # plt.subplot(211)
      # plt.title('Spectrogram')
      # plt.plot(tod)
      # plt.xlabel('Sample')
      # plt.ylabel('Amplitude')
      # plt.subplot(212)
      # plt.specgram(tod,Fs=50)
      # plt.xlabel('Time')
      # plt.ylabel('Frequency')
      # plt.show()   


# for file in files:
#     with h5py.File(file, 'r') as hdf:
#         #ls = list(hdf.keys())
#         #print('List of datasets in this file: \n', ls)
#         tod = np.array(hdf['spectrometer/tod'][1,1,4,10000:-5000])
#         el  = np.array(hdf['pointing/elActual'])
#         az  = np.array(hdf['pointing/azActual'])
#         plt.subplot(4,1,1)
#         plt.plot(tod)
#         plt.subplot(4,1,2)
#         plt.plot(el)
#         plt.ylabel('el')
#         plt.subplot(4,1,3)
#         plt.plot(az)
#         plt.ylabel('az')
#         plt.subplot(4,1,4)
#         plt.plot(az, el)
#         plt.ylabel('el')
#         plt.xlabel('az')
#         plt.tight_layout()
#         plt.show()


# for file in files:
#    with h5py.File(file, 'r') as hdf:
#        #for i in range(17):
#       tod = np.array(hdf['spectrometer/tod'][5,1,4,10000:-5000]) 
#       #print(np.shape(tod))
#       plt.plot(tod)
#       plt.title(file)
#       plt.show()

