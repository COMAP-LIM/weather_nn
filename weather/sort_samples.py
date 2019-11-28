import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from shutil import copyfile

path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2019-11/'
files = glob.glob(path+'*.hd5')

save_path = '/mn/stornext/d16/cmbco/comap/marenras/master/training_set/'
save_choices = {'s': 'spikes', 'b': 'bad', 't': 'steps', 'a': 'aircondition', 'g': 'good', 'o': 'other'}


for f in files[:]:
    with h5py.File(f, 'r') as hdf:
        obsid = f[62:66]
        file_name = f[-35:]
        
        tod = np.array(hdf['spectrometer/tod'][:,:,10, 10000:-5000])
        fig = plt.figure(figsize=(6,4))

        plt.subplot(2, 2, 1)
        for i in range(20):
            try:
                plt.plot(tod[i,0,:])
            except:
                pass
    
        plt.subplot(2, 2, 2)
        for i in range(20):
            try:
                plt.plot(tod[i,1,:])
            except:
                pass

        plt.subplot(2, 2, 3)
        for i in range(20):
            try:
                plt.plot(tod[i,2,:])
            except:
                pass


        plt.subplot(2, 2, 4)
        for i in range(20):
            try:
                plt.plot(tod[i,3,:])
            except:
                pass

        plt.suptitle('Obsid: ' + obsid)
        plt.tight_layout()
        plt.show(block=False)

        c = input("Possible categories is: \n(g) good \n(b) bad \n(p) pass \nEnter category for obsid %s: " %(obsid))
        
        if c == 'p':
            pass
        else:
            file1 = open(save_choices[c] + '_new.txt', 'r')
            lines = file1.read().splitlines()
            if file_name in lines:
                print('Already in file')
                pass
            else:
                file1 = open(save_choices[c] + '_new.txt', 'a')
                file1.write(file_name + '\n')
                #copyfile(f, save_path + save_choices[c] + file_name)
        plt.close()

