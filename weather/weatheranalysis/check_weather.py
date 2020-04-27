import h5py
import numpy as np
import matplotlib.pyplot as plt 
import random 

#path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2019-04/'

file1 = open('../../weathernet/data/good_subsequences_ALL.txt', 'r')
lines = file1.read().splitlines()

#lines = []
#for i in range(50):
#    lines.append(random.choice(lines_g))


for line in lines:
    f =  filename = line.split()[0]
    month = f[14:21]
    obsid = int(f[7:13])
    path = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/' + month + '/'
    with h5py.File(path + f, 'r') as hdf:
        try:
            temp = np.mean(np.array(hdf['/hk/array/weather/airTemperature']))
            dewtemp = np.mean(np.array(hdf['/hk/array/weather/dewPointTemp']))
            pressure = np.mean(np.array(hdf['/hk/array/weather/pressure']))
            humidity = np.mean(np.array(hdf['/hk/array/weather/relativeHumidity']))
            rain = np.mean(np.array(hdf['/hk/array/weather/rainToday']))
            status = np.mean(np.array(hdf['/hk/array/weather/status']))
            windspeed = np.mean(np.array(hdf['/hk/array/weather/windSpeed']))
            winddeg = np.mean(np.array(hdf['/hk/array/weather/windDirection']))
            #mjd = np.mean(np.array(hdf['/hk/array/weather/utc']))

            file1 = open('good_weather_means_new.txt', 'a')
            file1.write('%.2f   %.2f   %.2f   %.2f   %.2f   %.2f   %.2f   %.2f\n' %(temp, dewtemp, pressure, humidity, rain, status, windspeed, winddeg))

            print(obsid)
            """
            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(mjd, temp)
            plt.xlabel('MJD')
            plt.title('Temperature [K]')
            
            plt.subplot(2,2,2)
            plt.plot(mjd, dewtemp)
            plt.xlabel('MJD')
            plt.title('Dewpoint temperature [K]')
            
            plt.subplot(2,2,3)
            plt.plot(mjd, humidity)
            plt.xlabel('MJD')
            plt.title('Humidity')

            plt.subplot(2,2,4)
            plt.plot(mjd, rain)
            plt.title('Rain')
            plt.xlabel('MJD')
            plt.tight_layout()
            plt.savefig('checkweather/'+ 'good_' + obsid + '_weather1.png')
            
            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(mjd, pressure)
            plt.xlabel('MJD')
            plt.title('Pressure [Pa]')
        
            plt.subplot(2,2,2)
            plt.plot(mjd,status)
            plt.xlabel('MJD')
            plt.title('Status')

            plt.subplot(2,2,3)
            plt.plot(mjd, windspeed)
            plt.xlabel('MJD')
            plt.title('Wind speed [m/s]')

            plt.subplot(2,2,4)
            plt.plot(mjd, winddeg)
            plt.xlabel('MJD')
            plt.title('Wind degree')
            plt.tight_layout()
            plt.savefig('checkweather/' + 'good_' + obsid + '_weather2.png')
    
            tod = np.array(hdf['spectrometer/tod'][:,:,10, 10000:-5000])
            fig = plt.figure()
        
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
            plt.savefig('checkweather/'+ 'good_' + obsid + '_tod.png')
            """
        except:
            print("passing obsid", obsid)
            pass
