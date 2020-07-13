import numpy as np
from scipy import stats

good_means_file = 'good_weather_means_new.txt' 
bad_means_file = 'bad_weather_means_new.txt'

temp_g = np.loadtxt(good_means_file, usecols=0)
dewtemp_g = np.loadtxt(good_means_file, usecols=1)
pressure_g = np.loadtxt(good_means_file, usecols=2)
humidity_g = np.loadtxt(good_means_file, usecols=3)
rain_g = np.loadtxt(good_means_file, usecols=4)
status_g = np.loadtxt(good_means_file, usecols=5)
windspeed_g = np.loadtxt(good_means_file, usecols=6)
winddeg_g  = np.loadtxt(good_means_file, usecols=7)

temp_b = np.loadtxt(bad_means_file, usecols=0)
dewtemp_b = np.loadtxt(bad_means_file, usecols=1)
pressure_b = np.loadtxt(bad_means_file, usecols=2)
humidity_b = np.loadtxt(bad_means_file, usecols=3)
rain_b = np.loadtxt(bad_means_file, usecols=4)
status_b = np.loadtxt(bad_means_file, usecols=5)
windspeed_b = np.loadtxt(bad_means_file, usecols=6)
winddeg_b  = np.loadtxt(bad_means_file, usecols=7)

def test_statistic(x1, x2):
    n1 = len(x1)
    n2 = len(x2)

    std1 = np.std(x1)
    std2 = np.std(x2)

    sp = np.sqrt((n1-1)/(n1+n2-2)*std1**2 + (n2-1)/(n1+n2-2)*std2**2)
    se = sp*np.sqrt(1.0/n1 + 1.0/n2)

    if se == 0:
        t = -9
    else:
        t = (np.mean(x1) - np.mean(x2))/se

    df = n1 + n2 - 2
    p = 1 - stats.t.cdf(t,df=df)
    
    return t, p


print()
print('                  t-value  p-value  good     bad')
print('Temperature     : %.2f    %.4f' %test_statistic(temp_g, temp_b), '  %.2f    %.2f' %(np.mean(temp_g), np.mean(temp_b)))
print('Dew temperature : %.2f    %.4f' %test_statistic(dewtemp_g, dewtemp_b), '  %.2f    %.2f' %(np.mean(dewtemp_g), np.mean(dewtemp_b)))
print('Pressure        :  %.2f    %.4f' %test_statistic(pressure_g, pressure_b), '  %.2f   %.2f' %(np.mean(pressure_g), np.mean(pressure_b)))
print('Humidity        :  %.2f    %.4f' %test_statistic(humidity_g, humidity_b), '  %.2f     %.2f' %(np.mean(humidity_g), np.mean(humidity_b)))
print('Rain            : %.2f    %.4f' %test_statistic(rain_g, rain_b), '  %.2f     %.2f' %(np.mean(rain_g), np.mean(rain_b)))
print('Status          : %.2f    %.4f' %test_statistic(status_g, status_b), '  %.2f     %.2f' %(np.mean(status_g), np.mean(status_b)))
print('Wind speed      : %.2f    %.4f' %test_statistic(windspeed_g, windspeed_b), '  %.2f     %.2f' %(np.mean(windspeed_g), np.mean(windspeed_b)))
print('Wind deg        :  %.2f    %.4f' %test_statistic(winddeg_g, winddeg_b), '  %.2f   %.2f' %(np.mean(winddeg_g), np.mean(winddeg_b)))
print()
print('n good:', len(temp_g))
print('n bad :', len(temp_b))
print()
