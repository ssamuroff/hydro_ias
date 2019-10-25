import numpy as np
import pylab as plt
from pylab import gca
plt.switch_backend('pdf')
plt.style.use('y1a1')
from matplotlib import rcParams
import fitsio as fi

rcParams['xtick.major.size'] = 3.5
rcParams['xtick.minor.size'] = 1.7
rcParams['ytick.major.size'] = 3.5
rcParams['ytick.minor.size'] = 1.7
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'



x0 = np.array([-18.989, -19.664, -20.270, -21.153])
A0 = np.array([3.045724e-01, 2.017377, 1.057999, 1.244393])
dA0 = np.array([6.819294e-01, 6.731764e-01, 7.058361e-01, 6.432579e-01])
x1 = np.array([-19.409, -20.081, -20.703, -21.650])
A1 = np.array([1.102331, 2.662284, 2.247608, 2.280453])
dA1 = np.array([6.430263e-01,5.635402e-01,5.922032e-01,5.904630e-01])
x2 = np.array([-19.803, -20.445, -21.078, -22.046])
A2 = np.array([ 1.786912, 1.719141, 1.616444, 2.016498])
dA2 = np.array([5.727869e-01, 6.220391e-01, 6.151392e-01,5.876449e-01])
x3 = np.array([-20.189, -20.794, -21.429, -22.429])
A3 = np.array([1.976067,2.344858,3.481462,2.932006  ])
dA3 = np.array([7.123555e-01,7.055276e-01,6.140195e-01,6.213617e-01])


plt.close()
plt.subplot(111)
plt.errorbar(x0, A0, yerr=dA0, color='steelblue', linestyle='none', label='$z=0.0$', marker='.')
plt.errorbar(x1, A1, yerr=dA1, color='plum', linestyle='none', label='$z=0.3$', marker='^')
plt.errorbar(x2, A2, yerr=dA2, color='royalblue', linestyle='none', label='$z=0.62$', marker='x')
plt.errorbar(x3, A3, yerr=dA3, color='midnightblue', linestyle='none', label='$z=1.00$', marker='+')

#plt.xlim(0,1.1)
plt.xticks(visible=True)
#plt.yticks([1.5,2,2.5,3.0], fontsize=16)
plt.ylim(-4,7.2)
plt.axhline(0,color='k',ls=':')
plt.legend(loc='lower left', fontsize=16)

plt.ylabel(r'$A_{i}$', fontsize=16)

plt.xlabel('Absolute Magnitude $M_r$', fontsize=16)
plt.subplots_adjust(hspace=0,wspace=0,bottom=0.14,left=0.14, right=0.98, top=0.98)
plt.savefig('ai_blue_mag.pdf')
plt.savefig('ai_blue_mag.png')





dA0 = np.array([6.819294e-01, 6.731764e-01, 7.058361e-01, 6.432579e-01])
x1 = np.array([20.767752206044392, 20.09575220604439, 19.47375220604439, 18.526752206044392])
A1 = np.array([1.102331, 2.662284, 2.247608, 2.280453])
dA1 = np.array([6.430263e-01,5.635402e-01,5.922032e-01,5.904630e-01])
x2 = np.array([22.261144625346606, 21.619144625346607, 20.986144625346608, 20.018144625346608])
A2 = np.array([ 1.786912, 1.719141, 1.616444, 2.016498])
dA2 = np.array([5.727869e-01, 6.220391e-01, 6.151392e-01,5.876449e-01])
x3 = np.array([23.125349242637284, 22.520349242637284, 21.885349242637286, 20.885349242637286])
A3 = np.array([1.976067,2.344858,3.481462,2.932006  ])
dA3 = np.array([7.123555e-01,7.055276e-01,6.140195e-01,6.213617e-01])


plt.close()
plt.subplot(111)
plt.errorbar(x1, A1, yerr=dA1, color='plum', linestyle='none', label='$z=0.3$', marker='^')
plt.errorbar(x2, A2, yerr=dA2, color='royalblue', linestyle='none', label='$z=0.62$', marker='x')
plt.errorbar(x3, A3, yerr=dA3, color='midnightblue', linestyle='none', label='$z=1.00$', marker='+')

#plt.xlim(0,1.1)
plt.xticks(visible=True)
#plt.yticks([1.5,2,2.5,3.0], fontsize=16)
plt.ylim(-4,7.2)
plt.axhline(0,color='k',ls=':')
plt.legend(loc='lower left', fontsize=16)

plt.ylabel(r'$A_{i}$', fontsize=16)

plt.xlabel('Apparent Magnitude $r$', fontsize=16)
plt.subplots_adjust(hspace=0,wspace=0,bottom=0.14,left=0.14, right=0.98, top=0.98)
plt.savefig('ai_blue_apmag.pdf')
plt.savefig('ai_blue_apmag.png')