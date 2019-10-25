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



x = np.array([0.06,0.3,0.625,1.0])
A_red_high = np.array([3.48817, 5.31122, 6.09772, 7.19743])
A_red_low = np.array([-0.26807, 0.895864, -0.825007, -0.00632499])
A_blue_high = np.array([1.72815, 2.24772, 2.05666, 3.11556])
A_blue_low = np.array([0.641746, 1.75808, 1.46844, 2.27447])

dA_red_high = np.array([[3.30122,3.66031], [5.01213,5.45053], [5.777,6.37477], [6.64533,7.3447]])
dA_red_low = np.array([[-0.757882,0.310056], [-0.321163,1.86612], [-1.88873,1.26929], [-2.38148,2.94299]])
dA_blue_high = np.array([[1.49907,1.96855], [2.03678,2.42186], [1.82241,2.26435], [2.87058,3.32752]])
dA_blue_low = np.array([[0.500091,0.782863], [1.58052,1.90558], [1.28008,1.64257], [2.06155,2.46573]])

dA_red_high.T[0] = abs(A_red_high-dA_red_high.T[0])
dA_red_high.T[1] = abs(dA_red_high.T[1]-A_red_high)

dA_red_low.T[0] = abs(A_red_low-dA_red_low.T[0])
dA_red_low.T[1] = abs(dA_red_low.T[1]-A_red_low)

dA_blue_high.T[0] = abs(A_blue_high-dA_blue_high.T[0])
dA_blue_high.T[1] = abs(dA_blue_high.T[1]-A_blue_high)

dA_blue_low.T[0] = abs(A_blue_low-dA_blue_low.T[0])
dA_blue_low.T[1] = abs(dA_blue_low.T[1]-A_blue_low)



plt.close()
plt.subplot(111)
plt.errorbar(x+0.01,A_red_high, yerr=dA_red_high.T, color='darkred',linestyle='none', label='High Mass Red Galaxies', marker='.')
plt.errorbar(x+0.02,A_red_low, yerr=dA_red_low.T, markeredgecolor='pink', ecolor='pink', markerfacecolor='white',linestyle='none', label='Low Mass Red Galaxies', marker='^')
plt.errorbar(x-0.02,A_blue_high,yerr=dA_blue_high.T, markeredgecolor='midnightblue', ecolor='midnightblue', markerfacecolor='midnightblue',linestyle='none', label='High Mass Blue Galaxies', marker='x')
plt.errorbar(x,A_blue_low,yerr=dA_blue_low.T, markeredgecolor='steelblue', ecolor='steelblue', markerfacecolor='steelblue',linestyle='none', label='Low Mass Blue Galaxies', marker='+')

plt.xlim(0,1.1)
plt.xticks(visible=True)
#plt.yticks([1.5,2,2.5,3.0], fontsize=16)
plt.ylim(-4.4,7.6)
plt.axhline(0,color='k',ls=':')
plt.legend(loc='lower left', fontsize=14)

plt.ylabel(r'$A_{i}$', fontsize=16)

plt.xlabel('Redshift $z$', fontsize=16)
plt.subplots_adjust(hspace=0,wspace=0,bottom=0.14,left=0.14, right=0.98, top=0.98)
plt.savefig('ai_colours_masses_redshift.pdf')
plt.savefig('ai_colours_masses_redshift.png')