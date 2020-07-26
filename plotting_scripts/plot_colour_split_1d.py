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



x = np.array([0.02,0.3,0.625,1.0])
c1 = np.array([3.06005, 4.09835, 4.8119, 5.2892])
dc1 = np.array([[2.05846, 2.80877], [2.63536, 3.7293], [3.25863, 4.3138], [3.12228, 4.45665]])
c2 = np.array([-0.0579288, 0.884252, 0.956029, 1.4865])
dc2 = np.array([[-0.952383, 0.414895], [0.0414099, 1.86282], [0.257184, 2.27668], [1.08588, 3.2711] ])

dc1.T[0] = abs(c1-dc1.T[0])
dc1.T[1] = abs(dc1.T[1]-c1)

dc2.T[0] = abs(c2-dc2.T[0])
dc2.T[1] = abs(dc2.T[1]-c2)

A1 = np.array([2.987013,  4.760670, 5.625894, 6.381556])
dA1 = np.array([3.670063e-01, 4.849381e-01, 6.194130e-01, 8.154452e-01])



plt.close()
plt.subplot(211)
plt.errorbar(x,A1,yerr=dA1,ecolor='darkred', markeredgecolor='darkred',markerfacecolor='white', linestyle='none', label='NLA', marker='o', markersize=3.5)
plt.errorbar(x+0.01,c1,yerr=dc1.T,color='darkred',linestyle='none', label='TATT $(A_1)$', marker='^')
plt.errorbar(x-0.01,c2,yerr=dc2.T,color='darkred',linestyle='none', label='TATT $(A_2)$', marker='v')

plt.xlim(0,1.1)
plt.xticks(visible=False)
plt.yticks([-4, -2, 0, 2, 4, 6], fontsize=16)
plt.ylim(-4,7.6)
plt.axhline(0,color='k',ls=':')
#l1 = plt.legend([t1,m1,i1], ["TNG $(A_1)$", "MBII $(A_1)$", "Illustris $(A_1)$"], fontsize=12, loc='upper left')
#l2 = plt.legend([t2,m2,i2], ["TNG $(A_2)$", "MBII $(A_2)$", "Illustris $(A_2)$"], fontsize=12, loc='upper center')
#gca().add_artist(l1)

plt.ylabel(r'$A_{i}$', fontsize=16)
plt.annotate('Red Galaxies', fontsize=16, xy=(0.68,-3))




c1 = np.array([0.791682, 1.32729, 1.81782, 2.43755])
dc1 = np.array([(0.183168, 0.582813), (0.656493, 1.15922), (1.30871, 1.7629), (1.60544, 2.19339)] )
c2 = np.array([0.0924801, -0.146981, 0.361471, 0.0112941])
dc2 = np.array([(-0.189517, 0.642678), (-0.250263, 0.734603), (0.14963, 0.99073), (-0.362448, 0.80374)])
A1 = np.array([9.392091e-01,  1.745490e+00, 2.389427e+00, 2.577800])
dA1 = np.array([2.822485e-01, 2.698413e-01, 2.414924e-01, 3.198142e-01])

  


dc1.T[0] = abs(c1-dc1.T[0])
dc1.T[1] = abs(dc1.T[1]-c1)

dc2.T[0] = abs(c2-dc2.T[0])
dc2.T[1] = abs(dc2.T[1]-c2)


plt.subplot(212)
plt.errorbar(x,A1,dA1,ecolor='midnightblue', markeredgecolor='midnightblue',markerfacecolor='white', linestyle='none', label='NLA', marker='o', markersize=3.5)
plt.errorbar(x+0.01,c1,yerr=dc1.T,color='midnightblue',linestyle='none', label='TATT $(A_1)$', marker='^')
plt.errorbar(x-0.01,c2,yerr=dc2.T,color='midnightblue',linestyle='none', label='TATT $(A_2)$', marker='v')



plt.errorbar([0.09],[1.],yerr=[0.8],ecolor='steelblue', markeredgecolor='steelblue',markerfacecolor='steelblue', linestyle='none', marker='o', markersize=3.5, alpha=0.4) #sdss main blue J18 
plt.errorbar([0.335],[0.75],yerr=[0.74],ecolor='steelblue', markeredgecolor='steelblue',markerfacecolor='steelblue', linestyle='none', marker='^', markersize=3.5, alpha=0.4) # gama z2b J18
plt.errorbar([0.28],[2],yerr=[1.],ecolor='steelblue', markeredgecolor='steelblue',markerfacecolor='steelblue', linestyle='none', marker='>', markersize=3.5, alpha=0.4) # SDSS C5 S15
plt.errorbar([0.51],[0.24],yerr=[1.27],ecolor='steelblue', markeredgecolor='steelblue',markerfacecolor='steelblue', linestyle='none', marker='<', markersize=3.5, alpha=0.4) #wigglez

plt.legend(fontsize=12, loc='upper left')

plt.xlim(0,1.1)
plt.xticks(visible=True)
plt.yticks([-4, -2, 0, 2, 4, 6], fontsize=16)
plt.ylim(-4,7.6)
plt.axhline(0,color='k',ls=':')

plt.annotate('Blue Galaxies', fontsize=16, xy=(0.68,-3))
plt.ylabel(r'$A_{i}$', fontsize=16)
plt.xlabel('Redshift $z$', fontsize=16)
plt.subplots_adjust(hspace=0.05,wspace=0,bottom=0.14,left=0.14, right=0.7, top=0.98)
plt.savefig('ai_colour_split_redshift.pdf')
plt.savefig('ai_colour_split_redshift.png')