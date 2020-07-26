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
c1 = np.array([2.219711,3.839522,4.099039,3.648769])
dc1 = np.array([8.230667e-01,9.966744e-01,9.958348e-01,1.157521e+00])
c2 = np.array([5.901462e-01,1.057397,1.445419,2.265162])
dc2 = np.array([1.344486,1.428057,1.670525,1.897726])
A1 = np.array([3.314232,4.914953,6.055728,7.316289])
dA1 = np.array([2.473590e-01,3.839672e-01,4.526129e-01,4.583788e-01])

   

plt.close()
plt.subplot(211)
plt.errorbar(x,A1,yerr=dA1,ecolor='darkred', markeredgecolor='darkred',markerfacecolor='white', linestyle='none', label='NLA', marker='o', markersize=3.5)
plt.errorbar(x+0.01,c1,yerr=dc1,color='darkred',linestyle='none', label='TATT $(A_1)$', marker='^')
plt.errorbar(x-0.01,c2,yerr=dc2,color='darkred',linestyle='none', label='TATT $(A_2)$', marker='v')

plt.errorbar([0.28],[4.6],yerr=[0.5],ecolor='pink', markeredgecolor='pink',markerfacecolor='pink', linestyle='none', marker='o', markersize=3.5, alpha=0.4) # S15 main
plt.errorbar([0.33],[3.55],yerr=[0.9],ecolor='pink', markeredgecolor='pink',markerfacecolor='pink', linestyle='none', marker='^', markersize=3.5, alpha=0.4) #J19 G Z1R
plt.errorbar([0.17],[3.63],yerr=[0.79],ecolor='pink', markeredgecolor='pink',markerfacecolor='pink', linestyle='none', marker='>', markersize=3.5, alpha=0.4) #J19 G Z12
plt.errorbar([0.12],[2.50],yerr=[0.77],ecolor='pink', markeredgecolor='pink',markerfacecolor='pink', linestyle='none', marker='*', markersize=6.5, alpha=0.4) #J19 SR
plt.errorbar([0.54],[4.52],yerr=[0.64],ecolor='pink', markeredgecolor='pink',markerfacecolor='pink', linestyle='none', marker='D', markersize=3.5, alpha=0.4) # S15 main


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




c1 = np.array([4.717413e-01,1.223915,1.774102,2.037401])
dc1 = np.array([4.916875e-01,5.584162e-01,4.470668e-01,5.476024e-01] )
c2 = np.array([4.604694e-01,4.607973e-01,5.712782e-01,3.998119e-01])
dc2 = np.array([6.949677e-01,8.101498e-01,7.601825e-01,8.677662e-01])
A1 = np.array([9.154687e-01,1.657148,2.417625,2.622739])
dA1 = np.array([2.282343e-01,2.099128e-01,1.807742e-01,2.267402e-01])


plt.subplot(212)
plt.errorbar(x,A1,dA1,ecolor='midnightblue', markeredgecolor='midnightblue',markerfacecolor='white', linestyle='none', label='NLA', marker='o', markersize=3.5)
plt.errorbar(x+0.01,c1,yerr=dc1,color='midnightblue',linestyle='none', label='TATT $(A_1)$', marker='^')
plt.errorbar(x-0.01,c2,yerr=dc2,color='midnightblue',linestyle='none', label='TATT $(A_2)$', marker='v')



plt.errorbar([0.09],[1.],yerr=[0.8],ecolor='steelblue', markeredgecolor='steelblue',markerfacecolor='steelblue', linestyle='none', marker='o', markersize=3.5, alpha=0.4) #sdss main blue J18 
plt.errorbar([0.335],[0.75],yerr=[0.74],ecolor='steelblue', markeredgecolor='steelblue',markerfacecolor='steelblue', linestyle='none', marker='^', markersize=3.5, alpha=0.4) # gama z2b J18
#plt.errorbar([0.28],[2],yerr=[1.],ecolor='steelblue', markeredgecolor='steelblue',markerfacecolor='steelblue', linestyle='none', marker='>', markersize=3.5, alpha=0.4) # SDSS C5 S15
plt.errorbar([0.6],[0.15],yerr=[1.05],ecolor='steelblue', markeredgecolor='steelblue',markerfacecolor='steelblue', linestyle='none', marker='<', markersize=3.5, alpha=0.4) #wigglez

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