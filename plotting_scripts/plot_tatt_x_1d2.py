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
c1_mbii = np.array([3.265333,2.745985e+00,3.680383,4.598858])
dc1_mbii = np.array([9.081121e-01,1.092018,1.052405,1.061243])
c2_mbii = np.array([-2.612264,3.474238e-01,1.212064,1.068435])
dc2_mbii = np.array([8.306540e-01,1.576615,2.083341,2.608056])

anla_mbii = np.array([1.665360,2.712455,2.922029,3.382812])
danla_mbii = np.array([3.105485e-01,3.491394e-01,3.583489e-01,2.357950e-01])
  
   
x0 = np.array([0.00,0.3,0.625,1.0])
c1_tng = np.array([1.291744,1.760114,1.644592,2.264720])
dc1_tng = np.array([4.901287e-01,5.110007e-01,5.067119e-01,5.315949e-01])
c2_tng = np.array([3.173342e-01,6.530024e-01,6.535486e-01,6.262007e-01])
dc2_tng = np.array([6.524938e-01,7.598870e-01,7.851422e-01,8.013186e-01])

anla_tng = np.array([1.713431,2.352913,2.444378,3.314982])
danla_tng = np.array([0.1654441,0.1809880,0.1881600,0.1904790])


c1_ill = np.array([2.325144e-01,2.305686,-4.292113e-02,5.853311e-01])
dc1_ill = np.array([8.699082e-01,1.386613e+00,1.234218e+00,1.166175e+00])
c2_ill = np.array([8.062827e-01, -3.871559e-01,1.352749, 9.048338e-01])
dc2_ill = np.array([9.318218e-01, 1.007575e+00, 1.023365e+00, 9.893312e-01])

anla_ill = np.array([1.494811,0.7538351,2.228394,2.383584])
danla_ill = np.array([3.118451e-01,5.386573e-01,3.610159e-01,4.880047e-01])
   


s=8.0

plt.close()
plt.subplot(111)
t1 = plt.errorbar(x+0.01,c1_tng,dc1_tng,color='darkmagenta',linestyle='none', label='TNG $(A_1)$', marker='^', markersize=s)
t2 = plt.errorbar(x-0.01,c2_tng,dc2_tng,color='darkmagenta',linestyle='none', label='TNG $(A_2)$', marker='v', markersize=s)
t3 = plt.errorbar(x-0.005,anla_tng,danla_tng,color='darkmagenta',linestyle='none', label=r'TNG $(A_1, \mathrm{NLA})$', marker='*', markersize=s)
m1 = plt.errorbar(x+0.02,c1_mbii,dc1_mbii,color='midnightblue',linestyle='none', label='MBII $(A_1)$', marker='^', markersize=s)
m2 = plt.errorbar(x-0.02,c2_mbii,dc2_mbii,color='midnightblue',linestyle='none', label='MBII $(A_2)$', marker='v', markersize=s)
m3 = plt.errorbar(x+0.005,anla_mbii,danla_mbii,color='midnightblue',linestyle='none', label=r'MBII $(A_1, \mathrm{NLA})$', marker='*', markersize=s)
i1 = plt.errorbar(x+0.03,c1_ill,dc1_ill,markeredgecolor='forestgreen', ecolor='forestgreen', markerfacecolor='white',linestyle='none', label='Illustris $(A_1)$', marker='^', markersize=s)
i2 = plt.errorbar(x-0.03,c2_ill,dc2_ill,markeredgecolor='forestgreen', ecolor='forestgreen', markerfacecolor='white',linestyle='none', label='Illustris $(A_2)$', marker='v', markersize=s)
i3 = plt.errorbar(x-0.035,anla_ill,danla_ill,markeredgecolor='forestgreen', ecolor='forestgreen', markerfacecolor='white',linestyle='none', label=r'Illustris $(A_1, \mathrm{NLA})$', marker='*', markersize=s)

plt.xlim(0,1.1)
plt.xticks(visible=True, fontsize=16)
plt.yticks(visible=True, fontsize=16)
#plt.yticks([1.5,2,2.5,3.0], fontsize=16)
plt.ylim(-4,7.2)
plt.axhline(0,color='k',ls=':')
l1 = plt.legend([t1,m1,i1], ["TNG $(A_1)$", "MBII $(A_1)$", "Illustris $(A_1)$"], fontsize=12, loc='upper left')
l2 = plt.legend([t2,m2,i2], ["TNG $(A_2)$", "MBII $(A_2)$", "Illustris $(A_2)$"], fontsize=12, loc='upper center')
l3 = plt.legend([t3,m3,i3], [r"TNG $(A_1, \mathrm{NLA})$", r"MBII $(A_1, \mathrm{NLA})$", r"Illustris $(A_1, \mathrm{NLA})$"], fontsize=12, loc='upper right')
gca().add_artist(l1)
gca().add_artist(l2)

plt.ylabel(r'$A_{i}$', fontsize=20)

plt.xlabel('Redshift', fontsize=20)
plt.subplots_adjust(hspace=0,wspace=0,bottom=0.14,left=0.14, right=0.98, top=0.98)
plt.savefig('ai_redshift_v2.pdf')
plt.savefig('ai_redshift_v2.png')
