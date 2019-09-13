import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')
import numpy as np

   
   
A = np.array([1.707923,2.375591,2.203458, 2.618426])
dA = np.array([0.2395762, 0.1158124, 0.2497444, 0.2719336])
bg = np.array([0.7898388, 0.6689826, 0.8612283, 1.096169])
dbg = np.array([7.268597e-02, 7.001251e-02, 6.023656e-02, 5.494424e-02])
x = np.array([0.062,0.3,0.625,1.00])


plt.subplot(211)
plt.errorbar(x,A,dA,color='midnightblue',linestyle='none', label='MBII', marker='.')

plt.xlim(0,1.1)
plt.xticks(visible=False)
plt.yticks([1.5,2,2.5], fontsize=16)
plt.ylim(1.4,2.65)
plt.legend(fontsize=12, loc='lower right')
plt.ylabel(r'$A_{\rm IA}$', fontsize=16)

plt.subplot(212)
plt.errorbar(x,bg,dbg,color='midnightblue',linestyle='none', marker='.')
plt.xlim(0,1.1)
plt.xticks(visible=True)
plt.yticks([0.6,0.8,1.0,1.2], fontsize=16)
plt.ylabel('$b_g$', fontsize=16)
plt.xlabel('Redshift $z$', fontsize=16)
plt.subplots_adjust(hspace=0,wspace=0,bottom=0.14,left=0.14)
plt.savefig('bias_aia_redshift.pdf')
plt.savefig('bias_aia_redshift.png')

plt.close()
plt.subplot(111)
plt.errorbar(x,A,dA,color='midnightblue',linestyle='none', label='MBII', marker='.')

plt.xlim(0,1.1)
plt.xticks(visible=True)
plt.yticks([1.5,2,2.5,3.0], fontsize=16)
plt.ylim(1.4,3.)
plt.legend(fontsize=12, loc='lower right')
plt.ylabel(r'$A_{\rm IA}$', fontsize=16)

plt.xlabel('Redshift $z$', fontsize=16)
plt.subplots_adjust(hspace=0,wspace=0,bottom=0.14,left=0.14)
plt.savefig('aia_redshift.pdf')
plt.savefig('aia_redshift.png')