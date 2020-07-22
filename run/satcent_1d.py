import numpy as np
import pylab as plt
from pylab import gca
plt.switch_backend('pdf')
plt.style.use('y1a1')
from matplotlib import rcParams
import fitsio as fi
from scipy.interpolate import interp1d 

rcParams['xtick.major.size'] = 3.5
rcParams['xtick.minor.size'] = 1.7
rcParams['ytick.major.size'] = 3.5
rcParams['ytick.minor.size'] = 1.7
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'


print('Snapshot 99')
# load the data for this snapshot
base = 'tng_cs/2pt/txt/wpp_%d_%s_%s.txt'
ccpath = base%(99, "central", "central")
sspath = base%(99, "satellite", "satellite")
cspath = base%(99, "central", "satellite")

x,wcc = np.loadtxt(ccpath).T
x,wss = np.loadtxt(sspath).T
x,wsc = np.loadtxt(cspath).T

scales_mask = np.concatenate(np.argwhere(x>6))

# covariance matrices
cov_cc = np.loadtxt('tng_cc/cov/covmat-analytic.txt')[48:96,48:96]
cov_ss = np.loadtxt('tng_ss/cov/covmat-analytic.txt')[48:96,48:96]

cov_sc = cov_ss

C_cc = cov_cc[:12,:12]
C_ss = cov_ss[:12,:12]
C_sc = cov_sc[:12,:12]

C0_cc = np.linalg.inv(C_cc[8:,8:])
C0_ss = np.linalg.inv(C_ss[8:,8:])
C0_sc = np.linalg.inv(C_sc[8:,8:])


Ecc = np.sqrt(np.diag(C_cc))
Ess = np.sqrt(np.diag(C_ss))
Esc = np.sqrt(np.diag(C_sc))

# load theory and interpolate
yf = np.loadtxt('data/theory/tng/wpp_99_nla_a1.txt')
xf = np.loadtxt('data/theory/tng/r_p.txt')

mask = (xf>0.05) & (xf<200)
T = 10**interp1d(np.log10(xf[mask]), np.log10(yf[mask]) )(np.log10(x))




chi2_cc = []
chi2_ss = []
chi2_sc = []

samples = np.linspace(0,8,600)
for A in samples:
	R_cc = (wcc[scales_mask] - A*A*T[scales_mask])
	R_ss = (wss[scales_mask] - A*A*T[scales_mask])
	R_sc = (wsc[scales_mask] - A*A*T[scales_mask])
	#import pdb ; pdb.set_trace()
	chi2_cc.append(np.dot(R_cc, np.dot(C0_cc,R_cc)))
	chi2_ss.append(np.dot(R_ss, np.dot(C0_ss,R_ss)))
	chi2_sc.append(np.dot(R_sc, np.dot(C0_sc,R_sc)))



Lcc = np.exp(-2*np.array(chi2_cc))
Lcc /= np.trapz(Lcc,samples)
Lss = np.exp(-2*np.array(chi2_ss))
Lss /= np.trapz(Lss,samples)
Lsc = np.exp(-2*np.array(chi2_sc))
Lsc /= np.trapz(Lsc,samples)

plt.plot(samples, Lcc, color='darkred', lw=1.5, ls=':', label=r'$A_{\rm II}^{cc}$')
plt.plot(samples, Lss, color='midnightblue', lw=1.5, ls='--', label=r'$A_{\rm II}^{ss}$')
plt.plot(samples, Lsc, color='darkmagenta', lw=1.5, ls='-', label=r'$A_{\rm II}^{sc}$')

plt.xlabel(r'$A_{\rm II}$', fontsize=18)
plt.legend(loc='upper right', fontsize=18)
plt.yticks(visible=False)
plt.xlim(0,5)
plt.ylim(ymin=0)
plt.title('$z=0.0$', fontsize=18)
plt.savefig('wpp_sat_cent_like_z0.png')
plt.savefig('wpp_sat_cent_like_z0.pdf')
plt.close()


Lcc_99 = np.copy(Lcc)
Lss_99 = np.copy(Lss)
Lsc_99 = np.copy(Lsc)



plt.errorbar(x,x*wcc,yerr=x*Ecc,color='darkred', marker='.', linestyle='none', label='$cc$')
plt.errorbar(x,x*wss,yerr=x*Ess,color='midnightblue', marker='.', linestyle='none', label='$ss$')
plt.errorbar(x,x*wsc,yerr=x*Esc,color='purple', marker='.', linestyle='none', label='$sc$')

plt.ylabel('$w_{++}$')
plt.xlabel('$r_p$')
plt.xscale('log')
plt.savefig('sat_cent_wpp.png')
plt.savefig('sat_cent_wpp.pdf')
plt.close()














# snapshot 78 ; z = 0.3
print('Snapshot 78')


ccpath = base%(78, "central", "central")
sspath = base%(78, "satellite", "satellite")
cspath = base%(78, "central", "satellite")

x,wcc = np.loadtxt(ccpath).T
x,wss = np.loadtxt(sspath).T
x,wsc = np.loadtxt(cspath).T

scales_mask = np.concatenate(np.argwhere(x>6))

# covariance matrices

C_cc = cov_cc[12:24,12:24]
C_ss = cov_ss[12:24,12:24]
C_sc = cov_sc[12:24,12:24]

C0_cc = np.linalg.inv(C_cc[8:,8:])
C0_ss = np.linalg.inv(C_ss[8:,8:])
C0_sc = np.linalg.inv(C_sc[8:,8:])


Ecc = np.sqrt(np.diag(C_cc))
Ess = np.sqrt(np.diag(C_ss))
Esc = np.sqrt(np.diag(C_sc))

# load theory and interpolate
yf = np.loadtxt('data/theory/tng/wpp_78_nla_a1.txt')
xf = np.loadtxt('data/theory/tng/r_p.txt')

mask = (xf>0.05) & (xf<200)
T = 10**interp1d(np.log10(xf[mask]), np.log10(yf[mask]) )(np.log10(x))




chi2_cc = []
chi2_ss = []
chi2_sc = []

samples = np.linspace(0,8,600)
for A in samples:
	R_cc = (wcc[scales_mask] - A*A*T[scales_mask])
	R_ss = (wss[scales_mask] - A*A*T[scales_mask])
	R_sc = (wsc[scales_mask] - A*A*T[scales_mask])
	#import pdb ; pdb.set_trace()
	chi2_cc.append(np.dot(R_cc, np.dot(C0_cc,R_cc)))
	chi2_ss.append(np.dot(R_ss, np.dot(C0_ss,R_ss)))
	chi2_sc.append(np.dot(R_sc, np.dot(C0_sc,R_sc)))



Lcc = np.exp(-2*np.array(chi2_cc))
Lcc /= np.trapz(Lcc,samples)
Lss = np.exp(-2*np.array(chi2_ss))
Lss /= np.trapz(Lss,samples)
Lsc = np.exp(-2*np.array(chi2_sc))
Lsc /= np.trapz(Lsc,samples)

plt.plot(samples, Lcc, color='darkred', lw=1.5, ls=':', label=r'$A_{\rm II}^{cc}$')
plt.plot(samples, Lss, color='midnightblue', lw=1.5, ls='--', label=r'$A_{\rm II}^{ss}$')
plt.plot(samples, Lsc, color='darkmagenta', lw=1.5, ls='-', label=r'$A_{\rm II}^{sc}$')

plt.xlabel(r'$A_{\rm II}$', fontsize=18)
plt.legend(loc='upper right', fontsize=18)
plt.yticks(visible=False)
plt.xlim(0,5)
plt.ylim(ymin=0)
plt.title('$z=0.3$', fontsize=18)
plt.savefig('wpp_sat_cent_like_z1.png')
plt.savefig('wpp_sat_cent_like_z1.pdf')
plt.close()


Lcc_78 = np.copy(Lcc)
Lss_78 = np.copy(Lss)
Lsc_78 = np.copy(Lsc)




# snapshot 62 ; z = 0.625
print('Snapshot 62')

ccpath = base%(62, "central", "central")
sspath = base%(62, "satellite", "satellite")
cspath = base%(62, "central", "satellite")

x,wcc = np.loadtxt(ccpath).T
x,wss = np.loadtxt(sspath).T
x,wsc = np.loadtxt(cspath).T

scales_mask = np.concatenate(np.argwhere(x>6))

# covariance matrices

C_cc = cov_cc[24:36,24:36]
C_ss = cov_ss[24:36,24:36]
C_sc = cov_sc[24:36,24:36]

C0_cc = np.linalg.inv(C_cc[8:,8:])
C0_ss = np.linalg.inv(C_ss[8:,8:])
C0_sc = np.linalg.inv(C_sc[8:,8:])


Ecc = np.sqrt(np.diag(C_cc))
Ess = np.sqrt(np.diag(C_ss))
Esc = np.sqrt(np.diag(C_sc))

# load theory and interpolate
yf = np.loadtxt('data/theory/tng/wpp_62_nla_a1.txt')
xf = np.loadtxt('data/theory/tng/r_p.txt')

mask = (xf>0.05) & (xf<200)
T = 10**interp1d(np.log10(xf[mask]), np.log10(yf[mask]) )(np.log10(x))


chi2_cc = []
chi2_ss = []
chi2_sc = []

samples = np.linspace(0,8,600)
for A in samples:
	R_cc = (wcc[scales_mask] - A*A*T[scales_mask])
	R_ss = (wss[scales_mask] - A*A*T[scales_mask])
	R_sc = (wsc[scales_mask] - A*A*T[scales_mask])
	#import pdb ; pdb.set_trace()
	chi2_cc.append(np.dot(R_cc, np.dot(C0_cc,R_cc)))
	chi2_ss.append(np.dot(R_ss, np.dot(C0_ss,R_ss)))
	chi2_sc.append(np.dot(R_sc, np.dot(C0_sc,R_sc)))



Lcc = np.exp(-2*np.array(chi2_cc))
Lcc /= np.trapz(Lcc,samples)
Lss = np.exp(-2*np.array(chi2_ss))
Lss /= np.trapz(Lss,samples)
Lsc = np.exp(-2*np.array(chi2_sc))
Lsc /= np.trapz(Lsc,samples)

plt.plot(samples, Lcc, color='darkred', lw=1.5, ls=':', label=r'$A_{\rm II}^{cc}$')
plt.plot(samples, Lss, color='midnightblue', lw=1.5, ls='--', label=r'$A_{\rm II}^{ss}$')
plt.plot(samples, Lsc, color='darkmagenta', lw=1.5, ls='-', label=r'$A_{\rm II}^{sc}$')

plt.xlabel(r'$A_{\rm II}$', fontsize=18)
plt.legend(loc='upper right', fontsize=18)
plt.yticks(visible=False)
plt.xlim(0,5)
plt.ylim(ymin=0)
plt.title('$z=0.6$', fontsize=18)
plt.savefig('wpp_sat_cent_like_z2.png')
plt.savefig('wpp_sat_cent_like_z2.pdf')
plt.close()

Lcc_62 = np.copy(Lcc)
Lss_62 = np.copy(Lss)
Lsc_62 = np.copy(Lsc)


# snapshot 50 ; z = 1.000
print('Snapshot 50')

ccpath = base%(50, "central", "central")
sspath = base%(50, "satellite", "satellite")
cspath = base%(50, "central", "satellite")

x,wcc = np.loadtxt(ccpath).T
x,wss = np.loadtxt(sspath).T
x,wsc = np.loadtxt(cspath).T

scales_mask = np.concatenate(np.argwhere(x>6))

# covariance matrices

C_cc = cov_cc[36:48,36:48]
C_ss = cov_ss[36:48,36:48]
C_sc = cov_sc[36:48,36:48]

C0_cc = np.linalg.inv(C_cc[8:,8:])
C0_ss = np.linalg.inv(C_ss[8:,8:])
C0_sc = np.linalg.inv(C_sc[8:,8:])


Ecc = np.sqrt(np.diag(C_cc))
Ess = np.sqrt(np.diag(C_ss))
Esc = np.sqrt(np.diag(C_sc))

# load theory and interpolate
yf = np.loadtxt('data/theory/tng/wpp_50_nla_a1.txt')
xf = np.loadtxt('data/theory/tng/r_p.txt')

mask = (xf>0.05) & (xf<200)
T = 10**interp1d(np.log10(xf[mask]), np.log10(yf[mask]) )(np.log10(x))


chi2_cc = []
chi2_ss = []
chi2_sc = []

samples = np.linspace(0,8,600)
for A in samples:
	R_cc = (wcc[scales_mask] - A*A*T[scales_mask])
	R_ss = (wss[scales_mask] - A*A*T[scales_mask])
	R_sc = (wsc[scales_mask] - A*A*T[scales_mask])
	#import pdb ; pdb.set_trace()
	chi2_cc.append(np.dot(R_cc, np.dot(C0_cc,R_cc)))
	chi2_ss.append(np.dot(R_ss, np.dot(C0_ss,R_ss)))
	chi2_sc.append(np.dot(R_sc, np.dot(C0_sc,R_sc)))



Lcc = np.exp(-2*np.array(chi2_cc))
Lcc /= np.trapz(Lcc,samples)
Lss = np.exp(-2*np.array(chi2_ss))
Lss /= np.trapz(Lss,samples)
Lsc = np.exp(-2*np.array(chi2_sc))
Lsc /= np.trapz(Lsc,samples)

plt.plot(samples, Lcc, color='darkred', lw=1.5, ls=':', label=r'$A_{\rm II}^{cc}$')
plt.plot(samples, Lss, color='midnightblue', lw=1.5, ls='--', label=r'$A_{\rm II}^{ss}$')
plt.plot(samples, Lsc, color='darkmagenta', lw=1.5, ls='-', label=r'$A_{\rm II}^{sc}$')

plt.xlabel(r'$A_{\rm II}$', fontsize=18)
plt.legend(loc='upper right', fontsize=18)
plt.yticks(visible=False)
plt.xlim(0,5)
plt.ylim(ymin=0)
plt.title('$z=1.0$', fontsize=18)
plt.savefig('wpp_sat_cent_like_z3.png')
plt.savefig('wpp_sat_cent_like_z3.pdf')
plt.close()

Lcc_50 = np.copy(Lcc)
Lss_50 = np.copy(Lss)
Lsc_50 = np.copy(Lsc)




plt.subplot(411)
plt.plot(samples, Lcc_99, color='darkred', lw=1.5, ls=':', label=r'$A_{\rm II}^{cc}$')
plt.plot(samples, Lss_99, color='midnightblue', lw=1.5, ls='--')
plt.plot(samples, Lsc_99, color='darkmagenta', lw=1.5, ls='-')

#import pdb ; pdb.set_trace()
x0,L0 = np.loadtxt('plotting_scripts/datapoints/tng_like_sat.txt').T
#L0[np.isinf(L0)]=0
plt.fill_between(x0,L0,color='darkmagenta',alpha=0.2)
#path = '/Users/hattifattener/coma/hydro_ias/chains/rundir/out_tng_ss_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt'


plt.yticks(visible=False)
plt.xticks(visible=False)
plt.xlim(0,5)
plt.ylim(ymin=0)
plt.annotate('$z=0.0$', fontsize=16, xy=(3.8,1.74) )


plt.subplot(412)
plt.plot(samples, Lcc_78, color='darkred', lw=1.5, ls=':')
plt.plot(samples, Lss_78, color='midnightblue', lw=1.5, ls='--', label=r'$A_{\rm II}^{ss}$')
plt.plot(samples, Lsc_78, color='darkmagenta', lw=1.5, ls='-')


plt.yticks(visible=False)
plt.xticks(visible=False)
plt.xlim(0,5)
plt.ylim(ymin=0)
plt.annotate('$z=0.3$', fontsize=16, xy=(3.8,1.75) )
plt.legend(loc='upper left', fontsize=16)


plt.subplot(413)
plt.plot(samples, Lcc_62, color='darkred', lw=1.5, ls=':')
plt.plot(samples, Lss_62, color='midnightblue', lw=1.5, ls='--')
plt.plot(samples, Lsc_62, color='darkmagenta', lw=1.5, ls='-', label=r'$A_{\rm II}^{sc}$')


plt.yticks(visible=False)
plt.xticks(visible=False)
plt.xlim(0,5)
plt.ylim(ymin=0)
plt.annotate('$z=0.6$', fontsize=16, xy=(3.8,1.75) )
plt.legend(loc='upper left', fontsize=16)

plt.subplot(414)
plt.plot(samples, Lcc_50, color='darkred', lw=1.5, ls=':', label=r'$A_{\rm II}^{cc}$')
plt.plot(samples, Lss_50, color='midnightblue', lw=1.5, ls='--')
plt.plot(samples, Lsc_50, color='darkmagenta', lw=1.5, ls='-')


plt.yticks(visible=False)
plt.xticks(visible=True, fontsize=16)
plt.xlim(0,5)
plt.ylim(ymin=0)
plt.annotate('$z=1.0$', fontsize=16, xy=(3.8,1.77) )
plt.legend(loc='upper left', fontsize=16)

plt.xlabel(r'$A_{\rm II}$', fontsize=16)

plt.subplots_adjust(left=0.12,bottom=0.14, top=0.98, right=0.65, hspace=0, wspace=0)
plt.savefig('wpp_sat_cent_like_panels.png')
plt.savefig('wpp_sat_cent_like_panels.pdf')
plt.close()


