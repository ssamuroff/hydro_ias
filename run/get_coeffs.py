import numpy as np
import scipy.interpolate as spi
import sys

def compute_c1_baseline():
    C1_M_sun = 5e-14  
    M_sun = 1.9891e30  
    Mpc_in_m = 3.0857e22  
    C1_SI = C1_M_sun / M_sun * (Mpc_in_m)**3  
    
    G = 6.67384e-11  
    H = 100  
    H_SI = H * 1000.0 / Mpc_in_m 
    rho_crit_0 = 3 * H_SI**2 / (8 * np.pi * G)
    f = C1_SI * rho_crit_0
    return f

C1_RHOCRIT = compute_c1_baseline()

Dz = np.loadtxt('chains/rundir/example_output/growth_parameters/d_z.txt')
z = np.loadtxt('chains/rundir/example_output/growth_parameters/z.txt')

z0 = sys.argv[1]
z0 = float(z0)

print('Computing coefficients for redshift z=%f'%z0)

D = spi.interp1d(z,Dz)(z0)

Omega_m = 0.3089 

K = - C1_RHOCRIT * Omega_m / D
K2 = 5 * C1_RHOCRIT * Omega_m / D / D


print('C1 = A1 * %f'%K)
print('C2 = A2 * %f'%K2)
