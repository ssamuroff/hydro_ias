#!/bin/bash 
#SBATCH --job-name=4Mpc_tatt2_z0_mbii
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=80:00:00
#SBATCH -o /home/ssamurof/mbii/logs/4Mpc_mbii-wgp_wpp_wgg-tatt_z0-analytic_cov.log

cd /home/ssamurof/cosmosis/
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir

export SAMPLER=multinest
export DATA=mbii
export MODEL=tatt2_z0
export COV=fidcov_iteration4
export SAVEFILE=${DATA}_gp1_pp1_wgg1_${MODEL}_${SAMPLER}_${COV}_rmin4_rmax33
export VALUESFILE=../ini/values_${MODEL}_${DATA}.ini
export PARAMFILE=../ini/params_${MODEL}_4Mpc.ini
export PKDIR=/home/ssamurof/hydro_ias/data/theory/${DATA}


mpiexec -n 16 cosmosis $PARAMFILE --mpi -p runtime.sampler=$SAMPLER output.lock=F ia_like.apply_hartlap=F ia_like.datafile=/home/ssamurof/hydro_ias/data/2pt/fits/2pt_${DATA}_${COV}.fits