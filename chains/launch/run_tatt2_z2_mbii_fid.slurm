#!/bin/bash 
#SBATCH --job-name=tatt2_z2_mbii_widebtaprior
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=80:00:00
#SBATCH -o /home/ssamurof/mbii/logs/mbii-wgp_wpp_wgg-tatt_z2-analytic_cov_widebtaprior.log

cd /home/ssamurof/cosmosis/
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir

export SAMPLER=multinest
export DATA=mbii
export MODEL=tatt2_z2
export COV=fidcov_iteration4
export SAVEFILE=${DATA}_w_gp1_pp1_wgg1_${MODEL}_widebtaprior_${SAMPLER}_${COV}_rmin6_rmax33
export VALUESFILE=../ini/values_${MODEL}_${DATA}_widebtaprior.ini
export PARAMFILE=../ini/params_${MODEL}.ini
export PKDIR=/home/ssamurof/hydro_ias/data/theory/${DATA}


mpiexec -n 16 cosmosis $PARAMFILE --mpi -p runtime.sampler=$SAMPLER output.lock=F ia_like.apply_hartlap=F ia_like.datafile=/home/ssamurof/hydro_ias/data/2pt/fits/2pt_${DATA}_w_${COV}.fits