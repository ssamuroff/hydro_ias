#!/bin/bash 
#SBATCH --job-name=ss_nla2_z0_tng
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=80:00:00
#SBATCH -o /home/ssamurof/logs/tng_ss-wgp_wpp_wgg-nla_z0-analytic_cov.log

cd /home/ssamurof/cosmosis/
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export SPLIT=ss
export SAMPLER=multinest
export DATA=tng
export MODEL=nla2_z0
export COV=fidcov_iteration2
export SAVEFILE=${DATA}_${SPLIT}_gp1_pp1_wgg1_${MODEL}_${SAMPLER}_${COV}_rmin6_rmax33
export VALUESFILE=../ini/values_${MODEL}_${DATA}.ini
export PARAMFILE=../ini/params_${MODEL}_${DATA}.ini
export PKDIR=/home/ssamurof/hydro_ias/data/theory/${DATA}


mpiexec -n 16 cosmosis $PARAMFILE --mpi -p runtime.sampler=$SAMPLER output.lock=F ia_like.apply_hartlap=F ia_like.datafile=/home/ssamurof/hydro_ias/data/2pt/fits/2pt_${DATA}_${SPLIT}_${COV}.fits