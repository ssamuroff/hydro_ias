#!/bin/bash 
#SBATCH --job-name=lowM_nla2_z3_tng
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=80:00:00
#SBATCH -o /home/ssamurof/logs/tng_lowMs-wgp_wpp_wgg-nla_z3-analytic_cov.log

cd /home/ssamurof/cosmosis/
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir

export SAMPLER=multinest
export DATA=tng
export MODEL=nla2_z3
export COV=fidcovw_iteration2
export SAVEFILE=${DATA}_lowMs_gp1_pp1_gg1_${MODEL}_${SAMPLER}_${COV}_rmin5.5_rmax33
export VALUESFILE=../ini/values_${MODEL}_${DATA}.ini
export PARAMFILE=../ini/params_${MODEL}_${DATA}.ini
export PKDIR=/home/ssamurof/hydro_ias/data/theory/${DATA}


mpiexec -n 16 cosmosis $PARAMFILE --mpi -p runtime.sampler=$SAMPLER output.lock=F ia_like.apply_hartlap=F ia_like.datafile=/home/ssamurof/hydro_ias/data/2pt/fits/2pt_${DATA}_lowMs_${COV}.fits