#!/bin/bash 
#SBATCH --job-name=tatt2_z0_mbii_w
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=80:00:00
#SBATCH -o /home/ssamurof/mbii/logs/mbii_w-wgp_wpp_wgg-tatt_z0-analytic_cov.log

cd /home/ssamurof/cosmosis/
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir

export SAMPLER=test
export DATA=mbii
export MODEL=tatt2_z0
export COV=fidcov_iteration4
export SAVEFILE=${DATA}_w_gp1_pp1_wgg1_CHI2_${SAMPLER}_${COV}_rmin6_rmax33
export VALUESFILE=../ini/values_${MODEL}_${DATA}.ini
export PARAMFILE=../ini/params_${MODEL}.ini
export PKDIR=/home/ssamurof/hydro_ias/data/theory/${DATA}

cosmosis $PARAMFILE -p runtime.sampler=$SAMPLER output.lock=F ia_like.apply_hartlap=F ia_like.datafile=/home/ssamurof/hydro_ias/data/2pt/fits/2pt_${DATA}_w_${COV}.fits -v intrinsic_alignment_parameters.a1=1.97404 intrinsic_alignment_parameters.a2=0. intrinsic_alignment_parameters.bias_ta=0.0 bias_parameters.b_0_062=1.24221