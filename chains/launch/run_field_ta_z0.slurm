#!/bin/bash 
#SBATCH --job-name=nla_field
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=80:00:00
#SBATCH --mem 1000000
#SBATCH -o /home/ssamurof/mbii/logs/nla_field.log

cd /home/ssamurof/cosmosis/
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir

export SAMPLER=grid
export PARAMFILE=../ini/params_field.ini
export VALUESFILE=../ini/values_field.ini
export SAVEFILE=nla_field

mpiexec -n 16 cosmosis $PARAMFILE --mpi -p runtime.sampler=$SAMPLER output.lock=F -v intrinsic_alignment_parameters.c2=0. intrinsic_alignment_parameters.bias_ta=0.0