#!/bin/bash 
#SBATCH --job-name=R32_tatt_field
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=80:00:00
#SBATCH --mem 1000000
#SBATCH -o /home/ssamurof/mbii/logs/R32_tatt_field.log

cd /home/ssamurof/cosmosis/
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir

export RESOLUTION=32
export SAMPLER=multinest
export PARAMFILE=../ini/params_field.ini
export VALUESFILE=../ini/values_field.ini
export SAVEFILE=tatt_field_R${RESOLUTION}_test

mpiexec -n 5 cosmosis $PARAMFILE --mpi  -p runtime.sampler=$SAMPLER output.lock=F tatt_field.resolution=$RESOLUTION