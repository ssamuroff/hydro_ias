#!/bin/bash 
#SBATCH --job-name=R64_tatt_field
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=long
#SBATCH --time=80:00:00
#SBATCH --mem 1000000
#SBATCH -o /home/ssamurof/mbii/logs/R64_tatt_field.log

cd /home/ssamurof/cosmosis/
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir

export RESOLUTION=64
export SAMPLER=multinest
export PARAMFILE=../ini/params_field.ini
export VALUESFILE=../ini/values_field.ini
export SAVEFILE=tatt_field_R${RESOLUTION}_avg2

cosmosis $PARAMFILE -p runtime.sampler=$SAMPLER output.lock=F tatt_field.resolution=$RESOLUTION