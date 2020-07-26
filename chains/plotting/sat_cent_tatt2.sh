cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/sat_cent_tatt_tng_2

export C=out_tng_ac_gp1_pp1_gg1_tatt2_z0_widebtaprior_multinest_fidcovw_iteration2_rmin5.5_rmax33.txt
export S=out_tng_as_gp1_pp1_gg1_tatt2_z0_widebtaprior_multinest_fidcovw_iteration2_rmin5.5_rmax33.txt

postprocess -o ${OUT}_z0 --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/sat_cent_tatt_z0_2.py $S $C
postprocess -o ${OUT}_z0 --no-plots --factor-kde=1.1 -f png --extra ../plotting/sat_cent_tatt_z0_2.py $S $C

