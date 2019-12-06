cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/tng_tatt_jackknife

export FID=out_tng_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt
export JK=out_tng_gp1_pp1_wgg1_tatt2_z0_multinest_jkcov_rmin6_rmax33.txt


postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/tatt_jackknife.py $FID $JK
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/tatt_jackknife.py $FID $JK 

