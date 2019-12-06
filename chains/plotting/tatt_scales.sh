cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/tng_tatt_scales

export C0=out_tng_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin2_rmax33.txt
export C1=out_tng_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin4_rmax33.txt
export C2=out_tng_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt

export C3=out_tng_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin0.5_rmax33.txt

export SCRIPT=../plotting/tatt_scales.py

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra $SCRIPT $C3 $C0 $C1 $C2
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra $SCRIPT $C3 $C0 $C1 $C2 

