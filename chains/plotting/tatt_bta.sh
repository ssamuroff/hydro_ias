cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/tng_tatt_bta

export Z0=out_tng_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt
export Z1=out_tng_gp1_pp1_gg1_tatt2_z0_widebtaprior_multinest_fidcov_iteration4_rmin6_rmax33.txt
export Z2=out_tng_gp1_pp1_wgg1_tatt2_z3_multinest_fidcov_iteration4_rmin6_rmax33.txt
export Z3=out_tng_gp1_pp1_gg1_tatt2_z3_widebtaprior_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/tatt_bta.py $Z0 $Z1 $Z2 $Z3

postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/tatt_bta.py $Z0 $Z1 $Z2 $Z3

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/tatt_bta2.py $Z0 $Z1 $Z2 $Z3