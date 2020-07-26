cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/all_sims_tatt

export MB2=out_mbii_w_gp1_pp1_wgg1_tatt2_z0_widebtaprior_multinest_fidcovw_iteration4_rmin6_rmax33.txt
export TNG=out_tng_gp1_pp1_gg1_tatt2_z0_widebtaprior_multinest_fidcovw_iteration4_rmin5.5_rmax33.txt
export ILL=out_illustris_w_gp1_pp1_wgg1_tatt2_z0_widebtaprior_multinest_fidcovw_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/sims_tatt_z0.py $TNG $MB2 $ILL
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/sims_tatt_z0.py $TNG $MB2 $ILL



# and this one to get the stats
postprocess -o $OUT --no-plots  $ILL $MB2 $TNG 