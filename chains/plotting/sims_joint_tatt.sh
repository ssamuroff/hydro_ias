cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/joint_sims_tatt

export MB2=out_mbii_w_gp1_pp1_wgg1_tatt2_z0_widebtaprior_multinest_fidcov_iteration4_rmin6_rmax33.txt
export TNG=out_tng_gp1_pp1_gg1_tatt2_z0_widebtaprior_multinest_fidcov_iteration4_rmin6_rmax33.txt
export COMB=out_joint_gp1_pp1_gg1_tatt2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=0.9 -f pdf --extra ../plotting/sims_joint_tatt_z0.py $MB2 $TNG $COMB
postprocess -o $OUT --no-plots --factor-kde=0.9 -f png --extra ../plotting/sims_joint_tatt_z0.py $MB2 $TNG $COMB



# and this one to get the stats
postprocess -o $OUT --no-plots  $MB2 $TNG $COMB