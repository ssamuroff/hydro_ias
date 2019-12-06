cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/all_sims_nla

export MB2=out_mbii_w_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt
export TNG=out_tng_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL=out_illustris_w_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/sims_z0.py $TNG $MB2 $ILL
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/sims_z0.py $TNG $MB2 $ILL


export MB2=out_mbii_w_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration4_rmin6_rmax33.txt
export TNG=out_tng_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL=out_illustris_w_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/sims_z1.py $TNG $MB2 $ILL
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/sims_z1.py $TNG $MB2 $ILL

export MB2=out_mbii_w_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration4_rmin6_rmax33.txt
export TNG=out_tng_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL=out_illustris_w_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/sims_z2.py $TNG $MB2 $ILL
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/sims_z2.py $TNG $MB2 $ILL

export MB2=out_mbii_w_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration4_rmin6_rmax33.txt
export TNG=out_tng_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL=out_illustris_w_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/sims_z3.py $TNG $MB2 $ILL
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/sims_z3.py $TNG $MB2 $ILL



# and this one to get the stats
postprocess -o $OUT --no-plots  $MB2 $TNG 