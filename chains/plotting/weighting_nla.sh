cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/weighting_nla

export MB20=out_mbii_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL0=out_illustris_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt
export MB2=out_mbii_w_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL=out_illustris_w_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/weighting_z0.py $MB20 $MB2 $ILL $ILL0
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/weighting_z0.py $MB20 $MB2 $ILL $ILL0


export MB20=out_mbii_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL0=out_illustris_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration4_rmin6_rmax33.txt
export MB2=out_mbii_w_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL=out_illustris_w_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/weighting_z1.py $MB20 $MB2 $ILL $ILL0
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/weighting_z1.py $MB20 $MB2 $ILL $ILL0

export MB20=out_mbii_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL0=out_illustris_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration4_rmin6_rmax33.txt
export MB2=out_mbii_w_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL=out_illustris_w_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/weighting_z2.py $MB20 $MB2 $ILL $ILL0
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/weighting_z2.py $MB20 $MB2 $ILL $ILL0

export MB20=out_mbii_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL0=out_illustris_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration4_rmin6_rmax33.txt
export MB2=out_mbii_w_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration4_rmin6_rmax33.txt
export ILL=out_illustris_w_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/weighting_z3.py $MB20 $MB2 $ILL $ILL0
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/weighting_z3.py $MB20 $MB2 $ILL $ILL0



# and this one to get the stats
postprocess -o $OUT --no-plots  $MB2 $TNG 