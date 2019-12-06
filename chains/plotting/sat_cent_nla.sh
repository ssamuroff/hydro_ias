cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/sat_cent_nla_tng

export CC=out_tng_cc_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt
export SS=out_tng_ss_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt
export CS=out_tng_cs_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt
export SC=out_tng_sc_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/sat_cent_nla_z0.py $CC $SS $CS $SC
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/sat_cent_nla_z0.py $CC $SS $CS $SC


export CC=out_tng_cc_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export SS=out_tng_ss_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export CS=out_tng_cs_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export SC=out_tng_sc_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/sat_cent_nla_z1.py $CC $SS $CS $SC
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/sat_cent_nla_z1.py $CC $SS $CS $SC


export CC=out_tng_cc_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration2_rmin6_rmax33.txt
export SS=out_tng_ss_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration2_rmin6_rmax33.txt
export CS=out_tng_cs_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration2_rmin6_rmax33.txt
export SC=out_tng_sc_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration2_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/sat_cent_nla_z2.py $CC $SS $CS $SC
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/sat_cent_nla_z2.py $CC $SS $CS $SC

export CC=out_tng_cc_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt
export SS=out_tng_ss_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt
export CS=out_tng_cs_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt
export SC=out_tng_sc_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/sat_cent_nla_z3.py $CC $SS $CS $SC
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/sat_cent_nla_z3.py $CC $SS $CS $SC





