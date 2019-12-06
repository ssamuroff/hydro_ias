cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/mass_nla_tng

export H0=out_tng_highMs_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt
export H1=out_tng_highMs_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export H2=out_tng_highMs_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration2_rmin6_rmax33.txt
export H3=out_tng_highMs_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt

export L0=out_tng_lowMs_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt
export L1=out_tng_lowMs_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export L2=out_tng_lowMs_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration2_rmin6_rmax33.txt
export L3=out_tng_lowMs_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/mass_nla.py $L0 $L1 $L2 $L3 $H0 $H1 $H2 $H3 
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/mass_nla.py $L0 $L1 $L2 $L3 $H0 $H1 $H2 $H3 






