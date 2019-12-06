cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/weighting_tatt

export M0=out_mbii_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt
export M1=out_mbii_w_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt
export I0=out_illustris_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt
export I1=out_illustris_w_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/tatt_weighting.py $M1 $M0 $I1 $I0 
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/tatt_weighting.py $M1 $M0 $I1 $I0 

