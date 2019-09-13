cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/mbii_tatt

export Z0=out_mbii_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_rmin6_rmax33.txt
export Z1=out_mbii_gp1_pp1_wgg1_tatt2_z1_multinest_fidcov_rmin6_rmax33.txt
export Z2=out_mbii_gp1_pp1_wgg1_tatt2_z2_multinest_fidcov_rmin6_rmax33.txt
export Z3=out_mbii_gp1_pp1_wgg1_tatt2_z3_multinest_fidcov_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/tatt_redshift.py $Z0 $Z1 $Z2 $Z3 

