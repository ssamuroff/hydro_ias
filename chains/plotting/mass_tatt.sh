cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir

export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/mass_tatt_tng
export OUT1=/home/ssamurof/hydro_ias/chains/postprocessed/low_mass_tatt_tng
export OUT2=/home/ssamurof/hydro_ias/chains/postprocessed/high_mass_tatt_tng


export H0=out_tng_highMs_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt
export H1=out_tng_highMs_gp1_pp1_wgg1_tatt2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export H2=out_tng_highMs_gp1_pp1_wgg1_tatt2_z2_multinest_fidcov_iteration2_rmin6_rmax33.txt
export H3=out_tng_highMs_gp1_pp1_wgg1_tatt2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt

export L0=out_tng_lowMs_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt
export L1=out_tng_lowMs_gp1_pp1_wgg1_tatt2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export L2=out_tng_lowMs_gp1_pp1_wgg1_tatt2_z2_multinest_fidcov_iteration2_rmin6_rmax33.txt
export L3=out_tng_lowMs_gp1_pp1_wgg1_tatt2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt

postprocess -o $OUT1 --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/low_mass_tatt.py $L0 $L1 $L2 $L3 
postprocess -o $OUT2 --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/high_mass_tatt.py $H0 $H1 $H2 $H3 

postprocess -o $OUT1 --no-plots --factor-kde=1.1 -f png --extra ../plotting/low_mass_tatt.py $L0 $L1 $L2 $L3 
postprocess -o $OUT2 --no-plots --factor-kde=1.1 -f png --extra ../plotting/high_mass_tatt.py $H0 $H1 $H2 $H3 





postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/mass_tatt.py $L0 $L1 $L2 $L3 $H0 $H1 $H2 $H3 
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/mass_tatt.py $L0 $L1 $L2 $L3 $H0 $H1 $H2 $H3 