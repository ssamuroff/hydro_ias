cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/colour_tatt_tng

export R0=out_tng_red_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt
export R1=out_tng_red_gp1_pp1_wgg1_tatt2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export R2=out_tng_red_gp1_pp1_wgg1_tatt2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export R3=out_tng_red_gp1_pp1_wgg1_tatt2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt

export B0=out_tng_blue_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt
export B1=out_tng_blue_gp1_pp1_wgg1_tatt2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export B2=out_tng_blue_gp1_pp1_wgg1_tatt2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export B3=out_tng_blue_gp1_pp1_wgg1_tatt2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt

postprocess -o ${OUT}_z0 --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/colour_tatt_z0.py $R0 $B0
postprocess -o ${OUT}_z0 --no-plots --factor-kde=1.1 -f png --extra ../plotting/colour_tatt_z0.py $R0 $B0


postprocess -o ${OUT}_z1 --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/colour_tatt_z1.py $R1 $B1
postprocess -o ${OUT}_z1 --no-plots --factor-kde=1.1 -f png --extra ../plotting/colour_tatt_z1.py $R1 $B1



postprocess -o ${OUT}_z2 --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/colour_tatt_z2.py $R2 $B2
postprocess -o ${OUT}_z2 --no-plots --factor-kde=1.1 -f png --extra ../plotting/colour_tatt_z2.py $R2 $B2



postprocess -o ${OUT}_z3 --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/colour_tatt_z3.py $R3 $B3
postprocess -o ${OUT}_z3 --no-plots --factor-kde=1.1 -f png --extra ../plotting/colour_tatt_z3.py $R3 $B3







