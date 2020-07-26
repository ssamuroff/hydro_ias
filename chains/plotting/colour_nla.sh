cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/colour_nla_tng

export R0=out_tng_red_gp1_pp1_gg1_nla2_z0_multinest_fidcovw_iteration2_rmin5.5_rmax33.txt
export R1=out_tng_red_gp1_pp1_gg1_nla2_z1_multinest_fidcovw_iteration2_rmin5.5_rmax33.txt
export R2=out_tng_red_gp1_pp1_gg1_nla2_z2_multinest_fidcovw_iteration2_rmin5.5_rmax33.txt
export R3=out_tng_red_gp1_pp1_gg1_nla2_z3_multinest_fidcovw_iteration2_rmin5.5_rmax33.txt

export B0=out_tng_blue_gp1_pp1_gg1_nla2_z0_multinest_fidcovw_iteration2_rmin5.5_rmax33.txt
export B1=out_tng_blue_gp1_pp1_gg1_nla2_z1_multinest_fidcovw_iteration2_rmin5.5_rmax33.txt
export B2=out_tng_blue_gp1_pp1_gg1_nla2_z2_multinest_fidcovw_iteration2_rmin5.5_rmax33.txt
export B3=out_tng_blue_gp1_pp1_gg1_nla2_z3_multinest_fidcovw_iteration2_rmin5.5_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/colour_nla.py $R0 $R1 $R2 $R3 $B0 $B1 $B2 $B3 
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/colour_nla.py $R0 $R1 $R2 $R3 $B0 $B1 $B2 $B3 






