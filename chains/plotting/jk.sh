cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/mbii_jk_test

export JK0=out_mbii_gp1_pp1_wgg1_nla2_z0_multinest_jkcov_rmin6_rmax33.txt
export FID0=out_mbii_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/jk_z0.py $FID0 $JK0 



export JK1=out_mbii_gp1_pp1_wgg1_nla2_z1_multinest_jkcov_rmin6_rmax33.txt
export FID1=out_mbii_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/jk_z1.py $FID1 $JK1 


export JK2=out_mbii_gp1_pp1_wgg1_nla2_z2_multinest_jkcov_rmin6_rmax33.txt
export FID2=out_mbii_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/jk_z2.py $FID2 $JK2 


export JK3=out_mbii_gp1_pp1_wgg1_nla2_z3_multinest_jkcov_rmin6_rmax33.txt
export FID3=out_mbii_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/jk_z3.py $FID3 $JK3 

# and this one to get the stats
postprocess -o $OUT --no-plots  $FID0 $FID1 $FID2 $FID3 