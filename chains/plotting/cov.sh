cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/mbii_cov_test

export FID0=out_mbii_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration1_rmin6_rmax33.txt
export FID1=out_mbii_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration2_rmin6_rmax33.txt
export FID2=out_mbii_gp1_pp1_wgg1_nla2_z0_multinest_fidcov_iteration3_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/cov_z0.py $FID0 $FID1 $FID2 
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/cov_z0.py $FID0 $FID1 $FID2 


export FID0=out_mbii_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration1_rmin6_rmax33.txt
export FID1=out_mbii_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration2_rmin6_rmax33.txt
export FID2=out_mbii_gp1_pp1_wgg1_nla2_z1_multinest_fidcov_iteration3_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/cov_z1.py $FID0 $FID1 $FID2 
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/cov_z1.py $FID0 $FID1 $FID2 

export FID0=out_mbii_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration1_rmin6_rmax33.txt
export FID1=out_mbii_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration2_rmin6_rmax33.txt
export FID2=out_mbii_gp1_pp1_wgg1_nla2_z2_multinest_fidcov_iteration3_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/cov_z2.py $FID0 $FID1 $FID2 
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/cov_z2.py $FID0 $FID1 $FID2 

export FID0=out_mbii_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration1_rmin6_rmax33.txt
export FID1=out_mbii_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration2_rmin6_rmax33.txt
export FID2=out_mbii_gp1_pp1_wgg1_nla2_z3_multinest_fidcov_iteration3_rmin6_rmax33.txt

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra ../plotting/cov_z3.py $FID0 $FID1 $FID2 
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra ../plotting/cov_z3.py $FID0 $FID1 $FID2 



# and this one to get the stats
postprocess -o $OUT --no-plots  $FID0 $FID1 $FID2 