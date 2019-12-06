cd /home/ssamurof/cosmosis
source config/setup-cosmosis 
cd ../hydro_ias/chains/rundir

postprocess -o ~/test --factor-kde=1.1 --no-plots --extra ../plotting/tatt_direct_forecasts.py out_tatt_field_R32.txt out_tng500_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_rmin6_rmax33.txt out_tng1000_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_rmin6_rmax33.txt

postprocess -o ~/test --factor-kde=1.1 --no-plots -f pdf --extra ../plotting/tatt_direct_forecasts.py out_tatt_field_R32.txt out_tng500_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_rmin6_rmax33.txt out_tng1000_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_rmin6_rmax33.txt