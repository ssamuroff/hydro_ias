cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd ../hydro_ias/chains/rundir


postprocess -o ~/test --factor-kde=1.1 --no-plots -f pdf --extra=../plotting/tatt_direct_bias.py out_tatt_field_R16_avg2.txt out_tatt_field_R32_avg2.txt out_tatt_field_R64_avg2.txt out_tatt_field_R128_avg2.txt out_tng_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt 

postprocess -o ~/test --factor-kde=1.1 --no-plots -f png --extra=../plotting/tatt_direct_bias.py out_tatt_field_R16_avg2.txt out_tatt_field_R32_avg2.txt out_tatt_field_R64_avg2.txt out_tatt_field_R128_avg2.txt out_tng_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_iteration4_rmin6_rmax33.txt