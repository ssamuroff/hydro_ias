cd /home/ssamurof/cosmosis
source config/setup-cosmosis
cd /home/ssamurof/hydro_ias/chains/rundir


export OUT=/home/ssamurof/hydro_ias/chains/postprocessed/mbii_tatt_bmodes

export C0=tatt/nobmodes/out_mbii_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_rmin4_rmax33.txt
export C1=out_mbii_gp1_pp1_wgg1_tatt2_z0_multinest_fidcov_rmin4_rmax33.txt

export SCRIPT=../plotting/tatt_bmodes.py

postprocess -o $OUT --no-plots --factor-kde=1.1 -f pdf --extra $SCRIPT $C1 $C0 
postprocess -o $OUT --no-plots --factor-kde=1.1 -f png --extra $SCRIPT $C1 $C0

