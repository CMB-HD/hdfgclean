SIMDIR=${1?"Error: You must provide the path to your hd_sims_dir."}

if [ ! -d "${SIMDIR}" ]; then
  mkdir $SIMDIR
fi

if [ ! -d "${SIMDIR}/ra26dec6_3x3deg_hdsims" ]; then
  mkdir $SIMDIR/ra26dec6_3x3deg_hdsims
fi

if [ ! -d "${SIMDIR}/ra26dec6_3x3deg_hdsims/intermediate_maps" ]; then
  mkdir $SIMDIR/ra26dec6_3x3deg_hdsims/intermediate_maps
fi

# maps:
wget -c https://raw.githubusercontent.com/CMB-HD/sim_files_for_example_notebooks/refs/heads/main/ra26dec6_3x3deg_hdsims/intermediate_maps/090_tsz_4.1x4.1deg_s10.fits -O $SIMDIR/ra26dec6_3x3deg_hdsims/intermediate_maps/090_tsz_4.1x4.1deg_s10.fits
wget -c https://raw.githubusercontent.com/CMB-HD/sim_files_for_example_notebooks/refs/heads/main/ra26dec6_3x3deg_hdsims/intermediate_maps/148_tsz_4.1x4.1deg_s10.fits -O $SIMDIR/ra26dec6_3x3deg_hdsims/intermediate_maps/148_tsz_4.1x4.1deg_s10.fits
wget -c https://raw.githubusercontent.com/CMB-HD/sim_files_for_example_notebooks/refs/heads/main/ra26dec6_3x3deg_hdsims/intermediate_maps/219_tsz_4.1x4.1deg_s10.fits -O $SIMDIR/ra26dec6_3x3deg_hdsims/intermediate_maps/219_tsz_4.1x4.1deg_s10.fits
wget -c https://raw.githubusercontent.com/CMB-HD/sim_files_for_example_notebooks/refs/heads/main/ra26dec6_3x3deg_hdsims/intermediate_maps/277_tsz_4.1x4.1deg_s10.fits -O $SIMDIR/ra26dec6_3x3deg_hdsims/intermediate_maps/277_tsz_4.1x4.1deg_s10.fits
wget -c https://raw.githubusercontent.com/CMB-HD/sim_files_for_example_notebooks/refs/heads/main/ra26dec6_3x3deg_hdsims/intermediate_maps/ksz_4.1x4.1deg_s10.fits -O $SIMDIR/ra26dec6_3x3deg_hdsims/intermediate_maps/ksz_4.1x4.1deg_s10.fits
wget -c https://raw.githubusercontent.com/CMB-HD/sim_files_for_example_notebooks/refs/heads/main/ra26dec6_3x3deg_hdsims/intermediate_maps/kappa_4.1x4.1deg_s10.fits -O $SIMDIR/ra26dec6_3x3deg_hdsims/intermediate_maps/kappa_4.1x4.1deg_s10.fits

# catalogs:
wget -c https://raw.githubusercontent.com/CMB-HD/sim_files_for_example_notebooks/refs/heads/main/ra26dec6_3x3deg_hdsims/intermediate_maps/sz_4.1x4.1deg.txt -O $SIMDIR/ra26dec6_3x3deg_hdsims/intermediate_maps/sz_4.1x4.1deg.txt
wget -c https://raw.githubusercontent.com/CMB-HD/sim_files_for_example_notebooks/refs/heads/main/ra26dec6_3x3deg_hdsims/intermediate_maps/cib_hd_4.1x4.1deg.csv -O $SIMDIR/ra26dec6_3x3deg_hdsims/intermediate_maps/cib_hd_4.1x4.1deg.csv
wget -c https://raw.githubusercontent.com/CMB-HD/sim_files_for_example_notebooks/refs/heads/main/ra26dec6_3x3deg_hdsims/intermediate_maps/radio_4.1x4.1deg.txt -O $SIMDIR/ra26dec6_3x3deg_hdsims/intermediate_maps/radio_4.1x4.1deg.txt
