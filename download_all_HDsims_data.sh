#!/bin/bash

datadir=${1:-$(pwd)}
echo "HD sims files will be placed in $datadir"

cd $datadir

if [ ! -d "ra6dec6_10x10deg_hdsims" ]; then
  mkdir ra6dec6_10x10deg_hdsims
fi
cd ra6dec6_10x10deg_hdsims


urls=(
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/unlensed_cmbTQU0058_pixwin_11x11deg.fits"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/cmbTQU0058_pixwin_11x11deg.fits"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/kappa_11x11deg.fits"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/ksz_pixwin_11x11deg.fits"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/all_allgz_pixwin_11x11deg.tar.gz"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/radio_11x11deg.csv"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/cib_11x11deg.csv"
  "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/sz_11x11deg.csv"
)

for url in "${urls[@]}"; do
  echo "Downloading $url"
  wget -c "$url"
done

if [ -f "all_allgz_pixwin_11x11deg.tar.gz" ]; then
  echo "Extracting all_allgz_pixwin_11x11deg.tar.gz ..."
  tar -xzf all_allgz_pixwin_11x11deg.tar.gz
fi


if [ ! -d "spectra" ]; then
  mkdir spectra
fi
cd spectra

wget -c "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/hdsims_spectra_cls.tar.gz"
tar -xzf hdsims_spectra_cls.tar.gz
mv hdsims_spectra_cls/*.txt .
rmdir hdsims_spectra_cls
wget -c "https://lambda.gsfc.nasa.gov/data/simulation/hdsims/hdsims_spectra_dls.tar.gz"
tar -xzf hdsims_spectra_dls.tar.gz
mv hdsims_spectra_dls/*.txt .
rmdir hdsims_spectra_dls
