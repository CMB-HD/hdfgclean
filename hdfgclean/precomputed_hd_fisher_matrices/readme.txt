This directory contains pre-computed Fisher matrices using the simulation-based (`v1.2` in `hdMockData`) CMB-HD delensed TT/TE/EE/BB and CMB lensing (kk) mock data (theory power spectra, noise curves, and covariance matrices; all depend on the simulation-based temperature FG+noise curves of MacInnis et. al. 2026). In some cases, CMB-HD is combined with the mock DESI BAO data in `hdfisher` (from arXiv:2309.03021). We also provide the fiducial (CAMB) parameters used for the theory calculation. 

These files are only used in the `reproduce_10x10.ipynb` notebook, located in the `reproduce_10x10/` directory of the `hdfgclean` github repository. See the `hdfisher` repository (https://github.com/CMB-HD/hdfisher) for the code used to calculate the Fisher matrices.


The CAMB parameter (`.yaml`) files contain the HD `v1.2` CAMB settings for models without (`fiducial_parameters.yaml`) and with (`fiducial_parameters_baryonic_feedback.yaml`) baryonic feedback; these are in a format that can be passed to the `hdfisher` code.


All Fisher matrices have the Gaussian prior of `sigma(tau) = 0.005` applied; in models with baryonic feedback, a 0.06% prior is applied to the `HMCode_logT_AGN` parameter. The CMB-HD covariance matrices used to calculate them are for a sky fraction `fsky = 0.59`. 

We calculate Fisher matrices for three sets of varied parameters:
- LCDM+Neff+mnu model (8 parameters): vary the six LCDM parameters, the effective number of relativistic species, and the sum of the neutrino masses (in eV) ; no baryomic feedback
- LCDM+Neff+mnu+TAGN model (9 parameters): vary the six LCDM parameters, the effective number of relativistic species, the sum of the neutrino masses (in eV), and the strength of baryonic feedback which is characterized by the HMcode parameter log(T_AGN)
- LCDM+Neff+mnu+TAGN+kSZ model (11 parameters): vary the six LCDM parameters, the effective number of relativistic species, the sum of the neutrino masses (in eV),  the strength of baryonic feedback, and parameters for the amplitude (A_kSZ) and slope (n_kSZ) of the kSZ power spectrum.

We also calculate Fisher matrices using two sets of the v1.2 simulation-based HD mock data; they differ in the way that the lensing reconstruction noise is calculated (which changes both the delensed power spectra and covariance matrix):
- "MV" lensing uses the MV combination of the TT, TE, TB, EE, and EB lensing estimators
- "pol-only" lensing uses only the EE and EB estimators
See MacInnis et. al. (2026) for details.


The Fisher matrix files in this directory are described below:

- `cdm/lcdm_nnu_mnu_delensedCMB_BAO.txt`: CMB-HD delensed TT/TE/EE/BB + lensing kk using "MV" lensing, combined with DESI BAO, for the 8-parameter LCDM+Neff+mnu model

- `cdm_kkMVpol/lcdm_nnu_mnu_delensedCMB_BAO.txt`: CMB-HD delensed TT/TE/EE/BB + lensing kk using "pol-only" lensing, combined with DESI BAO, for the 8-parameter LCDM+Neff+mnu model

- `cdm_baryons/` directory: Fisher matrices for the 9-parameter LCDM+Neff+mnu+TAGN and 11-parameter LCDM+Neff+mnu+TAGN+kSZ models, using "MV" lensing
  - `lcdm_nnu_mnu_TAGN_delensedCMB_BAO.txt`: CMB-HD delensed TT/TE/EE/BB + lensing kk combined with DESI BAO, for the 9-parameter LCDM+Neff+mnu+TAGN model
  - `lcdm_nnu_mnu_TAGN_delensedCMB_kk_BAO.txt`: only CMB-HD lensing kk combined with DESI BAO, for the 9-parameter LCDM+Neff+mnu+TAGN model; in this case, additional priors of `sigma(ombh2) = 0.00036` and `sigma(ns) = 0.02` are also applied.
  - `lcdm_nnu_mnu_TAGN_delensedCMB_tt.txt`: only CMB-HD delensed TT, for the 9-parameter LCDM+Neff+mnu+TAGN model
  - `lcdm_nnu_mnu_TAGN_ksz_delensedCMB_BAO.txt`: CMB-HD delensed TT/TE/EE/BB + lensing kk combined with DESI BAO, for the 11-parameter LCDM+Neff+mnu+TAGN+kSZ model
  - `lcdm_nnu_mnu_TAGN_ksz_delensedCMB_BAO_v1.1HDdata.txt`: same as above, but calculated using the previous (`v1.1`) HD mock data from arXiv:2405.12220
  
- `cdm_baryons_kkMVpol/` directory: Fisher matrices for the 9-parameter LCDM+Neff+mnu+TAGN and 11-parameter LCDM+Neff+mnu+TAGN+kSZ models, using "pol-only" lensing
  - `lcdm_nnu_mnu_TAGN_delensedCMB_BAO.txt`: CMB-HD delensed TT/TE/EE/BB + lensing kk combined with DESI BAO, for the 9-parameter LCDM+Neff+mnu+TAGN model
  - `lcdm_nnu_mnu_TAGN_delensedCMB_ee_bb.txt`: only CMB-HD delensed EE/BB, for the 9-parameter LCDM+Neff+mnu+TAGN model
  - `lcdm_nnu_mnu_TAGN_delensedCMB_te_ee_bb_kk_BAO.txt`: only CMB-HD delensed TE/EE/BB + lensing kk combined with DESI BAO, for the 9-parameter LCDM+Neff+mnu+TAGN model
  - `lcdm_nnu_mnu_TAGN_delensedCMB_te_ee_bb.txt`: only CMB-HD delensed TE/EE/BB, for the 9-parameter LCDM+Neff+mnu+TAGN model
  - `lcdm_nnu_mnu_TAGN_delensedCMB_te.txt`: only CMB-HD delensed TE, for the 9-parameter LCDM+Neff+mnu+TAGN model
  - `lcdm_nnu_mnu_TAGN_ksz_delensedCMB_BAO.txt`: CMB-HD delensed TT/TE/EE/BB + lensing kk combined with DESI BAO, for the 11-parameter LCDM+Neff+mnu+TAGN+kSZ model

