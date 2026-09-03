import os
import numpy as np
from hdsims import siminfo as si, simutils


# ----- general -----

max_patch_size = 3
patch_apod_width = 0.25

p2d_smooth_npix = 3

# RMS maps
rms_niter = 10
rms_nsigma = 3
rms_smooth = True
rms_smooth_pix = None
rms_fixed_gw = False
rms_use_overlap_pix = True
rms_gw_sources = 10 # arcmin
rms_gw_clusters = 40 # arcmin

# iterative FG cleaning
sources_snr_threshold_list = [250, 100, 75, 50, 40, 30, 25, 20, 15, 12.5, 10, 7.5, 5, 4]
remeasure_sources_snr_threshold_list = [250, 100, 50, 25, 15, 10, 5, 4]
min_num_iter_sources_per_snr = 10 # once we find fewer than this many sources, move on to next SNR threshold
max_ntimes_remeasure_sources = 25
clusters_snr_threshold_list = [50, 25, 15, 12.5, 10, 7.5, 5, 4]
clusters_iter_match_radius = 1 # for matching clusters measured on same iter w/ different profiles/filters


# spectral index for sources:
index_min_snr = 10
index_nsigma_to_remove = 2.5

cluster_profile_sigmas = np.arange(0.25, 0.8, 0.05)
min_snr_for_extrap = None

# masks
mask_snr_threshold = 5
mask_apod_width = 1 # arcmin
mask_abs_snr = True
mask_snr_threshold_below = None


# ----- for hd / hdsims -----

freqs = [90, 148, 219, 277] # for FG cleaning
spectra_freqs = [90, 148] # for power spectra

# matching detected sources/clusters to true sources/clusters:
cluster_match_radius = 1 # arcminute
max_abs_flux_diff_vs_err = 3

# noise maps used in matched filter calculations:
noise_map_ra_ctr = 26
noise_map_dec_ctr = si.dec_ctr
noise_map_cmb_seed = 15
noise_map_noise_seeds = {freq: freq for freq in si.freqs}
noise_map_cib_model = si.baseline_cib_model_name
# we provide catalogs for either 4 square degree or 9 square degree 
# noise maps used for the matched filter calculations:
noise_map_widths = [2, 3]
noise_map_apod_widths = {2: 0.2, 3: 0.25}
# paths to catalogs used to get noise maps for matched filter calculations:
noise_map_source_catalog_files = {}
noise_map_cluster_catalog_files = {}
noise_map_catalogs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'catalogs_for_noise_maps')
_noise_map_ctr = f"ra{simutils.round_str(noise_map_ra_ctr)}dec{simutils.round_str(noise_map_dec_ctr)}"
for w in noise_map_widths:
    _noise_map_size = f"{simutils.round_str(w)}x{simutils.round_str(w)}deg"
    _noise_map_apod = f"apod{simutils.round_str(noise_map_apod_widths[w])}deg"
    _catalog_dir_name = '_'.join([_noise_map_ctr, _noise_map_size, _noise_map_apod])
    _catalog_dir = os.path.join(noise_map_catalogs_dir, _catalog_dir_name)
    noise_map_source_catalog_files[w] = {f: os.path.join(_catalog_dir, f'{f:03d}GHz_sources.csv') for f in freqs}
    noise_map_cluster_catalog_files[w] = os.path.join(_catalog_dir, 'clusters.csv')
# defaults:
noise_map_width = 3
noise_map_height = noise_map_width
noise_map_apod_width = noise_map_apod_widths[noise_map_width]
noise_map_source_catalog_fnames = noise_map_source_catalog_files[noise_map_width]
noise_map_cluster_catalog_fname = noise_map_cluster_catalog_files[noise_map_width]

noise_map_kwargs = {'freqs': freqs, 'pol': False, 
                    'ra_ctr': noise_map_ra_ctr, 'dec_ctr': noise_map_dec_ctr,
                    'width': noise_map_width, 'height': noise_map_height, 
                    'apod_width': noise_map_apod_width, 
                    'cmb_seed': noise_map_cmb_seed,
                    'noise_seeds': noise_map_noise_seeds}


# masking known, very bright sources before FG cleaning:
min_masked_flux = 1000 # mJy
source_mask_apod_width = 1 # arcmin
# masking known large, massive, nearby clusters after FG cleaning:
min_M500_to_mask = 2e14 # solar masses
max_z_to_mask = 0.2
min_cluster_mask_radius = 5 # arcmin
cluster_mask_apod_width = 5 # arcmin

