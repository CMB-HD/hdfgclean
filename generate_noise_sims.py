import os
import argparse
from hdsims import hdsims, utils
from hdfgclean import fgclean_info as fgi, hdfgclean_utils


# define command-line args and parse them:
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('hd_sims_dir', help='Path to your hdsims output directory')
parser.add_argument('--lowres-sims-dir', default=None, help='Path to the full-sky S10 sims & catalogs')
parser.add_argument('--pol', action='store_true', help="Whether to generate CMB polarization (Q and U) maps in addition to the temperature map")
args = parser.parse_args()

# --- run hdsims: ---

# check if the lower-resolution maps have been saved:
bash_file_to_run = hdfgclean_utils.save_commands_to_download_s10_files(args.hd_sims_dir, verbose=True)
if bash_file_to_run is not None:
    print(f"\nAfter downloading the necessary lower-resolution simulation files, re-run this python script.")

# initialize the HDSims class:
log = utils.get_logger(name='hdsims', fmt="{message:s}") # use logging to print out messages as they are logged
kwargs = {**fgi.noise_map_kwargs, 'lowres_sims_dir': args.lowres_sims_dir, 'pol': args.pol, 'verbose': True, 'log': log}
simlib = hdsims.HDSims(os.path.abspath(args.hd_sims_dir), **kwargs)

# generate the sims:
simlib.generate_hd_sims()
# also save the apodization window (used when convolving the beam):
if not os.path.exists(simlib.get_apod_window_fname(shape=simlib.padded_shape, wcs=simlib.padded_wcs)):
    simlib.get_apod_window(shape=simlib.padded_shape, wcs=simlib.padded_wcs, save=True)

