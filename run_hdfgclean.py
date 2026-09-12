import argparse
import time # only used for logging here
from hdsims import utils # only used for logging here 
from hdfgclean import hdfgclean, mpi # `mpi` only used for logging here

# --- set up to parse command-line arguments ---

# define descriptions that will be printed out in the 'help' message:
description = "Run the CMB-HD FG cleaning procedure on a set of HD simulations, with options to match catalogs of detcted sources and clusters to the true catalogs, take the power spectra of the maps before and after FG cleaning, and plot the results."
epilog = "See the documentation of the `hdfgclean.hdfgclean.HDFGClean`, `hdfgclean.fgclean.FGClean`, and `hdsims.hdsims.HDSims` classes for information about the settings in the configuration file, and the `HDFGClean.run_hdfgclean` method for additional options."

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description, epilog=epilog)
parser.add_argument('config_file', help="Path to a YAML configuration file containing the settings used to initialize the `hdfgclean.hdfgclean.HDFGClean` class. This file must contain the path to your `hd_sims_dir`, and may contain any other parameters accepted by `HDFGClean`.")
parser.add_argument('--match', action='store_true', help="After FG cleaning, match the catalogs of detected sources and clusters to the catalogs of true sources and clusters in the maps.")
parser.add_argument('--spectra',  action='store_true', help="Take the power spectra of the input and FG cleaned 90 and 148 GHz maps.")
parser.add_argument('--plots', action='store_true', help="Plot the FG cleaned maps, their power spectra (if it has been calculated by passing `--spectra`), and the match between detected and true sources and clusters (if the matching was done by passing `--match`).")
parser.add_argument('--test',  action='store_true', help="Just initialize `HDFGClean` from the `config_file` and exit. Also prints out the number of smaller patches that the map will be divided into when running the FG cleaning. Note that this will create any necessary FG cleaning output directories.")
args = parser.parse_args()

# --- initialize the `HDFGClean` class: ---
log = utils.get_logger(name='hdfgclean', fmt="{message:s}") # use logging to print out messages as they are logged
fgcleanlib = hdfgclean.HDFGClean.from_config(args.config_file, log=log)

if args.test:
    log.info(f"[MPI rank {mpi.rank:2d}] Successfully initialized `HDFGClean` from `{args.config_file}`.")
    if mpi.is_rank0: # print out how many smaller patches there are in the map
        log.info(f"The map will be divided into {fgcleanlib.patches.num_patches} smaller patche(s) when running FG cleaning.")

else: # run FG cleaning:
    t = time.time() # time it
    fgcleanlib.run_hdfgclean(take_power=args.spectra, match_to_true_catalogs=args.match, make_plots=args.plots)
    log.info(f"[MPI rank {mpi.rank:2d}] {utils.tmsg(time.time() - t)} to run FG cleaning.")

