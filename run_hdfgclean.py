import argparse
import time # only used for logging here
from hdsims import utils # only used for logging here 
from hdfgclean import hdfgclean, fgclean_info as fgi, mpi # `mpi` only used for logging here

# --- command-line arguments: ---

# define descriptions that will be printed out in the 'help' message:
description = ("Run the CMB-HD FG cleaning procedure on a set of HD "
               "simulations, with options to match catalogs of detcted "
               "sources and clusters to the true catalogs, take the power "
               "spectra of the maps before and after FG cleaning, and plot "
               "the results.")
epilog = ("See the documentation of the `hdfgclean.hdfgclean.HDFGClean`, "
          "`hdfgclean.fgclean.FGClean`, and `hdsims.hdsims.HDSims` "
          "classes for information about the settings in the configuration "
          "file, and the `HDFGClean.run_hdfgclean` method for additional "
          "options.")
formatter_class = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter_class,
                                 description=description, epilog=epilog)

# loading the config file:
config_help = ("Path to a YAML configuration file containing the settings used"
               " to initialize the `hdfgclean.hdfgclean.HDFGClean` class. This"
               " file must contain the path to your `hd_sims_dir`, and may"
               " contain any other parameters accepted by `HDFGClean`.")
safe_help = ("Allow execution of arbitrary python code when loading the YAML"
             " configuration file by passing `safe=False` to"
             " `hdfgclean.fgutils.load_yaml`. This may be necessary if, e.g.,"
             " a numpy array was saved in the configuration file, but"
             " **NOTE** that this should only be done for YAML files from"
             " trusted sources. See the PyYAML documentation and"
             " `hdfgclean.fgutils.load_yaml` for more information.")
parser.add_argument('config_file', help=config_help)
parser.add_argument('--no_safe_load', action='store_true', help=safe_help)
# saving maps:
maps_help = ("Save the maps after FG cleaning, along with maps of the detected"
             " point sources and clusters at each frequency.")
parser.add_argument('--savemaps', action='store_true', help=maps_help)
# masks:
masks_help = ("Do not calculate and save the masks after FG cleaning;"
              " they are saved by default.")
parser.add_argument('--nomasks', action='store_true', help=masks_help)
parser.add_argument('--mask_freqs', default=fgi.spectra_freqs, type=int,
                    nargs='*', help="Mask frequencies (in GHz).")
# matching:
match_help = ("After FG cleaning, match the catalogs of detected sources"
              " and clusters to the catalogs of true sources and clusters"
              " in the maps.")
match_freqs_help = ("Frequencies (in GHz) for the matching."
                    " Uses all frequencies by default.")
parser.add_argument('--match', action='store_true', help=match_help)
parser.add_argument('--match_freqs', default=None, type=int,
                    nargs='*', help=match_freqs_help)
# power spectra:
spectra_help = "Take power spectra of the maps before and after FG cleaning."
parser.add_argument('--spectra',  action='store_true', help=spectra_help)
parser.add_argument('--spectra_freqs', default=fgi.spectra_freqs, type=int,
                    nargs='*', help="Power spectra frequencies (in GHz).")
# plots:
plots_help = ("Plot the FG cleaned maps, their power spectra (if it has been"
              " calculated by passing `--spectra`), and the match between"
              " detected and true sources and clusters (if the matching was"
              " done by passing `--match`).")
parser.add_argument('--plots', action='store_true', help=plots_help)
# testing the config file:
test_help = ("Just initialize `HDFGClean` from the `config_file` and exit."
             " Also prints out the number of smaller patches that the map will"
             " be divided into when running the FG cleaning. Note that this"
             " will create any necessary FG cleaning output directories.")
parser.add_argument('--test',  action='store_true', help=test_help)


# --- initialize the `HDFGClean` class: ---
args = parser.parse_args() # parse the arguments
log = utils.get_logger(name='hdfgclean', fmt="{message:s}") # to print out messages
use_safe_load = not args.no_safe_load
fgcleanlib = hdfgclean.HDFGClean.from_config(args.config_file, log=log, safe=use_safe_load)

if args.test:
    log.info(f"[MPI rank {mpi.rank:2d}] Successfully initialized `HDFGClean` from `{args.config_file}`.")
    if mpi.is_rank0: # print out how many smaller patches there are in the map
        log.info(f"The map will be divided into {fgcleanlib.patches.num_patches} smaller patche(s) when running FG cleaning.")

else: # run FG cleaning:
    t = time.time() # time it
    save_maps = args.savemaps
    masks = not args.nomasks
    fgcleanlib.run_hdfgclean(save_maps_after_subtraction=save_maps,
                             save_subtracted_sources_maps=save_maps,
                             save_subtracted_clusters_maps=save_maps,
                             make_masks=masks, mask_freqs=args.mask_freqs,
                             match_to_true_catalogs=args.match,
                             freqs_to_match=args.match_freqs,
                             take_power=args.spectra,
                             spectra_freqs=args.spectra_freqs,
                             make_plots=args.plots)
    log.info(f"[MPI rank {mpi.rank:2d}] {utils.tmsg(time.time() - t)} to run FG cleaning.")

