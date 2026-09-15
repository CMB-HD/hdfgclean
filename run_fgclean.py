import argparse
import time # only used for logging here
from hdsims import utils # only used for logging here
from hdfgclean import fgclean, mpi # `mpi` only used for logging here


# --- set up to parse command-line arguments ---

# define descriptions that will be printed out in the 'help' message:
description = "Run the CMB-HD FG cleaning procedure on a set of maps."
epilog = ("See the documentation of the `hdfgclean.fgclean.FGClean` class for"
          " information about the settings in the configuration file, and the "
          "`FGClean.run_fgclean` method for additional options.")
formatter_class = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter_class,
                                 description=description, epilog=epilog)

# loading the config file:
config_help = ("Path to a YAML configuration file containing the settings used"
               " to initialize the `hdfgclean.fgclean.FGClean` class. This"
               " file must contain the path to the input map and the beam size"
               " at each frequency, and may contain any other parameters"
               " accepted by `FGClean`.")
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
maps_help = ("Save the maps after FG cleaning, along with maps of the"
             " detected point sources and clusters at each frequency.")
parser.add_argument('--savemaps', action='store_true', help=maps_help)
# masks:
masks_help = ("Do not calculate and save the masks after FG cleaning;"
              " By default, a mask is saved for each map frequency.")
parser.add_argument('--nomasks', action='store_true', help=masks_help)
# testing the config file:
test_help = ("Just initialize `FGClean` from the `config_file` and exit."
             " Also prints out the number of smaller patches that the map"
             " will be divided into when running the FG cleaning. Note that"
             " this will create any necessary FG cleaning output directories.")
parser.add_argument('--test',  action='store_true', help=test_help)
args = parser.parse_args()


# --- initialize the `FGClean` class: ---
args = parser.parse_args() # parse the arguments
log = utils.get_logger(name='fgclean', fmt="{message:s}") # to print out messages
use_safe_load = not args.no_safe_load
fgcleanlib = fgclean.FGClean.from_config(args.config_file, log=log, safe=use_safe_load)

if args.test:
    log.info(f"[MPI rank {mpi.rank:2d}] Successfully initialized `FGClean` from `{args.config_file}`.")
    if mpi.is_rank0: # print out how many smaller patches there are in the map
        num_patches = fgcleanlib.patches.num_patches
        log.info(f"The map will be divided into {num_patches} smaller patches when running FG cleaning.")

else: # run FG cleaning:
    t = time.time() # time it
    save_maps = args.savemaps
    masks = not args.nomasks
    fgcleanlib.run_fgclean(save_maps_after_subtraction=save_maps,
                           save_subtracted_sources_maps=save_maps,
                           save_subtracted_clusters_maps=save_maps,
                           make_masks=masks)
    log.info(f"[MPI rank {mpi.rank:2d}] {utils.tmsg(time.time() - t)} to run FG cleaning.")

