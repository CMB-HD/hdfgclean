# Foreground cleaning for CMB-HD

This repository contains code to run the extragalactic foreground (FG) cleaning procedure presented in [MacInnis, Ange, Sehgal, Kable, and Blackstad (2026)](https://arxiv.org/abs/XXXX.XXXXX) (**TODO:LINK2PAPER**), which detects and removes the thermal SZ (tSZ), cosmic infrared background (CIB), and radio galaxies from a set of ultrahigh-resolution lensed CMB temperature maps that also include the kinetic SZ (kSZ) and instrumental noise. Please cite that work if you use this code.

The code we provide here can be used to iteratively detect, measure, and remove the CIB, radio galaxies, and tSZ clusters from a set of maps at multiple frequencies.

For maps that were generated with the [hdsims](https://github.com/CMB-HD/hdsims) package (including the ultrahigh-resolution simulations available on [LAMBDA](https://lambda.gsfc.nasa.gov/simulation/ultrahigh_resolution_sims.html)), we also provide methods to:
- Take the power spectrum of the maps (or any combination of their individual components) before or after FG cleaning, with or without applying a mask.
- Match the catalogs of detected point sources (CIB + radio) or clusters to the catalogs of all true point sources or clusters in the maps.
- Generate the relevant plots of MacInnis et. al. (2026)

The main code we provide can be run with `run_hdfgclean.py` for maps generated with `hdsims`, or `run_fgclean.py` for a more general set of maps compatible with the [pixell](https://pixell.readthedocs.io/en/latest/readme.html) package (see below for further details).

We also provide python code in the `reproduce_10x10` directory to identically reproduce the 100-square-degree FG-cleaned maps presented in MacInnis et. al. (2026). In the `examples` directory we provide (**TODO**) a similar example using a smaller set of four-square-degree maps, and (**TODO**) an example to run the foreground cleaning on a different set of maps.

---

The rest of this readme contains the following sections:

- [Installation instructions](#installation-instructions)
  - [Required packages](#required-packages)
- [Overview of foreground cleaning](#overview-of-foreground-cleaning)
- [How to use the code](#how-to-use-the-code)
  - [Using MPI (strongly recommended)](#using-mpi-strongly-recommended)
  - [Using `HDFGClean` with `hdsims`](#using-hdfgclean-with-hdsims)
    - [Before using `HDFGClean`](#before-using-hdfgclean)
    - [Initializing `HDFGClean`](#initializing-hdfgclean)
    - [Using `HDFGClean`](#using-hdfgclean)
  - [Using `FGClean` with any map(s)](#using-fgclean-with-any-maps)
- [Reproducing the results of MacInnis et. al. (2026)](#reproducing-the-results-of-macinnis-et-al-2026)
- [Examples](#examples)

---


## Installation instructions

To install the `hdfgclean` package, navigate to the directory where you would like to place this repository. Then clone and install it via `pip`:

```
git clone https://github.com/CMB-HD/hdfgclean.git
cd hdfgclean
pip install . --user
```


### Required packages

To use the `hdfgclean` code, you will need to install Python 3 and several Python packages. We make use of Python 3.11.6 and:
- [hdsims](https://github.com/CMB-HD/hdsims)
- [hdMockData](https://github.com/CMB-HD/hdMockData) 
- [numpy](https://numpy.org/) 1.26.0
- [scipy](https://scipy.org/) 1.11.3
- [pandas](https://pandas.pydata.org/) 2.1.3
- [matplotlib](https://matplotlib.org/) 3.8.1
- [pixell](https://pixell.readthedocs.io/) 0.23.14
- [camb](https://camb.readthedocs.io/) 1.5.4
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation) 6.0.3
- [mpi4py](https://mpi4py.readthedocs.io) 4.1.1 (optional but *strongly* recommended)


If you would like to reproduce the parameter forecasts of MacInnis et. al. (2026) (**TODO**: refer to relevant section in readme), you will also need to install [hdfisher](https://github.com/CMB-HD/hdfisher) and [getdist](https://getdist.readthedocs.io).

---


## Overview of foreground cleaning

Below is a brief summary of the foreground cleaning procedure of [**TODO:LINK2PAPER**](https://arxiv.org/abs/XXXX.XXXXX). (You do not need to read this section before running the code we provide here, but it may provide some helpful context.)

We start with a set of CMB temperature maps at different frequencies, containing the tSZ and/or CIB and radio sources.
> **TODO** (make a note somewhere to clarify this?): code allows you to turn off source/cluster subtraction (so you can, for whatever reason, use maps that don't have tSZ or don't have point sources); *but* for sources, if doing the "extrapolation" step (default behavior), it assumes the map has both CIB and radio - i.e., can't turn off extrapolation for only CIB / only radio
- In MacInnis et. al. (2026), we used 0.04 arcminute-resolution simulations generated with `hdsims` (**TODO** : refer them to different section for more info) at 90, 148, 219, and 277 GHz; each map is the sum of the beam- and pixel-window-convolved lensed CMB, kSZ, tSZ, CIB, and radio galaxies, plus instrumental noise, using the instrumental noise levels and beam sizes for [CMB-HD](https://cmb-hd.org/survey-and-instrument-overview/).


The foreground cleaning is done iteratively by first subtracting point sources or tSZ clusters detected with a signal-to-noise ratio (SNR) above some threshold from the map(s), and then moving down to lower SNR thresholds until the minimum (by default, SNR = 4) is reached. On a given iteration, we take the map(s) after the previous iteration and:
- _Filter_ the map(s) to isolate either point sources or clusters by applying a filter that is matched to the shape of the object.
  - We use the beam profile at a given frequency when filtering the maps to isolate point sources; each frequency map is filtered independently of the others.
  - We use a set of different cluster profiles when filtering the maps to isolate tSZ clusters; for each profile, a multi-frequency matched filter is applied to all maps simultaneously, producing a single filtered map in Compton-y units.
- _Detect_ point sources or clusters as regions (i.e., groups of pixels) in the filtered map(s) with SNR above a given threshold, and _measure_ the position and amplitude (either the flux for point sources, or the amplitude of the maximum-SNR pixel in Compton-y units) of each.
- _Subtract_ these point sources or clusters from the map(s) at their measured positions, using their measured amplitude and the profile of the filter used to detect them.


We first remove CIB and radio point sources from the maps at each frequency, and then remove the tSZ clusters from the source-subtracted maps at all frequencies simultaneously; we do not attempt to remove any other foregrounds, such as the kSZ.
- After subtracting all sources detected with SNR above the minimum threshold from each map, we calculate an average $\nu_\mathrm{max}$ - to - $\nu$ CIB spectral index using bright sources detected at both frequencies, and then identify any CIB sources detected at the maximum frequency $\nu_\mathrm{max}$ but not at a lower frequency $\nu < \nu_\mathrm{max}$. We use the measured spectral index to extrapolate the fluxes of these CIB sources from $\nu_\mathrm{max}$ to the lower frequency $\nu$, and subtract them from the map at frequency $\nu$.
- Similarly, we identify any radio sources detected at the lowest frequency $\nu_\mathrm{min}$ but not at a higher frequency $\nu > \nu_\mathrm{min}$, and use the same approach to subtract these radio sources from the map at frequency $\nu$.

> **TODO** : we don't do the extrapolation+subtraction of radio sources at 277 GHz (b/c much fewer radio sources than CIB, so any residual dim radio at 277 won't make a big difference ; and also b/c we are just using 277 to do fg cleaning on 90/148)

---


## How to use the code

The `hdfgclean` package provides two python classes with methods to run the FG cleaning on a set of maps and calculate or the results: 
- The `HDFGClean` class in the `hdgfclean.py` module runs the FG cleaning on maps generated by [hdsims](https://github.com/CMB-HD/hdsims), matches the catalogs of detected point sources and clusters to the true catalogs, and calculates the power spectra of the maps.
- The `FGClean` class in the `fgclean.py` module is designed to run the FG cleaning on any set of maps. (**TODO**: the maps must be compatible with pixell - not sure if they must be CAR maps, or just maps with rectangular pixels?)

Both classes will save a catalog of all detected point sources at each frequency and a catalog of all detected clusters. By default, the FG-cleaned maps will also be saved, and the masks of residual sources and clusters will be calculated and saved. Both classes provide methods to access these results. (**TODO**: or any others for hdfgclean - mention this?)


> **TODO**: make a note that `HDFGClean` and `FGClean` use the same code? `HDFGClean` is just a "wrapper" around both hdsims and the `FGClean` class (don't want to give the impression that we treat the hdsims differently)

> **TODO**: also make a note that the code (all of it, including everything in `FGClean`) has only been tested using hdsims (not any other sims)


### Using MPI (strongly recommended)

By default, we divide maps that are larger than $3^\circ \times 3^\circ$ into a grid of smaller patches (with a default maximum size of $3^\circ \times 3^\circ$), and run the full FG cleaning procedure on each individual patch. This may (and should) be done in parallel using MPI; we recommend using one MPI process per smaller patch.
- For reference, the maps used in MacInnis et. al. (2026) were divided into 25 patches, and it took about five hours to run the FG cleaning on all patches simultaneously; this would have taken about 25 times as long without MPI.


After the FG cleaning has finished running on all patches, we combine the output of each into a set of maps and catalogs for the full region of the input maps.


Note that `HDFGClean` also provides methods to match the detected source/cluster catalogs to their true counterparts and to calculate the power spectra of the maps in parallel with MPI; see (**TODO**: where?) for more information. Here, we note that:
- The matching is done on each patch, but does not require as much memory as the FG cleaning, so we recommend that you still use one MPI process per patch, but spread across fewer nodes.
- The power spectra is calculated for the full-sized maps (as opposed to the smaller patches). This requires fewer MPI processes but more memory than the FG cleaning, so we recommend only using one or two MPI processes per node for this part (depending on your computing resources).


### Using `HDFGClean` with `hdsims`


#### Before using `HDFGClean`:

> **TODO** : technically, if they download the full-sky S10 sims and pass their `lowres_sims_dir` to `HDFGClean`, they don't *need* to do any of this ; but if they run FG cleaning with MPI, the sims will be generated using a single MPI process, so the other processes may be waiting for hours until this is finished


You **must** save the set of maps you wish to foreground-clean before initializing the `HDFGClean` class. By default, the HD simulations available on [LAMBDA](https://lambda.gsfc.nasa.gov/simulation/ultrahigh_resolution_sims.html) are used. The `download_all_HDsims_data.sh` script provided in this repository will download them; it can be run with the command

```
bash download_all_HDsims_data.sh /path/to/myHDsims
```

where `/path/to/myHDsims` is the path to the directory where you would like to save the simulations (about 66 GB of files). You must then provide this path to the `HDFGClean` class.

- Alternatively, you may generate a different set of simulations by following the instructions in [hdsims](https://github.com/CMB-HD/hdsims), and then initializing the `HDFGClean` class with the same arguments you passed to the `hdsims.hdsims.HDSims` class. (**TODO** : refer them to further instructions/details elsewhere)



By default, we also use a different set of $3^\circ \times 3^\circ$ maps generated by `hdsims` for the matched filter calculations (see MacInnis et. al. 2026 for further details). You **must** generate these maps before running any FG cleaning. To do so, first save their lower-resolution counterparts (112 MB of files, cut out from the full-sky [simulations](https://lambda.gsfc.nasa.gov/simulation/full_sky_sims_ov.html) of [Sehgal et. al. (2010)](https://arxiv.org/abs/0908.0540)) by running

```
bash download_s10sims_for_noise_maps.sh /path/to/myHDsims
```

and then generate the necessary higher-resolution maps (about 4 GB of files) by running

```
python generate_noise_sims.py /path/to/myHDsims
```

where the `download_s10sims_for_noise_maps.sh` and `generate_noise_sims.py` files are provided in this repository.


> **TODO** : tell them where they can find more information about all of this (e.g. how to use non-default hdsims to fg clean or for noise)


#### Initializing `HDFGClean`

There are two **required** arguments that you must pass to `HDFGClean`:
- A path to an `outdir_dir` where you would like to put the files saved by `HDFGClean`, and
- The path to your `hd_sims_dir` (e.g., `hd_sims_dir = '/path/to/myHDsims'`) where you have saved both sets of maps mentioned above: the maps you would like to apply the FG cleaning to, and the maps used to quantify the noise in the matched filter calculation.


`HDFGClean` also accepts additional, optional keyword arguments. The file `hdfgclean_defaults.yaml` provided here lists all of these options with their default values, and provides a brief description of each. We describe the most important ones in more detail in (**TODO**) `run_hdfgclean.ipynb` (see **TODO**:link the "Using `HDFGClean` section below).

> **TODO** : explain that `HDFGClean` inherits from `HDSims`, `FGClean`, and `HDFGCleanMaps` - has same methods and accepts same arguments

There are two ways to _initialize_ the `HDFGClean` class. The standard way is to pass arguments directly to the initialization method of the `HDFGClean` class, e.g.,

```
from hdfgclean import hdfgclean
output_dir = '/path/to/output'
hd_sims_dir = '/path/to/myHDsims'
hdfgcleanlib = hdfgclean.HDFGClean(output_dir, hd_sims_dir)
```

The second way is by saving the arguments in a configuration `.yaml` file (refer to `hdfgclean_defaults.yaml` for an example) by calling the `HDFGClean.save_config` class method. You can then use this file to initialize `HDFGClean` by calling the the `HDFGClean.from_config` class method, e.g.

```
from hdfgclean import hdfgclean
output_dir = '/path/to/output'
hd_sims_dir = '/path/to/myHDsims'
config_file = '/path/to/my_hdfgclean_config.yaml'
hdfgclean.HDFGClean.save_config(config_file, hd_sims_dir, output_dir=output_dir)
hdfgcleanlib = hdfgclean.HDFGClean.from_config(config_file)
```


#### Using `HDFGClean`

**Running the FG cleaning**:

The foreground cleaning can be run using the `run_hdfgclean` method of the `HDFGClean` class: e.g., once you have initialized the class as demonstrated above, you would just call `hdfgcleanlib.run_hdfgclean()` to run the full FG cleaning procedure. By default, after the maps have been FG-cleaned, the `run_hdfgclean` method will also:
- Match the catalogs of detected point sources (CIB + radio) or clusters to the catalogs of all true point sources or clusters in the maps
- Generate masks for any significant residual point sources or clusters remaining in the 90 and 148 GHz maps after FG cleaning
- Take the power spectrum of the maps (or any combination of their individual components) before or after FG cleaning, with or without applying a mask.
- Generate the relevant plots of MacInnis et. al. (2026)

We provide a python script, `run_hdfgclean.py`, which you can use to run the FG cleaning. You must pass the path to your `config_file` which will be used to initialize `HDFGClean`. By default, e.g. if you run the command

```
mpirun -np <N> python run_hdfgclean.py /path/to/my_hdfgclean_config.yaml
```

(where you would replace `<N>` with the number of MPI processes to use) the FG cleaning will be run on the maps, and the 90 and 148 GHz masks will be saved. To run all of the listed steps above (**note**: we don't recommend doing this; see the **TODO**:link "Using MPI" section above), you would pass

```
mpirun -np <N> python run_hdfgclean.py /path/to/my_hdfgclean_config.yaml --match --spectra --plots
```

Note that we also provide a `--test` option that can also be passed. In this case, `HDFGClean` will be initialized using your configuration file, and a message will be printed out telling you that the initialization was successful. A second message will be printed out telling you how many smaller patches your maps will be divided into, so you know how many MPI processes to use.

We also provide (**TODO**: add this) a python notebook, `run_hdfgclean.ipynb`, which will save your configuration `.yaml` file with your `output_dir`, `hd_sims_dir`, and any additional arguments you provide, and then print out the command(s) to pass this file to the `run_hdfgclean.py` python script.


**After running FG cleaning**:

> **TODO** : *briefly* mention methods used to load (or generate/calculate/etc) in catalogs / maps / power spectra / etc


### Using `FGClean` with any map(s)

**TODO**


---


## Reproducing the results of MacInnis et. al. (2026)

**TODO** 

(in the `reproduce_10x10` directory)


---


## Examples

**TODO** 

- 2x2 hdsims
- example for using `FGClean`: either use 10x10 or 2x2 hd sims, but treat them as "general" maps to show them how to use other sims - then can show we get same results

