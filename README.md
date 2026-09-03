# Foreground cleaning for CMB-HD

This repository contains the extragalactic foreground (FG) cleaning procedure presented in [MacInnis, Ange, Sehgal, Kable, and Blackstad (2026)](https://arxiv.org/abs/XXXX.XXXXX) (**TODO:LINK2PAPER**) to remove the thermal SZ (tSZ), cosmic infrared background (CIB), and radio galaxies from ultrahigh-resolution CMB temperature maps. Please cite that work if you use this code.

The `hdfgclean` code we provide here can be used to: (**TODO** : complete sentences)
- detect, measure, and remove CIB and radio galaxies from the maps, taking advantage of the frequency-dependence of the sources;
- detect, measure, and remove tSZ clusters from the maps 
- generate masks for any residual |SNR| > 5 sources or clusters remaining in the maps
- if using hdsims: methods to (1) take power spectra before and after FG cleaning ; (2) to match detected sources/clusters to true catalogs
 
We also provide an example to reproduce the results presented in [**TODO:LINK2PAPER**](https://arxiv.org/abs/XXXX.XXXXX), and instructions for running `hdfgclean` on different maps (**TODO**: hdsims or not - but must be CAR maps).

---

## (**TODO**) Overview of `hdfgclean`


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

If you would like to reproduce the parameter forecasts of [**TODO:LINK2PAPER**](https://arxiv.org/abs/XXXX.XXXXX), you will also need to install [hdfisher](https://github.com/CMB-HD/hdfisher) and [getdist](https://getdist.readthedocs.io).


### **TODO** : instructions to get hd sims?

- download 10x10 ra=6, dec=6 from LAMBDA
- otherwise refer to hdsims repo
- **mention that they should always generate hd sims separately, before running fg cleaning with mpi**


#### **TODO** : instructions for noise maps

- explain `generate_noise_sims.py` (currently it's kind of explained in `reproduce_10x10.ipynb`)


---

## (TODO) Usage

**TODO** : explain the files (`run_hdfgclean.ipynb`, `run_hdfgclean.py`, `hdfgclean_defaults.yaml`, etc.)


### (TODO) Reproducing the results in MacInnis et. al.

- see `reproduce_10x10.ipynb` in the `reproduce_10x10` directory (TODO : explain more)

### (TODO) Examples

- for a 2x2 patch
- more general example (using `run_fgclean.py`?)

