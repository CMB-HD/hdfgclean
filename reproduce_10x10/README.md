# Reproducing the results of MacInnis et. al. (2026)

The notebook `reproduce_10x10.ipynb` will provide instructions to run the full foreground cleaning procedure described in MacInnis et. al. (2026) and reproduce Figures 1 - 2 and 6 - 11 in that work. It can also be used to reproduce the parameter forecasts (Figures 12 - 13 and Tables 4 - 6) of MacInnis et. al. (2026); this is not done by default because it also requires [hdfisher](https://github.com/CMB-HD/hdfisher) and [getdist](https://getdist.readthedocs.io), in addition to the requirements of the `hdfgclean` package.

Along with the `reproduce_10x10.ipynb` notebook, we provide the following files here:

- The `reproduce_10x10.py` file is used to run the foreground cleaning.
- The `calculate_fisher.py` file can be used to calculate the Fisher matrices used for the parameter forecasts.

We explain how to use each file (if necessary) within the `reproduce_10x10.ipynb` notebook.

