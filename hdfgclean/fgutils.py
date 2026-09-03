from copy import deepcopy
from inspect import signature
import numpy as np
import pandas as pd
import camb
from hdsims import utils
from . import mpi


def dict_with_keys(idict, keys, copy=False):
    """Remove keys from a dictionary that are not in the given list of 
    keys.

    Parameters
    ----------
    idict : dict
        The dictionary.
    keys : list
        A list of dictionary keys to keep.
    copy : bool, default=False
        Whether to make a copy of the dictionary.

    Returns
    -------
    odict : dict
        The dictionary after removing any keys that are not in the list
        of `keys`.

    Notes
    -----
    The input `idict` dictionary does not need to contain each key in the
    given list of `keys`.
    """
    d = idict if (not copy) else deepcopy(idict)
    odict = {key: value for (key, value) in d.items() if key in keys}
    return odict


def dict_without_keys(idict, keys, copy=False):
    """Remove the given keys from a dictionary.

    Parameters
    ----------
    idict : dict
        The dictionary.
    keys : list
        A list of dictionary keys to remove from `idict`.
    copy : bool, default=False
        Whether to make a copy of the dictionary.

    Returns
    -------
    odict : dict
        The dictionary after removing any keys in the list of `keys`.
    """
    d = idict if (not copy) else deepcopy(idict)
    odict = {key: value for (key, value) in d.items() if key not in keys}
    return odict


def print_msg(msg, verbose=False, log=None, stacklevel=2):
    """Print out a message.

    Parameters
    ----------
    msg : str
        The message to print.
    verbose : bool, default=False
        Whether to print the message.
    log : logging.Logger or None, optional
        A `logging.Logger` instance to use when `verbose=True`. If `log`
        is passed, any messages will be passed to `log.info`.
        Otherwise, messages will be passed to the `print` function.
    stacklevel : int, default=2
        The number of stack frames skipped when computing the line number
        and function name for the logging event; see the `logging` module
        documentation for further details. Ignored if `log=None`.
    """
    if verbose:
        if mpi.size > 1:
            msg = f"[rank {mpi.rank:>2d}] {msg}"
        if log is not None:
            log.info(msg, stacklevel=stacklevel)
        else:
            print(msg)


def get_bin_centers(bin_edges, log=False):
    """Calculate the bin centers from a set of bin edges.

    Parameters
    ----------
    bin_edges : array_like of int or array_like of float
        The bin edges. The first element gives the lower edge of the
        first bin, and the remaining elements are the upper edge of each
        bin.
    log : bool, default=False
        Whether to find the center of the bins in linear or log space.

    Returns
    -------
    bin_ctrs : array_like of float
        The bin centers.
    """
    if log:
        log_edges = np.log10(bin_edges)
        log_ctrs = (log_edges[1:] + log_edges[:-1]) / 2
        bin_ctrs = 10**log_ctrs
    else:
        bin_ctrs = (bin_edges[1:] + bin_edges[:-1]) / 2
    return bin_ctrs


def make_uniform_bins(min_edge, max_edge, num_bins, log=False):
    """Calculate equally-spaced bin edges and bin centers.

    Parameters
    ----------
    min_edge, max_edge : int or float
        The minimum and maximum bin edges, respectively.
    num_bins : int
        The number of bins to calculate.
    log : bool, default=False
        If `log=False` (default), calculate bins that each have the same
        width in linear space; otherwise, if `log=True`, calculate bins
        that have uniform width in log space.

    Returns
    -------
    bin_edges : array_like of int
        An array of bin edges, with length `num_bins + 1`. The first
        element is the lower edge of the first bin; the remaining
        elements are the upper edge of each bin.
    bin_ctrs : array_like of float
        An array of bin centers, with length `num_bins`.
    """
    if log:
        bin_edges = np.logspace(np.log10(min_edge), np.log10(max_edge), num_bins+1)
    else:
        bin_edges = np.linspace(min_edge, max_edge, num_bins+1)
    bin_ctrs = get_bin_centers(bin_edges, log=log)
    return bin_edges, bin_ctrs


def angular_radius(z, physical_radius, pars, results=None):
    """Calculate the angular size on the sky of an object located at a
    given redshift with a given physical size.

    Parameters
    ----------
    z : float or array_like of float
        The redshift(s) of the object(s).
    physical_radius : float or array_like of float
        The physical (not comoving) size(s) of the object(s), in units
        of Mpc.
    pars : camb.model.CAMBparams
        A CAMB parameters instance to use for the calculation.
    results : camb.results.CAMBdata or None, optional
        A CAMB results instance to use for the calculation. If not
        provided, the CAMB results will be calculated by calling
        `camb.get_background(pars)`.

    Returns
    -------
    float or array_like of float
        The angular size(s) of the object(s), in units of arcminutes.
    """
    if results is None:
        results = camb.get_background(pars)
    ang_diam_dist = results.angular_diameter_distance(z)
    return utils.rad2arcmin(physical_radius/ang_diam_dist)


def beam_fwhm_to_solid_angle(beam_fwhm):
    """Calculate the solid angle for a Gaussian beam profile.

    Parameters
    ----------
    beam_fwhm : float
        The full-width at half-maximum of the Gaussian beam profile,
        in units of arcminutes.

    Returns
    -------
    beam_solid_angle : float
        The beam solid angle, in units of steradians.
    """
    beam_sigma = utils.fwhm2sigma(beam_fwhm)
    beam_solid_angle = 2 * np.pi * utils.arcmin2rad(beam_sigma)**2
    return beam_solid_angle


def beam_solid_angle_to_fwhm(beam_solid_angle):
    """Calculate the full-width at half-maximum for a Gaussian beam
    profile.

    Parameters
    ----------
    beam_solid_angle : float
        The beam solid angle, in units of steradians.

    Returns
    -------
    beam_fwhm : float
        The full-width at half-maximum of the Gaussian beam profile,
        in units of arcminutes.
    """
    beam_sigma = np.sqrt(beam_solid_angle / (2 * np.pi))
    beam_fwhm = utils.sigma2fwhm(utils.rad2arcmin(beam_sigma))
    return beam_fwhm


def extrapolate_flux(flux, from_freq, to_freq, index):
    """Calculate the flux at one frequency from the flux at a different
    frequency and the spectral index between them.

    Parameters
    ----------
    flux : float or array_like of float
        The flux measured at the `from_freq` frequency.
    from_freq : int or float
        The frequency at which the `flux` was measured.
    to_freq : int or float
        The new frequency extrapolate the flux to.
    index : float
        The spectral index between the `from_freq` and `to_freq`.

    Returns
    -------
    float or array_like of float
        The flux at the `to_freq` frequency.
    """
    return flux * (to_freq / from_freq)**index


def get_spectral_index(flux1, flux2, freq1, freq2):
    """Calculate the spectral index between two frequencies given the flux
    at each frequency.

    Parameters
    ----------
    flux1, flux2 : float or array_like of float
        The flux(es) at `freq1` and `freq2`, respectively.
    freq1, freq2 : int or float
        The two frequencies.

    Returns
    -------
    float or array_like of float
        The value(s) of the spectral index for each pair of flux
        measurements.
    """
    return np.log(flux1/flux2) / np.log(freq1/freq2)


def calc_avg_spectral_index(multifreq_catalog, freq1, freq2, remove_outliers=True,
                            niter_to_remove=10, nsigma_to_remove=3, verbose=False, log=None):
    """Calculate the mean spectral index between two frequencies from flux
    measurements at each frequency.

    Parameters
    ----------
    multifreq_catalog : pandas.DataFrame
        A catalog where each row is a flux measurement, with columns for
        the flux at each frequency. The columns should be named
        `f'fluxmJy_{freq1}GHz'` and `f'fluxmJy_{freq2}GHz'`.
    freq1, freq2 : int or float
        The two frequencies.
    remove_outliers : bool, default=True
        Whether to iteratively re-calculate the mean spectral index after
        removing any rows with an index that differs from the mean by more
        than `nsigma_to_remove` standard deviations.

    Returns
    -------
    index_mean, index_std : float
        The mean spectral index and the standard deviation.
    catalog_for_index : pandas.DataFrame
        A copy of the input `multifreq_catalog`, with an added column
        (named `'index'`) containing the spectral index for each row.
        If `remove_outliers=True`, only contains the rows used for the
        calculation after iteratively removing outliers.

    Other Parameters
    ----------------
    niter_to_remove : int, default=10
        The number of iterations to use when removing outliers. Ignored
        if `remove_outliers=False`.
    nsigma_to_remove : int or float, default=3
        The number of standard deviations used when removing outliers.
        Ignored if `remove_outliers=False`.
    verbose : bool, default=False
        Whether to print out the final spectral index mean and standard
        deviation, and the values after each iteration if
        `remove_outliers=True`.
    log : logging.Logger or None, optional
        A `logging.Logger` instance to use when `verbose=True`. If `log`
        is passed, any messages will be passed to `log.info`.
        Otherwise, messages will be passed to the `print` function.

    See Also
    --------
    get_spectral_index
    get_avg_spectral_index
    """
    catalog_for_index = multifreq_catalog.copy()
    catalog_for_index['index'] = get_spectral_index(catalog_for_index[f'fluxmJy_{freq1}GHz'].values,
                                                    catalog_for_index[f'fluxmJy_{freq2}GHz'].values, freq1, freq2)
    if remove_outliers:
        for i in range(niter_to_remove):
            index_mean = np.mean(catalog_for_index['index'].values)
            index_std = np.std(catalog_for_index['index'].values)
            print_msg((f"iter {i+1} for index: mean = {index_mean:10.6f}, standard deviation = {index_std:10.6f}"
                       f" (calculated from {len(catalog_for_index)} sources)"), verbose=verbose, log=log)
            imin = index_mean - nsigma_to_remove * index_std
            imax = index_mean + nsigma_to_remove * index_std
            catalog_for_index = catalog_for_index[catalog_for_index['index'].between(imin, imax)]
        index_mean = np.mean(catalog_for_index['index'].values)
        index_std = np.std(catalog_for_index['index'].values)
    else:
        index_mean = np.mean(catalog_for_index['index'].values)
        index_std = np.std(catalog_for_index['index'].values)
    print_msg((f"spectral index between {freq1} & {freq2} GHz: "
               f"mean = {index_mean:.2f}, standard deviation = {index_std:.2f} "), verbose=verbose, log=log)
    return index_mean, index_std, catalog_for_index


def get_avg_spectral_index(fluxes1, fluxes2, freq1, freq2, remove_outliers=True,
                           niter_to_remove=10, nsigma_to_remove=3, verbose=False, log=None):
    """Calculate the mean spectral index between two frequencies from flux
    measurements at each frequency.

    Parameters
    ----------
    fluxes1, fluxes2 : array_like of float
        The fluxes at `freq1` and `freq2`, respectively.
    freq1, freq2 : int or float
        The two frequencies.
    remove_outliers : bool, default=True
        Whether to iteratively re-calculate the mean spectral index after
        removing any rows with an index that differs from the mean by more
        than `nsigma_to_remove` standard deviations.

    Returns
    -------
    index_mean, index_std : float
        The mean spectral index and the standard deviation.

    Other Parameters
    ----------------
    niter_to_remove : int, default=10
        The number of iterations to use when removing outliers. Ignored
        if `remove_outliers=False`.
    nsigma_to_remove : int or float, default=3
        The number of standard deviations used when removing outliers.
        Ignored if `remove_outliers=False`.
    verbose : bool, default=False
        Whether to print out the final spectral index mean and standard
        deviation, and the values after each iteration if
        `remove_outliers=True`.
    log : logging.Logger or None, optional
        A `logging.Logger` instance to use when `verbose=True`. If `log`
        is passed, any messages will be passed to `log.info`.
        Otherwise, messages will be passed to the `print` function.

    See Also
    --------
    get_spectral_index
    calc_avg_spectral_index
    """
    catalog_for_index = pd.DataFrame({f'fluxmJy_{freq1}GHz': fluxes1, f'fluxmJy_{freq2}GHz': fluxes2})
    index_mean, index_std, _ = calc_avg_spectral_index(catalog_for_index, freq1, freq2,
                                                       remove_outliers=remove_outliers,
                                                       niter_to_remove=niter_to_remove,
                                                       nsigma_to_remove=nsigma_to_remove,
                                                       verbose=verbose, log=log)
    return index_mean, index_std


def f_tSZ(freq, TCMB=2.7255e6):
    """Calculate the tSZ frequency dependence.

    Parameters
    ----------
    freq : int or float
        The frequency, in units of GHz.
    TCMB : float, default=2.7255e6
        The mean CMB temperature, in units of uK.

    Returns
    -------
    float
        The tSZ frequency dependence, calculated at the given frequency.
    """
    h = 6.62607015e-27 # erg * s
    k_B = 1.380649e-16 # erg / K
    x = 1e9 * freq * h / (k_B * TCMB * 1e-6)
    return x * np.cosh(x/2) / np.sinh(x/2) - 4


def y_to_uK(x, freq, TCMB=2.7255e6):
    """Convert (a) value(s) of Compton y-parameter(s) to CMB temperature
    units at a given frequency.

    Parameters
    ----------
    x : float or array_like of float
        The value(s) of the Compton y-parameter.
    freq : int or float
        The frequency, in GHz.
    TCMB : float, default=2.7255e6
        The mean CMB temperature, in units of uK.

    Returns
    -------
    float or array_like of float
        The value(s) in units of uK.
    """
    return x * f_tSZ(freq, TCMB=TCMB) * TCMB


def uK_to_y(x, freq, TCMB=2.7255e6):
    """Convert (a) value(s) in CMB temperature units to the value(s) of
    the Compton y-parameter at a given frequency.

    Parameters
    ----------
    x : float or array_like of float
        The value(s) in units of uL.
    freq : int or float
        The frequency, in GHz.
    TCMB : float, default=2.7255e6
        The mean CMB temperature, in units of uK.

    Returns
    -------
    float or array_like of float
        The value(s) of the Compton y-parameter.
    """
    return (x / TCMB) / f_tSZ(freq, TCMB=TCMB)


def _arg_info(obj):
    """Get information about the arguments accepted by a callable object.

    Parameters
    ----------
    obj
        A callable object.

    Returns
    -------
    required_args, required_kwargs : list of str
        Lists of positional-only and keyword-only argument names,
        respectively, that can be passed to the callable object.
    optional_kwargs : dict
        A dictionary with optional keyword argument names (`str`) and
        their default values.
    allows_other_args, allows_other_kwargs : bool
        Whether the callable object allows additional arguments (by
        passing `*args`) and keyword arguments (by passing `**kwargs`),
        respectively.
    """
    sig = signature(obj)
    required_args = []
    required_kwargs = []
    optional_kwargs = {}
    allows_other_args = False
    allows_other_kwargs = False
    for param_name, param in sig.parameters.items():
        if param.default is param.empty:
            if param.kind is param.VAR_POSITIONAL:
                allows_other_args = True
            elif param.kind is param.VAR_KEYWORD:
                allows_other_kwargs = True
            elif param.kind is param.KEYWORD_ONLY:
                required_kwargs.append(param_name)
            else:
                required_args.append(param_name)
        else:
            optional_kwargs[param_name] = param.default
    return required_args, required_kwargs, optional_kwargs, allows_other_args, allows_other_kwargs


def _get_all_kwargs(cls):
    """Get a dictionary of all keyword arguments with their default values
    that can be passed to the given class and any of its parent classes.

    Parameters
    ----------
    cls
        A python class.

    Returns
    -------
    all_kwargs : dict
        A dictonary of all keyword argument names and default values.

    Notes
    -----
    This assumes that if the class is a derived class and it accepts
    `**kwargs`, it passes any `kwargs` to its parent class(es). It
    can only return keyword arguments that have been named and have some
    default value.
    """
    _, _, all_kwargs, _, allows_other_kwargs = _arg_info(cls)
    parents = cls.__bases__
    while allows_other_kwargs and (len(parents) > 0):
        new_parents = []
        kwargs_allowed = []
        for parent in parents:
            _, _, kwargs, _, allows_kwargs = _arg_info(parent)
            all_kwargs = {**kwargs, **all_kwargs}
            new_parents = [*new_parents, *parent.__bases__]
            kwargs_allowed.append(allows_kwargs)
        parents = new_parents
        allows_other_kwargs = any(kwargs_allowed)
    return all_kwargs

