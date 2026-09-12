"""General functions and classes for maps."""

import numpy as np
import pandas as pd
from pixell import enmap, pointsrcs
from hdsims import utils, fgcatalogs, maps
from . import fgclean_info as fgi, fgutils


def trim_map_geometry(shape, wcs, width_to_remove=0):
    """Calculate the geometry of the inner region of a map.

    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    width_to_remove : int or float, default=0
        The width (in units of degrees) along the outer map edges to
        be removed.

    Returns
    -------
    trimmed_shape : tuple of int
        The shape of the inner region of the map.
    trimmed_wcs : astropy.wcs.wcs.WCS
        The astropy World Coordinate System instance for the inner region
        of the map.
    """
    pixel_size = maps.get_map_resolution(shape, wcs)
    if width_to_remove > 0:
        # get geometry of only the inner, un-apodized region:
        ra_ctr, dec_ctr, width, height = maps.get_map_ctr_extent(shape, wcs)
        trimmed_width = width - 2 * width_to_remove
        trimmed_height = height - 2 * width_to_remove
        trimmed_shape, trimmed_wcs = maps.get_shape_wcs(pixel_size, ra_ctr, dec_ctr, trimmed_width,
                                                        height=trimmed_height)
    else:
        trimmed_shape = shape
        trimmed_wcs = wcs
    return trimmed_shape, trimmed_wcs


def dist2npix(dist, shape, wcs, units='degrees'):
    """Calculate the approximate number of map pixels corresponding to a
    given angular distance.

    Parameters
    ----------
    dist : int or float
        The angular distance.
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    units : {'degrees', 'radians', 'arcmin', 'arcsec'}
        The units of `dist`.

    Returns
    -------
    npix : int
        The approximate number of pixels in the map corresponding to the
        given distance.
    """
    pixel_size = maps.get_map_resolution(shape, wcs)
    if 'deg' in units.lower():
        dist = utils.deg2arcmin(dist)
    elif 'rad' in units.lower():
        dist = utils.rad2arcmin(dist)
    elif 'sec' in units.lower():
        dist /= 60
    elif 'arcmin' not in units.lower():
        raise ValueError(f"`{units = }`. The allowed units are `'degrees'`, `'radians'`, `'arcmin'`, or `'arcsec'`.")
    npix = round(dist / pixel_size)
    return npix


def npix2dist(npix, shape, wcs, units='degrees'):
    """Calculate an approximate angular distance corresponding to a given
    number of map pixels.

    Parameters
    ----------
    npix : int
        The number of map pixels.
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    units : {'degrees', 'radians', 'arcmin', 'arcsec'}
        The units of the returned angular distance.

    Returns
    -------
    dist : float
        The approximate angular distance corresponding to the given
        number of pixels in the map.
    """
    pixel_size = maps.get_map_resolution(shape, wcs)
    dist = pixel_size * npix
    if 'deg' in units.lower():
        dist = utils.arcmin2deg(dist)
    elif 'rad' in units.lower():
        dist = utils.arcmin2rad(dist)
    elif 'sec' in units.lower():
        dist *= 60
    elif 'arcmin' not in unit:
        raise ValueError(f"`{units = }`. The allowed units are `'degrees'`, `'radians'`, `'arcmin'`, or `'arcsec'`.")
    return dist


def make_binary_mask(ras, decs, radius, shape, wcs):
    """Make a binary mask with holes at the given coordinate positions 
    using a fixed hole radius.

    Parameters
    ----------
    ras, decs : float or array_like of float
        The R.A. & dec. coordinate pair(s), in degrees. Each should be a
        one-dimensional array, such that the `i`th coordinate pair is
        `ras[i]`, `decs[i]`, and both should have the same length.
    radius : float
        The hole radius, in arcminutes.
    shape : tuple of int
        The shape `(Ny, Nx)` of the mask, where `Ny` and `Nx` are the
        number of pixels along the dec. and R.A. directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the mask.

    Returns
    -------
    mask : pixell.enmap.ndmap
        The mask, with zeros inside the holes and ones everywhere else.
    """
    mask_radius = utils.arcmin2rad(radius)
    coords = np.deg2rad(np.array([decs, ras]))
    bool_binary_mask = enmap.distance_from(shape, wcs, coords, rmax=mask_radius) >= mask_radius
    binary_mask = enmap.ones(shape, wcs, dtype=int) * bool_binary_mask
    return binary_mask


def get_beam_template(shape, wcs, beam_fwhm, ra, dec, normalize=True):
    """Make a map of a Gaussian beam profile at the given position.

    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the map, where `Ny` and `Nx` are the
        number of pixels along the declination (dec.) and right ascension
        (R.A.) directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    beam_fwhm : float
        The full-width at half-maximum of the Gaussian beam, in
        arcminutes.
    ra, dec : float
        The R.A. and dec. of the center of the beam profile, in degrees.
    normalize : bool, default=True
        If `True`, normalize the values such that the maximum value in
        the map is one.

    Returns
    -------
    beam_template : pixell.enmap.ndmap
        The map of the beam profile.
    """
    # make a map of zeros, and set the pixel at the position (ra, dec) to one
    beam_template = enmap.zeros(shape, wcs)
    x, y = maps.coord2pix(ra, dec, shape, wcs)
    beam_template[y,x] = 1
    beam_template = maps.convolve_sim_with_beam(beam_template, beam_fwhm)
    if normalize: # set max = 1
        beam_template /= np.max(beam_template.data)
    return beam_template


def _multifreq_maps_to_mapdict(imaps, freqs, copy=True):
    """Place each individual map of a `pixell.enmap.ndmap` of shape
    `(nfreq, ny, nx)` into a dictionary of `(Ny, Nx)` maps, with a key
    for each frequency.

    Parameters
    ----------
    imaps : pixell.enmap.ndmap
        A single map with shape `(nfreq, ny, nx)`, where `nfreq` is the
        number of map frequencies, and `ny` and `nx` are the number of
        pixel rows (declination direction) and columns (right ascention)
        in each map, respectively.
    freqs : list of int or list of float
        The map frequencies, in the correct order such that `imaps[i]` is
        the map at frequency `freqs[i]` (for integer `i` between `0` and
        `nfreq - 1`).
    copy : bool, default=True
        Whether to make copies of the maps.

    Returns
    -------
    omap : dict of pixell.enmap.ndmap
        A dictionary of maps, each of shape `(Ny, Nx)`, with the keys
        given by the values in `freqs`.

    See Also
    --------
    _multifreq_mapdict_to_maps : The inverse function.
    """
    omaps = {}
    for i, freq in enumerate(freqs):
        omaps[freq] = imaps[i] if (not copy) else imaps[i].copy()
    return omaps


def _multifreq_mapdict_to_maps(imaps, freqs=None, copy=True):
    """Place the individual maps, each with shape `(Ny, Nx)`, at different
    frequencies into a single `pixell.enmap.ndmap` of shape
    `(nfreq, ny, nx)`.

    Parameters
    ----------
    imaps : dict of pixell.enmap.ndmap
        A dictionary of maps, each of shape `(Ny, Nx)`, with a key for
        each map frequency.
    freqs : list of int or list of float or None, optional
        The map frequencies. If not provided, the keys of the `imaps`
        dictionary will be used, sorted from lowest to highest frequency,
        such that `imaps[0]` and `imaps[-1]` are the maps at the lowest
        and highest frequency, respectively. If `freqs` is provided, that
        order will be used instead, and only those frequencies will be
        used.
    copy : bool, default=True
        Whether to make copies of the maps.

    Returns
    -------
    omap : pixell.enmap.ndmap
        A single map with shape `(nfreq, ny, nx)`, where `nfreq` is the
        number of map frequencies, and `ny` and `nx` are the number of
        pixel rows (declination direction) and columns (right ascention)
        in each map, respectively.

    See Also
    --------
    _multifreq_maps_to_mapdict : The inverse function.
    """
    if freqs is None:
        freqs = sorted(list(imaps.keys()))
    nfreq = len(freqs)
    shape = imaps[freqs[0]].shape
    wcs = imaps[freqs[0]].wcs
    omaps = enmap.zeros((nfreq, *shape), wcs)
    for i, freq in enumerate(freqs):
        if copy:
            omaps[i] = imaps[freq].copy()
        else:
            omaps[i] = imaps[freq]
    return omaps


def convert_tsz_map_units(imap, imap_units=None, omap_units=None, freq=None):
    """Convert between Compton-y units and CMB temperature units (uK).

    Parameters
    ----------
    imap : array_like of float
        The values to convert from `imap_units` to `omap_units`.
    imap_units, omap_units : str or None, optional
        The units of the `imap` and the units to convert to,
        respectively. If the two values are equal or if `omap_units` is
        `None`, nothing is done and the input `imap` is returned.
        Otherwise, each must be either `'y'` for Compton-y units or
        `'uK'` for CMB temperature units, and you must also pass the map
        frequency to `freq`.
    freq : float or None, optional
        The map frequency (in GHz). Must be provided if both `imap_units`
        and `omap_units` are provided (with different values for each).

    Returns
    -------
    omap : array_like of float
        The values converted to the `omap_units`.
    """
    if omap_units in [None, imap_units]:
        return imap
    else:
        allowed_units = ['y', 'uK']
        if (imap_units not in allowed_units) or (omap_units not in allowed_units):
            raise ValueError(f"Cannot convert from units of `'{imap_units}'` to `'{omap_units}'`. "
                             f"Each must be one of {allowed_units}.")
        elif freq is None:
            raise ValueError(f"You must provide a `freq` (map frequency, in GHz) to convert the map "
                             f"from units of `'{imap_units}'` to `'{omap_units}'`.")
        if imap_units == 'y':
            omap = fgutils.y_to_uK(imap, freq)
        else:
            omap = fgutils.uK_to_y(imap, freq)
        return omap


def tsz_sim_from_signal_map(imap, imap_units=None, omap_units=None, freq=None,
                            apply_apod=False,apod_width=None, apod_window=None,
                            convolve_pixwin=False, convolve_beam=False, beam_fwhm=None):
    """Given a map of the tSZ signal, convert to the desired map units and
    (optionally) apodize it and convolve the instrumental beam and pixel
    window function.

    Parameters
    ----------
    imap : pixell.enmap.ndmap
        A tSZ signal map.
    imap_units, omap_units : str or None, optional
        The units of the `imap` and the units to convert to,
        respectively. If the two values are equal or if `omap_units` is
        `None`, no unit conversion is done. Otherwise, each must be
        either `'y'` for Compton-y units or  `'uK'` for CMB temperature
        units, and you must also pass the map frequency to `freq`.
    freq : float or None, optional
        The map frequency (in GHz). Must be provided if both `imap_units`
        and `omap_units` are provided (with different values for each).
    apply_apod : bool, default=False
        Whether to apodize the map.
    apod_width : float or None, optional
        The apodization width (in degrees) used to apodize the map. Not
        used if `apply_apod=False`, or if `apply_apod=True` but an
        `apod_window` was passed.
    apod_window : pixell.enmap.ndmap or None, optional
        An apodization window to use if `apply_apod=True`.
    convolve_pixwin : bool, default=False
        Whether to convolve the map with the pixel window function.
    convolve_beam : bool, default=False
        Whether to convolve the map with a Gaussian beam. If `True`, you
        must also provide the `beam_fwhm`.
    beam_fwhm : float or None, optional
        The beam full-width at half-maximum (in arcminutes) of the
        Gaussian beam profile. Only used if `convolve_beam=True`.

    Returns
    -------
    omap : pixell.enmap.ndmap
        The requested map.

    Notes
    -----
    In general, maps must be apodized before convolving with the beam or
    pixel window, so you should pass `apply_apod=True` if the `imap` has
    not already been apodized.
    """
    # check if we can do the things requested:
    if apply_apod and (apod_window is None):
        if apod_width is None:
            raise ValueError(f"You passed `{apply_apod = }`, `{convolve_pixwin = }`, `{convolve_beam = }`."
                             " You must pass `apod_width` (an apodization width, in degrees) or `apod_window`"
                             " (map of the apodization window) in order apodize the map, and to convolve"
                             " the beam or pixel window function.")
        apod_window = maps.make_apod_window(shape, wcs, apod_width)
    if convolve_beam and (beam_fwhm is None):
        raise ValueError(f"`{convolve_beam = }` and `{beam_fwhm = }`. You must pass `beam_fwhm` "
                         "(the beam full-width at half-maximum, in units of arcminutes) to convolve the beam.")
    # if so, do them:
    omap = convert_tsz_map_units(imap, imap_units=imap_units, omap_units=omap_units, freq=freq)
    if apply_apod:
        omap *= apod_window
    if convolve_pixwin:
        omap = enmap.apply_window(omap)
    if convolve_beam:
        omap = maps.convolve_sim_with_beam(omap, beam_fwhm)
    return omap


def make_map_of_objs_with_profile(shape, wcs, ras, decs, amplitudes, radial_prof_func, rmax):
    """Make a map of objects with the given radial profile at the given
    positions with the given amplitudes.

    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    ras, decs : float or array_like of float
        The R.A. & dec. coordinate pair(s), in degrees. Each should be a
        one-dimensional array, such that the `i`th coordinate pair is
        `ras[i]`, `decs[i]`, and both should have the same length.
    amplitudes : float or array_like of float
        The amplitude to use for the profile at each position. Must
        either be a single value or an array of values with the same
        length as `ras` and `decs`.
    radial_profile_func : function
        A function for the radial profile that takes in a single argument
        for the angular distance (in arcminutes) from the origin.
    rmax : float
        A maximum angular distance (in arcminutes) from the origin,
        beyond which the radial profile is set to zero.

    Returns
    -------
    omap : pixell.enmap.ndmap
        A map of objects with the given `shape` and `wcs`.
    """
    coords = np.array([np.deg2rad(np.atleast_1d(decs)), np.deg2rad(np.atleast_1d(ras))])
    amps = np.atleast_1d(amplitudes)
    # get the radial profile
    r = np.logspace(-5, np.log10(rmax)) # in arcmin
    rprof = (utils.arcmin2rad(r), radial_prof_func(r))
    # make the map
    omap = enmap.zeros(shape, wcs)
    omap += pointsrcs.sim_objects(shape, wcs, coords, amps, rprof, vmin=1e-12, 
                                  rmax=utils.arcmin2rad(rmax), pixwin=False)
    return omap


def get_profile_template_maps(freqs, shape, wcs, radial_prof_func, rmax,
                              convolve_pixwin=False, convolve_beam=False, beam_fwhms=None):
    """Make a set of maps at the given frequencies with the given radial
    profile placed at the map center with unit amplitude.

    Parameters
    ----------
    freqs : list of int or list of float
        A list of map frequencies.
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    radial_profile_func : function
        A function for the radial profile that takes in a single argument
        for the angular distance (in arcminutes) from the origin.
    rmax : float
        A maximum angular distance (in arcminutes) from the origin,
        beyond which the radial profile is set to zero.
    convolve_pixwin : bool, default=False
        Whether to convolve the maps with the pixel window function.
    convolve_beam : bool, default=False
        Whether to convolve the maps with a Gaussian beam. If `True`, you
        must also provide `beam_fwhms`.
    beam_fwhm : dict of float or None, optional
        A dictionary with a key for each frequency with the Gaussian beam
        full-width at half-maximum (in arcminutes) at that frequency.
        Only used if `convolve_beam=True`.

    Returns
    -------
    template_maps : pixell.enmap.ndmap
        A single map with shape `(nfreq, ny, nx)`, where `nfreq` is the
        number of map frequencies, and `ny` and `nx` are the number of
        pixel rows (declination direction) and columns (right ascention)
        in each map, respectively; `template_maps[i]` is the map of the
        profile at frequency `freqs[i]`.
    """
    if convolve_beam:
        # make sure we can convolve the beam
        beam_err_msg = ("To convolve the beam, you must pass a dictionary of `beam_fwhms` with a key "
                        "for each frequency in `freqs` containing the beam full-width at half-maximum "
                        "(in arcminutes) at that frequency.")
        if beam_fwhms is None:
            raise ValueError(f"`{convolve_beam = }`, `{beam_fwhms = }`. {beam_err_msg}")
        # make sure we have a FWHM for each freq
        missing_freqs = [freq for freq in freqs if (freq not in beam_fwhms)]
        if len(missing_freqs) > 0:
            raise ValueError(f"`{convolve_beam = }` and {freqs = }, but the `beam_fwhms` dictionary is missing "
                             f"keys for the following frequencies: {missing_freqs}. {beam_err_msg}")
    else:
        beam_fwhms = {freq: None for freq in freqs}
    # make a single map of the profile at the origin with unit amplitude:
    ra_ctr, dec_ctr, _, _ = maps.get_map_ctr_extent(shape, wcs)
    signal_profile_template = make_map_of_objs_with_profile(shape, wcs, ra_ctr, dec_ctr, 1, radial_prof_func, rmax)
    signal_profile_template /= np.max(signal_profile_template) # force amplitude = 1
    # return maps at each frequency
    nfreq = len(freqs)
    template_maps = enmap.zeros((nfreq, *shape[-2:]), wcs)
    for i, freq in enumerate(freqs):
        template_maps[i] = tsz_sim_from_signal_map(signal_profile_template.copy(), convolve_pixwin=convolve_pixwin,
                                                   convolve_beam=convolve_beam, beam_fwhm=beam_fwhms[freq])
    return template_maps


def make_cluster_ymap_from_catalog(shape, wcs, catalog, profiles, ra_col='RADeg', dec_col='decDeg',
                                   amplitude_col='y_c', profile_name_col='template'):
    """Generate a map of the tSZ signal in Compton-y units from a set of
    coordinate positions in a catalog and the radial profile to use at
    each position.

    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    catalog : pandas.DataFrame
        A catalog with a row for each cluster to place in the map, and
        columns for the cluster right ascention and declination (in
        degrees), the name of the radial cluster profile (corresponding
        to a key in the `profiles` dictionary), and the amplitude (i.e.,
        the value in the map at the cluster position) in Compton-y units.
    profiles : dict of dict
        A dictionary of radial profile functions to use for the cluster
        profiles. Each key should be a name (`str`) for a given profile,
        and `profiles[profile_name]` should be a dictionary with the
        following keys and values:
        - `'radial_prof_func'` : A callable `function` for the radial
          profile. The first argument must be the angular distance (in
          arcminutes) from the origin.
        - `'args'` : A list of any additional positional arguments to
          pass to the profile function.
        - `'kwargs'` : A dictionary of any additional keyword arguments
          to pass to the profile function.
        - `'rmax'` : A maximum angular distance (in arcminutes) from the
          origin, beyond which the radial profile is set to zero.

    Returns
    -------
    omap : pixell.enmap.ndmap
        The tSZ map in Compton-y units.

    Other Parameters
    ----------------
    ra_col : str, default='RADeg'
        The catalog column name for the right ascention.
    dec_col : str, default='decDeg'
        The catalog column name for the declination.
    amplitude_col : str, default='y_c'
        The catalog column name for the amplitudes.
    profile_name_col : str, default='template'
        The catalog column name for the cluster profile names.
    """
    omap = enmap.zeros(shape, wcs)
    profile_names = list(set(catalog[profile_name_col].values))
    for profile_name in profile_names:
        # make copy of catalog, keeping only rows for this profile:
        catalog_for_profile = catalog[catalog[profile_name_col].eq(profile_name)]
        ras = catalog_for_profile[ra_col].values
        decs = catalog_for_profile[dec_col].values
        amps = catalog_for_profile[amplitude_col].values
        # define radial profile function:
        profile_info = profiles[profile_name]
        if 'args' in profile_info:
            args = profile_info['args']
        else:
            args = []
        if 'kwargs' in profile_info:
            kwargs = profile_info['kwargs']
        else:
            kwargs = {}
        radial_prof_func = lambda r: profile_info['radial_prof_func'](r, *args, **kwargs)
        rmax = profile_info['rmax']
        # make map of objects w. this profile and add it to the output map:
        omap += make_map_of_objs_with_profile(shape, wcs, ras, decs, amps, radial_prof_func, rmax)
    return omap


def make_cluster_sim_from_catalog(shape, wcs, catalog, profiles, freq,
                                  apply_apod=False, apod_width=None, apod_window=None,
                                  convolve_pixwin=False, convolve_beam=False, beam_fwhm=None,
                                  ra_col='RADeg', dec_col='decDeg',
                                  amplitude_col='y_c', profile_name_col='template'):
    """Generate a map of the tSZ signal in CMB temperature (uK) units
    from a set of coordinate positions in a catalog and the radial
    profile to use at each position, then (optionally) apodize it and
    convolve the instrumental beam and pixel window function.

    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    catalog : pandas.DataFrame
        A catalog with a row for each cluster to place in the map, and
        columns for the cluster right ascention and declination (in
        degrees), the name of the radial cluster profile (corresponding
        to a key in the `profiles` dictionary), and the amplitude (i.e.,
        the value in the map at the cluster position) in Compton-y units.
    profiles : dict of dict
        A dictionary of radial profile functions to use for the cluster
        profiles. Each key should be a name (`str`) for a given profile,
        and `profiles[profile_name]` should be a dictionary with the
        following keys and values:
        - `'radial_prof_func'` : A callable `function` for the radial
          profile. The first argument must be the angular distance (in
          arcminutes) from the origin.
        - `'args'` : A list of any additional positional arguments to
          pass to the profile function.
        - `'kwargs'` : A dictionary of any additional keyword arguments
          to pass to the profile function.
        - `'rmax'` : A maximum angular distance (in arcminutes) from the
          origin, beyond which the radial profile is set to zero.
    freq : int or float
        The map frequency, in GHz.
    apply_apod : bool, default=False
        Whether to apodize the map.
    apod_width : float or None, optional
        The apodization width (in degrees) used to apodize the map. Not
        used if `apply_apod=False`, or if `apply_apod=True` but an
        `apod_window` was passed.
    apod_window : pixell.enmap.ndmap or None, optional
        An apodization window to use if `apply_apod=True`.
    convolve_pixwin : bool, default=False
        Whether to convolve the map with the pixel window function.
    convolve_beam : bool, default=False
        Whether to convolve the map with a Gaussian beam. If `True`, you
        must also provide the `beam_fwhm`.
    beam_fwhm : float or None, optional
        The beam full-width at half-maximum (in arcminutes) of the
        Gaussian beam profile. Only used if `convolve_beam=True`.

    Returns
    -------
    omap : pixell.enmap.ndmap
        The tSZ map.

    Other Parameters
    ----------------
    ra_col : str, default='RADeg'
        The catalog column name for the right ascention.
    dec_col : str, default='decDeg'
        The catalog column name for the declination.
    amplitude_col : str, default='y_c'
        The catalog column name for the amplitudes.
    profile_name_col : str, default='template'
        The catalog column name for the cluster profile names.

    See Also
    --------
    make_cluster_sims_from_catalog : Maps at multiple frequencies.
    """
    ymap = make_cluster_ymap_from_catalog(shape, wcs, catalog, profiles, ra_col=ra_col, dec_col=dec_col,
                                          amplitude_col=amplitude_col, profile_name_col=profile_name_col)
    omap = tsz_sim_from_signal_map(ymap, imap_units='y', omap_units='uK', freq=freq,
                                   apod_width=apod_width, apod_window=apod_window, apply_apod=apply_apod,
                                   convolve_pixwin=convolve_pixwin, convolve_beam=convolve_beam, beam_fwhm=beam_fwhm)
    return omap


def make_cluster_sims_from_catalog(shape, wcs, catalog, profiles, freqs,
                                   apply_apod=False, apod_width=None, apod_window=None,
                                   convolve_pixwin=False, convolve_beam=False, beam_fwhms=None,
                                   ra_col='RADeg', dec_col='decDeg',
                                   amplitude_col='y_c', profile_name_col='template'):
    """Generate maps of the tSZ signal in CMB temperature (uK) units at
    the given frequencies from a set of coordinate positions in a catalog
    and the radial profile to use at each position, then (optionally)
    apodize each map and convolve the instrumental beam and pixel window
    function.

    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    catalog : pandas.DataFrame
        A catalog with a row for each cluster to place in the map, and
        columns for the cluster right ascention and declination (in
        degrees), the name of the radial cluster profile (corresponding
        to a key in the `profiles` dictionary), and the amplitude (i.e.,
        the value in the map at the cluster position) in Compton-y units.
    profiles : dict of dict
        A dictionary of radial profile functions to use for the cluster
        profiles. Each key should be a name (`str`) for a given profile,
        and `profiles[profile_name]` should be a dictionary with the
        following keys and values:
        - `'radial_prof_func'` : A callable `function` for the radial
          profile. The first argument must be the angular distance (in
          arcminutes) from the origin.
        - `'args'` : A list of any additional positional arguments to
          pass to the profile function.
        - `'kwargs'` : A dictionary of any additional keyword arguments
          to pass to the profile function.
        - `'rmax'` : A maximum angular distance (in arcminutes) from the
          origin, beyond which the radial profile is set to zero.
    freqs : list of int or list of float
        The map frequencies, in GHz.
    apply_apod : bool, default=False
        Whether to apodize the map.
    apod_width : float or None, optional
        The apodization width (in degrees) used to apodize the map. Not
        used if `apply_apod=False`, or if `apply_apod=True` but an
        `apod_window` was passed.
    apod_window : pixell.enmap.ndmap or None, optional
        An apodization window to use if `apply_apod=True`.
    convolve_pixwin : bool, default=False
        Whether to convolve the map with the pixel window function.
    convolve_beam : bool, default=False
        Whether to convolve the map with a Gaussian beam. If `True`, you
        must also provide the `beam_fwhm`.
    beam_fwhms : dict of float or None, optional
        A dictionary with a key for each frequency and the beam
        full-width at half-maximum (in arcminutes) of the Gaussian beam
        profile at that frequency. Only used if `convolve_beam=True`.

    Returns
    -------
    omaps : dict of pixell.enmap.ndmap
        A dictionary with a key for each frequency containing the tSZ map
        at that frequency.

    Other Parameters
    ----------------
    ra_col : str, default='RADeg'
        The catalog column name for the right ascention.
    dec_col : str, default='decDeg'
        The catalog column name for the declination.
    amplitude_col : str, default='y_c'
        The catalog column name for the amplitudes.
    profile_name_col : str, default='template'
        The catalog column name for the cluster profile names.

    See Also
    --------
    make_cluster_sim_from_catalog : Map at a single frequency.
    """
    ymap = make_cluster_ymap_from_catalog(shape, wcs, catalog, profiles, ra_col=ra_col, dec_col=dec_col,
                                          amplitude_col=amplitude_col, profile_name_col=profile_name_col)
    beam_fwhms = {freq: None for freq in freqs} if (beam_fwhms is None) else beam_fwhms
    omaps = {}
    for freq in freqs:
        omaps[freq] = tsz_sim_from_signal_map(ymap.copy(), imap_units='y', omap_units='uK', freq=freq,
                                              apply_apod=apply_apod, apod_width=apod_width,
                                              apod_window=apod_window, convolve_pixwin=convolve_pixwin,
                                              convolve_beam=convolve_beam, beam_fwhm=beam_fwhms[freq])
    return omaps




def divide_map_area_into_patches(shape, wcs, map_apod_width=0,
                                 max_patch_size=fgi.max_patch_size,
                                 patch_apod_width=fgi.patch_apod_width):
    """Divide a map into a grid of smaller patches.
    
    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    map_apod_width : int or float, default=0
        The apodization width (in degrees) of the region along the map 
        edges over which the map has been apodized. If the map is not
        apodized, pass `map_apod_width=0` (the default).
    max_patch_size : int or float, default=3
        The maximum allowed width or height (in degrees) of the smaller
        patches, including any padding along the edges to account for
        the apodization width of the patch and of the full map. See the
        "Notes" section for further details.
    patch_apod_width : int or float, default=0.25
        The apodization width (in degrees) that will be used to apodize 
        the smaller patches.
    
    Returns
    -------
    patch_ra_ctrs, patch_dec_ctrs : array_like of float
        The R.A. and dec. coordinates (in degrees), respectively, of the
        centers of the grid cells; i.e., the center of the un-apodized,
        non-overlapping regions of the patches.
    patch_width, patch_height : float
        The width and height (in degrees) of the inner, non-overlapping
        regions of the smaller patches.
        
    Notes
    -----
    A map of width `W` and height `H` apodized over a width 
    `map_apod_width` along its edges has an un-apodized inner region of 
    width `w = W - 2 * map_apod_width` and height 
    `h = H - 2 * map_apod_width`. This un-apodized region is divided into
    a grid with `ny` rows and `nx` columns, so the width and height of 
    each cell are `patch_width = w / nx` and `patch_height = h / ny`, 
    respectively.
    
    Each grid cell corresponds to a smaller patch in the full map. Each 
    patch is extended beyond its grid cell by an amount 
    `patch_apod_width` on each side, so the un-apodized regions of the 
    patches do not overlap with each other.  If the `map_apod_width` is 
    larger than the `patch_apod_width`, then patches lying along the 
    outer border of the grid will be extended by an additional amount 
    (the difference `map_apod_width - patch_apod_width`) on its 
    outer-facing sides, in order to reach the edges of the full map.
    """
    apod_width = max([patch_apod_width, map_apod_width])
    # get min/max ra/dec in the map and its width and height:
    ra_min, ra_max, dec_min, dec_max = maps.get_map_corner_coords(shape, wcs)
    ra_ctr, dec_ctr, width, height = maps.get_map_ctr_extent(shape, wcs)

    unapod_width = width - 2 * apod_width
    # compare number of pixels (`int` instead of `float`)
    if dist2npix(width, shape, wcs) <= dist2npix(max_patch_size, shape, wcs):
        patch_width = unapod_width
        patch_ra_ctrs = np.array([ra_ctr])
    else:
        num_patches_ra = 1
        # divide the un-apodized width in to `num_patches_ra` 
        # non-overlapping patches of equal width:
        patch_width = unapod_width / num_patches_ra
        # each patch is extended by the `patch_apod_width` on each side, 
        # so the non-overlapping region (of width `patch_width`) remains 
        # un-apodized; if the `map_apod_width` is larger than the 
        # `patch_apod_width`, patches along the edge of the map need to 
        # be extended by an additional distance (given by 
        # `map_apod_width - patch_apod_width`) in order to reach the edges
        # of the full map.
        # calculate the maximum patch size for patches along the map edges, 
        # accounting for this extra padding:
        max_patch_width = patch_width + apod_width + patch_apod_width
        # keep adding patches until the maximum patch size is smaller 
        # than the allowed `max_patch_size`:
        while dist2npix(max_patch_width, shape, wcs) > dist2npix(max_patch_size, shape, wcs):
            num_patches_ra += 1
            patch_width = unapod_width / num_patches_ra
            max_patch_width = patch_width + apod_width + patch_apod_width
        min_patch_ra_ctr = ra_min + apod_width + patch_width / 2
        max_patch_ra_ctr = ra_max - apod_width - patch_width / 2
        patch_ra_ctrs = np.linspace(min_patch_ra_ctr, max_patch_ra_ctr, num_patches_ra)

    unapod_height = height - 2 * apod_width
    if dist2npix(height, shape, wcs) <= dist2npix(max_patch_size, shape, wcs):
        patch_height = unapod_height
        patch_dec_ctrs = np.array([dec_ctr])
    else:
        num_patches_dec = 1
        patch_height = unapod_height / num_patches_dec
        max_patch_height = patch_height + apod_width + patch_apod_width
        while dist2npix(max_patch_height, shape, wcs) > dist2npix(max_patch_size, shape, wcs):
            num_patches_dec += 1
            patch_height = unapod_height / num_patches_dec
            max_patch_height = patch_height + apod_width + patch_apod_width
        min_patch_dec_ctr = dec_min + apod_width + patch_height / 2
        max_patch_dec_ctr = dec_max - apod_width - patch_height / 2
        patch_dec_ctrs = np.linspace(min_patch_dec_ctr, max_patch_dec_ctr, num_patches_dec)

    return patch_ra_ctrs, patch_dec_ctrs, patch_width, patch_height


class MapPatches:
    """Methods to divide a map into a grid of smaller patches.

    The un-apodized region of the map is divided into a grid; each grid
    cell corresponds to the inner, un-apodized region of a smaller patch,
    which extends beyond the grid cell to leave room for apodization, and
    for patches along the outer border of the grid, to ensure that the
    patch reaches the edge of the (full-sized) map.

    See Also
    --------
    divide_map_area_into_patches

    Notes
    -----
    A map of width `W` and height `H` apodized over a region of width
    `map_apod_width` along its edges has an inner, un-apodized region of
    width `w = W - 2 * map_apod_width` and height
    `h = H - 2 * map_apod_width`. We are primarily interested in the
    map within this un-apodized region.

    The un-apodized region is divided into a grid with `ny` rows and `nx`
    columns, so the width and height of each cell are `w / nx` and
    `h / ny`, respectively. Each grid cell corresponds to a smaller patch
    within the full map.

    We assume that the patches will also be apodized over a width
    `patch_apod_width`. Therefore, each patch is extended beyond the
    boundaries of its grid cell by an amount `patch_apod_width` on each
    side; i.e., each grid cell is the inner, un-apodized region of its
    corresponding patch.

    If the `map_apod_width` is larger than the `patch_apod_width`, then
    patches lying along the outer border of the grid must be extended by
    an additional amount (the difference bewteen `map_apod_width` and
    `patch_apod_width`) on its outer-facing sides, in order to reach the
    edges of the original map.

    We refer to the entire, extended patch as the "full patch" or "padded
    patch", and refer to the grid cells themselves as either the inner,
    un-apodized region of the patch or just "the patch". Note that the
    "padded" patch regions overlap with their neighbors, and the padded
    regions of patches along the edge of the map are larger than the
    other patches. The un-apodized region of each patch is the same size
    and does not overlap with any others, but it does not extend to the
    edge of the original.

    We also define the region of each patch to use when "combining" the
    patches back together, such that no patches overlap, but the
    combination of all patches covers the full region of the original map.
    For maps along the outer border of the grid, this region corresponds
    to the grid cell of the patch, but extended along its outer-facing
    sides to reach the map edges. Otherwise, it is simply the same as the
    grid cell (i.e., the inner, un-apodized region of the map).
    """

    def __init__(self, shape, wcs, map_apod_width=0,
                 max_patch_size=fgi.max_patch_size,
                 patch_apod_width=fgi.patch_apod_width):
        """Divide a map into a grid of smaller patches.

        Parameters
        ----------
        shape : tuple of int
            The shape `(Ny, Nx)` of the array holding the map data, where
            `Ny` and `Nx` are the number of pixels along the dec. and
            R.A. directions, respectively.
        wcs : astropy.wcs.wcs.WCS
            An astropy World Coordinate System instance for the
            pixelization of the map.
        map_apod_width : int or float, default=0
            The apodization width (in degrees) of the region along the
            map edges over which the map has been apodized. If the map is
            not apodized, pass `map_apod_width=0` (the default).
        max_patch_size : int or float, default=3
            The maximum allowed width or height (in degrees) of the
            smaller patches, including any padding along the edges to
            account for the apodization width of the patch and of the
            full map.
        patch_apod_width : int or float, default=0.25
            The apodization width (in degrees) that will be used to
            apodize the smaller patches.
        """
        # full map:
        self.map_apod_width = map_apod_width
        self.shape = shape[-2:]
        self.wcs = wcs
        self.ra_min, self.ra_max, self.dec_min, self.dec_max = maps.get_map_corner_coords(self.shape, self.wcs)
        map_res = maps.get_map_resolution(self.shape, self.wcs)

        # divide into patches:
        self.patch_apod_width = patch_apod_width
        (self.patch_ra_ctrs, self.patch_dec_ctrs,
         self.patch_width, self.patch_height) = divide_map_area_into_patches(self.shape, self.wcs,
                                                                             max_patch_size=max_patch_size,
                                                                             patch_apod_width=patch_apod_width,
                                                                             map_apod_width=map_apod_width)
        self.num_patch_cols = len(self.patch_ra_ctrs)
        self.num_patch_rows = len(self.patch_dec_ctrs)
        self.num_patches = self.num_patch_cols * self.num_patch_rows
        self.patch_nums = list(range(self.num_patches))

        self._patch_pixel_info = {} # filled in later if necessary
        self._patch_info = {}
        for i, ra in enumerate(self.patch_ra_ctrs):
            # check whether this patch is in the first or last column:
            first_col = (i == 0)
            last_col = (i == (self.num_patch_cols - 1))
            # calculate the region within the full "padded" patches in
            # this column (i.e., including extra space for apodization)
            # that does not overlap with any other patches, but does
            # extend to the map edges if the patch lies along the outer
            # border of the grid; this is the region of the patch that
            # will actually be used when combining the patches back
            # together:
            ra_min_to_use = self.ra_min if first_col else ra - self.patch_width / 2
            ra_max_to_use = self.ra_max if last_col else ra + self.patch_width / 2
            ra_ctr_to_use = (ra_max_to_use + ra_min_to_use) / 2
            width_to_use = ra_max_to_use - ra_min_to_use
            # calculate the full, "padded" region of patches in this column:
            padded_ra_min = ra_min_to_use if first_col else ra_min_to_use - patch_apod_width
            padded_ra_max = ra_max_to_use if last_col else ra_max_to_use + patch_apod_width
            padded_ra_ctr = (padded_ra_max + padded_ra_min) / 2
            padded_width = padded_ra_max - padded_ra_min
            # repeat for each row of patches:
            for j, dec in enumerate(self.patch_dec_ctrs):
                first_row = (j == 0)
                last_row = (j == (self.num_patch_rows - 1))
                # height of region to use when combining all patches:
                dec_min_to_use = self.dec_min if first_row else dec - self.patch_height / 2
                dec_max_to_use = self.dec_max if last_row else dec + self.patch_height / 2
                dec_ctr_to_use = (dec_max_to_use + dec_min_to_use) / 2
                height_to_use = dec_max_to_use - dec_min_to_use
                # height of full patch region:
                padded_dec_min = dec_min_to_use if first_row else dec_min_to_use - patch_apod_width
                padded_dec_max = dec_max_to_use if last_row else dec_max_to_use + patch_apod_width
                padded_dec_ctr = (padded_dec_max + padded_dec_min) / 2
                padded_height = padded_dec_max - padded_dec_min
                # assign a unique integer to the patch,
                # and add its info to the dict:
                pnum = j * self.num_patch_cols + i
                self._patch_info[pnum] = {'patch_num': pnum, 'patch_col_num': i, 'patch_row_num': j,
                                         'ra_ctr': ra, 'dec_ctr': dec,
                                         'width': self.patch_width, 'height': self.patch_height,
                                         'ra_min': ra - self.patch_width/2, 'ra_max': ra + self.patch_width/2,
                                         'dec_min': dec - self.patch_height/2, 'dec_max': dec + self.patch_height/2,
                                         'padded_width': padded_width, 'padded_height': padded_height,
                                         'padded_ra_ctr': padded_ra_ctr, 'padded_dec_ctr': padded_dec_ctr,
                                         'padded_ra_min': padded_ra_min, 'padded_ra_max': padded_ra_max,
                                         'padded_dec_min': padded_dec_min, 'padded_dec_max': padded_dec_max,
                                         'width_to_use': width_to_use, 'height_to_use': height_to_use,
                                         'ra_ctr_to_use': ra_ctr_to_use, 'dec_ctr_to_use': dec_ctr_to_use,
                                         'ra_min_to_use': ra_min_to_use, 'ra_max_to_use': ra_max_to_use,
                                         'dec_min_to_use': dec_min_to_use, 'dec_max_to_use': dec_max_to_use}
                (self._patch_info[pnum]['shape'],
                 self._patch_info[pnum]['wcs']) = maps.get_shape_wcs(map_res, ra, dec, self.patch_width,
                                                                     height=self.patch_height)
                (self._patch_info[pnum]['padded_shape'],
                 self._patch_info[pnum]['padded_wcs']) = maps.get_shape_wcs(map_res, padded_ra_ctr, padded_dec_ctr,
                                                                            padded_width, height=padded_height)
                (self._patch_info[pnum]['shape_to_use'],
                 self._patch_info[pnum]['wcs_to_use']) = maps.get_shape_wcs(map_res, ra_ctr_to_use, dec_ctr_to_use,
                                                                            width_to_use, height=height_to_use)


    def get_num_patches(self):
        """Return the total number of patches, the number of patch
        columns (along the R.A. direction), and number of patch rows
        (along the dec. direction)
        """
        return self.num_patches, self.num_patch_cols, self.num_patch_rows


    def get_patch_dimensions(self, padded=False, patch_num=None):
        """Width and height of a patch.

        Parameters
        ----------
        padded : bool, default=False
            Whether to return the dimensions of the full, padded patch,
            or just its inner, un-apodized region (the default).
            If `padded=True`, you must also pass the `patch_num`.
        patch_num : int or None, default=None
            The unique number assigned to the patch. Must be provided if
            `padded=True`; otherwise, if `padded=False`, it will be
            ignored.

        Returns
        -------
        width, height : float
            The width and height of the patch, in degrees.

        See Also
        --------
        get_patch_num : The unique integer assigned to a given patch.
        """
        if padded:
            if patch_num is None:
                raise ValueError("The integer `patch_num` must be provided when `padded=True`.")
            width = self._patch_info[patch_num]['padded_width']
            height = self._patch_info[patch_num]['padded_height']
        else:
            width = self.patch_width
            height = self.patch_height
        return width, height


    def get_patch_center_coords(self, patch_num, padded=False):
        """Coordinates of the patch center.

        Parameters
        ----------
        patch_num : int
            The unique number assigned to the patch.
        padded : bool, default=False
            Whether to return the center of the full, padded patch, or
            just its inner, un-apodized region (the default).

        Returns
        -------
        ra, dec : float
            The right ascension and declination coordinates (in degrees),
            respectively, of the patch center.

        See Also
        --------
        get_patch_num : The unique integer assigned to a given patch.
        """
        if padded:
            ra = self._patch_info[patch_num]['padded_ra_ctr']
            dec = self._patch_info[patch_num]['padded_dec_ctr']
        else:
            ra = self._patch_info[patch_num]['ra_ctr']
            dec = self._patch_info[patch_num]['dec_ctr']
        return ra, dec


    def get_patch_geometry(self, patch_num=None, patch_col_num=None,
                           patch_row_num=None, ra=None, dec=None, padded=True):
        """The geometry of the patch.

        One of the following must be passed:
        - `patch_num`
        - both `patch_col_num` and `patch_row_num`
        - both `ra` and `dec`

        Parameters
        ----------
        patch_num : int or None, default=None
            The unique number assigned to the patch. Must be passed if
            `patch_col_num`, `patch_row_num`, `ra`, and `dec` are not
            passed.
        patch_col_num, patch_row_num : int or None, default=None
            The column and row numbers, respectively, for the patch.
            The columns and rows are numbered, starting from zero, with
            the first column/row corresponding to the minimum R.A./dec.
            in the original map. Must be passed if `patch_num`, `ra`, and
            `dec` are not passed.
        ra, dec : float or None, default=None
            A pair of R.A. and dec. coordinates (in degrees) located
            within the original map. The patch whose non-overlapping
            region contains this position will be used. Must be passed if
            `patch_num`, `patch_col_num`, and `patch_row_num` are not
            passed.
        padded : bool, default=False
            Whether to return the geometry of the full, padded patch, or
            just its inner, un-apodized region (the default).

        Returns
        -------
        shape : tuple of int
            The shape `(Ny, Nx)` of the array holding the patch map data,
            where `Ny` and `Nx` are the number of pixels along the dec.
            and R.A. directions, respectively.
        wcs : astropy.wcs.wcs.WCS
            An astropy World Coordinate System instance for the
            pixelization of the patch map.
        """
        if patch_num is None:
            patch_num = self.get_patch_num(ra=ra, dec=dec, patch_col_num=patch_col_num, patch_row_num=patch_row_num)
        shape_key = 'padded_shape' if padded else 'shape'
        wcs_key = 'padded_wcs' if padded else 'wcs'
        return self._patch_info[patch_num][shape_key], self._patch_info[patch_num][wcs_key]


    def get_patch_geometry_to_use(self, patch_num=None, patch_col_num=None,
                                  patch_row_num=None, ra=None, dec=None,
                                  include_apodized_region=True):
        """The geometry of the non-overlapping region of a patch.

        One of the following must be passed:
        - `patch_num`
        - both `patch_col_num` and `patch_row_num`
        - both `ra` and `dec`

        Parameters
        ----------
        patch_num : int or None, default=None
            The unique number assigned to the patch. Must be passed if
            `patch_col_num`, `patch_row_num`, `ra`, and `dec` are not
            passed.
        patch_col_num, patch_row_num : int or None, default=None
            The column and row numbers, respectively, for the patch.
            The columns and rows are numbered, starting from zero, with
            the first column/row corresponding to the minimum R.A./dec.
            in the original map. Must be passed if `patch_num`, `ra`, and
            `dec` are not passed.
        ra, dec : float or None, default=None
            A pair of R.A. and dec. coordinates (in degrees) located
            within the original map. The patch whose non-overlapping
            region contains this position will be used. Must be passed if
            `patch_num`, `patch_col_num`, and `patch_row_num` are not
            passed.
        include_apodized_region : bool, default=True
            For patches located along the outer border, whether to return
            the geometry for the non-overlapping patch region that
            extends to the edges of the original map (the default). Not
            used for other patches.

        Returns
        -------
        shape : tuple of int
            The shape `(Ny, Nx)` of the array holding the patch map data,
            where `Ny` and `Nx` are the number of pixels along the dec.
            and R.A. directions, respectively.
        wcs : astropy.wcs.wcs.WCS
            An astropy World Coordinate System instance for the
            pixelization of the patch map.
        """
        if patch_num is None:
            patch_num = self.get_patch_num(ra=ra, dec=dec, patch_col_num=patch_col_num, patch_row_num=patch_row_num)
        shape_key = 'shape_to_use' if include_apodized_region else 'shape'
        wcs_key = 'wcs_to_use' if include_apodized_region else 'wcs'
        return self._patch_info[patch_num][shape_key], self._patch_info[patch_num][wcs_key]


    def get_patch_region_to_use(self, patch_num=None, patch_col_num=None,
                                patch_row_num=None, ra=None, dec=None,
                                include_apodized_region=True):
        """Minimum and maximum right ascension (R.A.) and declination
        (dec.) of the non-overlapping region of a patch.

        One of the following must be passed:
        - `patch_num`
        - both `patch_col_num` and `patch_row_num`
        - both `ra` and `dec`

        Parameters
        ----------
        patch_num : int or None, default=None
            The unique number assigned to the patch. Must be passed if
            `patch_col_num`, `patch_row_num`, `ra`, and `dec` are not
            passed.
        patch_col_num, patch_row_num : int or None, default=None
            The column and row numbers, respectively, for the patch.
            The columns and rows are numbered, starting from zero, with
            the first column/row corresponding to the minimum R.A./dec.
            in the original map. Must be passed if `patch_num`, `ra`, and
            `dec` are not passed.
        ra, dec : float or None, default=None
            A pair of R.A. and dec. coordinates (in degrees) located
            within the original map. The patch whose non-overlapping
            region contains this position will be used. Must be passed if
            `patch_num`, `patch_col_num`, and `patch_row_num` are not
            passed.
        include_apodized_region : bool, default=True
            For patches located along the outer border, whether to return
            the values for the non-overlapping patch region that extends
            to the edges of the original map (the default). Not used for
            other patches.

        Returns
        -------
        ra_min, ra_max, dec_min, dec_max : float
            The minimum and maximum R.A. and dec. (in degrees),
            respectively, in the patch.
        """
        if patch_num is None:
            patch_num = self.get_patch_num(ra=ra, dec=dec, patch_col_num=patch_col_num, patch_row_num=patch_row_num)
        if include_apodized_region:
            return (self._patch_info[patch_num]['ra_min_to_use'], self._patch_info[patch_num]['ra_max_to_use'],
                    self._patch_info[patch_num]['dec_min_to_use'], self._patch_info[patch_num]['dec_max_to_use'])
        else:
            return (self._patch_info[patch_num]['ra_min'], self._patch_info[patch_num]['ra_max'],
                    self._patch_info[patch_num]['dec_min'], self._patch_info[patch_num]['dec_max'])


    def _patch_num_for_coords(self, ra, dec):
        """Returns the unique number assigned to the patch whose
        non-overlapping region contains the given R.A. and dec.
        coordinates (in degrees).
        """
        if not utils.ra_dec_are_in_patch(ra, dec, self.ra_min, self.dec_min, self.ra_max, self.dec_max):
            raise ValueError(f"`{ra=}`, `{dec=}` is outside of the map.")
        patch_num = 0
        ra_min, ra_max, dec_min, dec_max = self.get_patch_region_to_use(patch_num=patch_num)
        coords_in_patch = utils.ra_dec_are_in_patch(ra, dec, ra_min, dec_min, ra_max, dec_max)
        while not coords_in_patch:
            patch_num += 1
            ra_min, ra_max, dec_min, dec_max = self.get_patch_region_to_use(patch_num=patch_num)
            coords_in_patch = utils.ra_dec_are_in_patch(ra, dec, ra_min, dec_min, ra_max, dec_max)
        return patch_num


    def _patch_num_for_row_and_col(self, patch_row_num, patch_col_num):
        """Returns the unique number assigned to the patch in the given
        row and column.
        """
        if (patch_col_num + 1) > self.num_patch_cols:
            raise ValueError(f"`{patch_col_num = }`: there are only {self.num_patch_cols} patches in the R.A. direction.")
        if (patch_row_num + 1) > self.num_patch_rows:
            raise ValueError(f"`{patch_row_num = }`: there are only {self.num_patch_rows} patches in the dec. direction.")
        patch_num = patch_row_num * self.num_patch_cols + patch_col_num
        return patch_num


    def get_patch_num(self, ra=None, dec=None, patch_col_num=None, patch_row_num=None):
        """The unique number assigned to a patch.

        Parameters
        ----------
        ra, dec : float or None, default=None
            A pair of R.A. and dec. coordinates (in degrees) located
            within the original map. The number of the patch whose
            non-overlapping region contains this position will be used.
            Must be passed if `patch_col_num`,and `patch_row_num` are not
            passed.
        patch_col_num, patch_row_num : int or None, default=None
            The column and row numbers, respectively, for the patch.
            The columns and rows are numbered, starting from zero, with
            the first column/row corresponding to the minimum R.A./dec.
            in the original map. Must be passed if `ra` and  `dec` are
            not passed; otherwise, will be ignored if `ra` and  `dec` are
            passed.

        Returns
        -------
        patch_num : int
            The unique number assigned to the patch.
        """
        if (ra is not None) and (dec is not None):
            patch_num = self._patch_num_for_coords(ra, dec)
        elif (patch_col_num is not None) and (patch_row_num is not None):
            patch_num = self._patch_num_for_row_and_col(patch_row_num, patch_col_num)
        else:
            raise ValueError(f"You must provide either (1) a pair of `ra` and `dec` coordinates "
                             "(right ascension and declination, respectively, in degrees),"
                             " or (2) the `patch_col_num` and `patch_row_num` for the patch.")
        return patch_num


    def get_patch_row_and_col(self, patch_num):
        """Row and column of the given patch.

        Parameters
        ----------
        patch_num : int
            The unique number assigned to the patch.

        Returns
        -------
        patch_col_num, patch_row_num : int
            The column and row numbers, respectively, for the patch.
            The columns and rows are numbered, starting from zero, with
            the first column/row corresponding to the minimum R.A./dec.
            in the original map.
        """
        patch_col_num = patch_num % self.num_patch_cols
        patch_row_num = (patch_num - patch_col_num) // self.num_patch_cols
        return patch_row_num, patch_col_num


    def trim_patch_catalog(self, catalog, patch_num, padded=True,
                           ra_col='RADeg', dec_col='decDeg'):
        """Remove any rows from a catalog with coordinate positions
        outside of the given patch.

        Parameters
        ----------
        catalog : pandas.DataFrame
            A catalog containing columns for R.A. and dec. coordinates
            (in degrees).
        patch_num : int
            The unique number assigned to the patch.
        padded : bool, default=False
            Whether to use the foorprint of the full, padded patch, or
            just its inner, un-apodized region (the default).
        ra_col, dec_col : str, optional
            The name of the columns for the R.A. and dec. coordinates,
            respectively. Defaults are `'RADeg'` and `'decDeg'`.

        Returns
        -------
        ocat : pandas.DataFrame
            The catalogs containing only rows with coordinates inside the
            given patch.

        See Also
        --------
        get_patch_num : The unique integer assigned to a given patch.
        """
        icat_cols = catalog.columns.values
        ocat = catalog.copy()
        ocat = maps.add_pixel_coords_to_catalog(ocat, self.shape, self.wcs, ra_key=ra_col, dec_key=dec_col)
        xmin, xmax, ymin, ymax = self._patch_pixel_lims(patch_num, padded=padded)
        ocat = ocat[ocat['x_pixel'].between(xmin, xmax) & ocat['y_pixel'].between(ymin, ymax)][icat_cols].copy()
        return ocat


    def trim_patch_catalog_to_use(self, catalog, patch_num,
                                  include_apodized_region=True,
                                  ra_col='RADeg', dec_col='decDeg'):
        """Remove any rows from a catalog with coordinate positions
        outside of the non-overlapping region of the given patch.

        Parameters
        ----------
        catalog : pandas.DataFrame
            A catalog containing columns for R.A. and dec. coordinates
            (in degrees).
        patch_num : int
            The unique number assigned to the patch.
        include_apodized_region : bool, default=True
            For patches located along the outer border, whether to use
            the footprint of the non-overlapping patch region that
            extends to the edges of the original map (the default). Not
            used for other patches.
        ra_col, dec_col : str, optional
            The name of the columns for the R.A. and dec. coordinates,
            respectively. Defaults are `'RADeg'` and `'decDeg'`.

        Returns
        -------
        ocat : pandas.DataFrame
            The catalogs containing only rows with coordinates inside the
            given non-overlapping patch region.

        See Also
        --------
        get_patch_num : The unique integer assigned to a given patch.
        """
        icat_cols = catalog.columns.values
        ocat = catalog.copy()
        ocat = maps.add_pixel_coords_to_catalog(ocat, self.shape, self.wcs, ra_key=ra_col, dec_key=dec_col)
        xmin, xmax, ymin, ymax = self._patch_pixel_lims_to_use(patch_num, include_apodized_region=include_apodized_region)
        ocat = ocat[ocat['x_pixel'].between(xmin, xmax) & ocat['y_pixel'].between(ymin, ymax)][icat_cols].copy()
        return ocat


    def _patch_pixel_lims(self, patch_num, padded=True):
        """Returns the minimum and maximum x and y pixels for the given
        patch region within the full, original map.
        """
        if patch_num not in self._patch_pixel_info:
            self._calc_pixel_info_for_patches(patch_nums=[patch_num])
        pinfo = self._patch_pixel_info[patch_num]
        if padded:
            return pinfo['padded_xmin'], pinfo['padded_xmax'], pinfo['padded_ymin'], pinfo['padded_ymax']
        else:
            return pinfo['xmin'], pinfo['xmax'], pinfo['ymin'], pinfo['ymax']


    def _patch_pixel_lims_to_use(self, patch_num, include_apodized_region=True):
        """Returns the minimum and maximum x and y pixels for the given
        non-overlapping patch region within the full, original map.
        """
        if patch_num not in self._patch_pixel_info:
            self._calc_pixel_info_for_patches(patch_nums=[patch_num])
        pinfo = self._patch_pixel_info[patch_num]
        if include_apodized_region:
            return pinfo['xmin_to_use'], pinfo['xmax_to_use'], pinfo['ymin_to_use'], pinfo['ymax_to_use']
        else:
            return pinfo['xmin'], pinfo['xmax'], pinfo['ymin'], pinfo['ymax']


    def _get_xy_pix_maps(self):
        """Maps of x and y pixel values in the full, original map."""
        ypix_map, xpix_map = enmap.pixmap(self.shape, wcs=self.wcs)
        return xpix_map, ypix_map


    def _calc_pixel_lims_for_region(self, shape, wcs, xpix_map=None, ypix_map=None):
        """Returns the minimum and maximum x and y pixels for the given
        geometry within the full, original map.
        """
        if (xpix_map is None) or (ypix_map is None):
            xpix_map, ypix_map = self._get_xy_pix_maps()
        xpix_map = enmap.project(xpix_map.copy(), shape, wcs)
        ypix_map = enmap.project(ypix_map.copy(), shape, wcs)
        xmin = int(np.min(xpix_map))
        xmax = int(np.max(xpix_map))
        ymin = int(np.min(ypix_map))
        ymax = int(np.max(ypix_map))
        return xmin, xmax, ymin, ymax


    def _calc_pixel_info_for_patches(self, patch_nums=None):
        """Calculate and store the minimum and maximum x and y pixels for
        the given patches within the full, original map.
        """
        patch_nums = patch_nums if (patch_nums is not None) else self.patch_nums
        patch_nums_to_calc = []
        pix_info_keys = ['xmin', 'xmax', 'ymin', 'ymax']
        keys = [*pix_info_keys, *[f'padded_{key}' for key in pix_info_keys],
                *[f'{key}_to_use' for key in pix_info_keys]]
        for pnum in patch_nums:
            if pnum not in self._patch_pixel_info:
                patch_nums_to_calc.append(pnum)
            elif not all([key in self._patch_pixel_info[pnum] for key in keys]):
                patch_nums_to_calc.append(pnum)
        # calculate maps of x and y pixels once:
        if len(patch_nums_to_calc) > 0:
            xpix_map, ypix_map = self._get_xy_pix_maps()
            kwargs = {'xpix_map': xpix_map, 'ypix_map': ypix_map}
        # loop through patches:
        for pnum in patch_nums_to_calc:
            if pnum not in self._patch_pixel_info:
                self._patch_pixel_info[pnum] = {}
            patch_shape, patch_wcs = self.get_patch_geometry(patch_num=pnum, padded=False)
            padded_patch_shape, padded_patch_wcs = self.get_patch_geometry(patch_num=pnum, padded=True)
            patch_shape_to_use, patch_wcs_to_use = self.get_patch_geometry_to_use(patch_num=pnum)
            if 'xmin' not in self._patch_pixel_info[pnum]:
                (self._patch_pixel_info[pnum]['xmin'],
                 self._patch_pixel_info[pnum]['xmax'],
                 self._patch_pixel_info[pnum]['ymin'],
                 self._patch_pixel_info[pnum]['ymax']) = self._calc_pixel_lims_for_region(patch_shape, patch_wcs,
                                                                                          **kwargs)
            if 'padded_xmin' not in self._patch_pixel_info[pnum]:
                (self._patch_pixel_info[pnum]['padded_xmin'],
                 self._patch_pixel_info[pnum]['padded_xmax'],
                 self._patch_pixel_info[pnum]['padded_ymin'],
                 self._patch_pixel_info[pnum]['padded_ymax']) = self._calc_pixel_lims_for_region(padded_patch_shape,
                                                                                                 padded_patch_wcs,
                                                                                                 **kwargs)
            if 'xmin_to_use' not in self._patch_pixel_info[pnum]:
                (self._patch_pixel_info[pnum]['xmin_to_use'],
                 self._patch_pixel_info[pnum]['xmax_to_use'],
                 self._patch_pixel_info[pnum]['ymin_to_use'],
                 self._patch_pixel_info[pnum]['ymax_to_use']) = self._calc_pixel_lims_for_region(patch_shape_to_use,
                                                                                                 patch_wcs_to_use,
                                                                                                 **kwargs)

