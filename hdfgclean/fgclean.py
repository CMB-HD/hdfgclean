import os
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
from pixell import enmap
from hdsims import  utils, simutils, fgcatalogs, maps
from . import mpi, fgclean_info as fgi, fgutils, fgmaps, fgfilters, iterfgclean, fgclean_sources, fgclean_clusters, fgmasks


class FGClean:
    
    @classmethod
    def from_config(cls, config_fname, safe=True, **kwargs):
        """Initialize the `FGClean` class from a configuration YAML file.

        Parameters
        ----------
        config_fname : str
            The configuration file name.

        Returns
        -------
        An instance of `FGClean`.

        Other Parameters
        ----------------
        safe : bool, default=True
            By default (when `safe=True`), the YAML file is loaded by
            calling `yaml.safe_load`. If `safe=False`, the file is loaded
            by calling `yaml.load` and passing `Loader=yaml.Loader`;
            NOTE that this allows the execution of any arbitrary python
            code when loading a YAML file, so you should only pass
            `safe=False` for files from a trusted source.
        **kwargs : dict
            Additional keyword arguments passed to `FGClean`. If an
            argument is present in both the configuration file and in the
            `kwargs` dictionary, the latter is used.

        See Also
        --------
        save_config : Save a configuration file.
        fgutils.load_yaml : Load a YAML file with the PyYAML package.

        Notes
        -----
        See `fgutils.load_yaml` and the PyYAML documentation for
        information about the `safe` argument. Passing `safe=False` is
        required to load YAML files in which, e.g., numpy arrays have
        been saved, but you should never pass `safe=False` unless your
        configuration file is from a trusted source.
        """
        # load the file and update `config` dict with any `kwargs`:
        config = fgutils.load_yaml(config_fname, safe=safe)
        config = {**config, **kwargs}

        # get required args:
        default_output_dir, _ = os.path.split(config_fname)
        output_dir = config.get('output_dir', default_output_dir)
        # input maps:
        if 'imaps' not in config:
            raise ValueError("You must provide `imaps` dictionary with the"
                             " input map file names at each frequency.")
        else:
            imaps = config['imaps']
        # beams:
        if 'beam_fwhms' not in config:
            raise ValueError("You must include the `beam_fwhms` dictionary"
                             " with the beam size for each map frequency.")
        else:
            beam_fwhms = config['beam_fwhms'].copy()

        # get kwargs:
        required_arg_names = ['output_dir', 'imaps', 'beam_fwhms']
        allowed_kwarg_names = list(fgutils._get_all_kwargs(cls).keys())
        fgclean_keys = [*required_arg_names, *allowed_kwarg_names]
        ignored_keys = [key for key in config if (key not in fgclean_keys)]
        if (len(ignored_keys) > 0) and mpi.is_rank0:
            warnings.warn(f"The following keys in {config_fname} or in"
                          " `kwargs` are not accepted by the FGClean"
                          f" class and will be ignored: {ignored_keys}")
        fgclean_kwargs = fgutils.dict_with_keys(config, allowed_kwarg_names)
        # load in catalogs
        catalog_kwarg_names = ['sources_to_mask_before_fgclean_catalog',
                               'clusters_to_mask_after_fgclean_catalog']
        for key in catalog_kwarg_names:
            fname = kwargs.get(key, None)
            if fname is not None:
                kwargs[key] = fgcatalogs.load_catalog(fname)

        return cls(output_dir, imaps, beam_fwhms, **fgclean_kwargs)


    @classmethod
    def save_config(cls, config_fname, imaps, beam_fwhms,
                    output_dir=None, overwrite=False, **kwargs):
        """Save a configuration YAML file that can be used to initialize
        `FGClean`.

        Parameters
        ----------
        config_fname : str
            The configuration file name.
        imaps : dict of str
            A dictionary of the map file names at each frequency. Must
            have a key (`int` or `float`; we recommend the former) for
            each map frequency (in GHz), and the corresponding value
            is the file name of the saved map at that frequency (as a
            `.fits` file that can be read with `pixell.enmap.read_map`).
            These are the maps that will be FG cleaned; note that they
            should be apodized.
        beam_fwhms : dict of float
            A dictionary with the beam size at each frequency (full-width
            at half-maximum, in arcminutes, for a Gaussian beam profile).
            The keys must be the same as in the `imaps` dictionary.
        output_dir : str, optional
            Absolute path to the directory where files will be saved.
            By default, this will be the directory in which your
            configuration file will be saved (determined based on its
            file name).
        overwrite : bool, default=False
            Whether to overwrite an existing configuration file with the
            same file name.
        **kwargs : dict
            Additional keyword arguments that can be passed to `FGClean`.
            Use `str` file names instead of `pixell.enmap.ndmap` (`.fits`
            files) or `pandas.DataFrame` objects (`.csv` files).

        See Also
        --------
        from_config : Initialize `FGClean` from a configuration YAML file

        Notes
        -----
        A YAML file containing only built-in Python types (e.g. `int`,
        `float`, `list`, `dit`) may be loaded by passing `safe=True` to
        the `from_config` method. If you save any other type of object
        (e.g., numpy arrays or callable functions), you must pass
        `safe=False` to the `from_config` method in order to load the
        file. Please see the notes in `from_config` and the PyYAML
        documentation before passing `safe=False` when loading YAML
        files.
        """
        if os.path.exists(config_fname) and (not overwrite):
            raise FileExistsError(f"The file {config_fname} already exists."
                                  " You may pass a new `config_fname`, or"
                                  " you may pass `overwrite=True` to"
                                  " overwrite the existing file.")
        # make sure `imaps` is not dict of actual `pixell.enmap.ndmap`s:
        if any([isinstance(imap, enmap.ndmap) for imap in imaps.values()]):
            raise ValueError("The values in the `imaps` dictionary should"
                             " be a `.fits` file name for each map.")
        # same for any noise maps:
        noise_map_kwarg_names = ['noise_maps_for_source_filters',
                                 'noise_maps_for_cluster_filters',
                                 'noise_maps_for_source_mask_filters']
        for key in noise_map_kwarg_names:
            if key in kwargs:
                map_values = kwargs[key].values()
                if any([isinstance(imap, enmap.ndmap) for imap in map_values]):
                    raise ValueError(f"The values in the `{key}` dictionary "
                                     "should be a `.fits` file name for each map.")
        # and for catalogs:
        catalog_kwarg_names = ['sources_to_mask_before_fgclean_catalog',
                               'clusters_to_mask_after_fgclean_catalog']
        for key in catalog_kwarg_names:
            if key in kwargs:
                if isinstance(kwargs[key], pd.DataFrame):
                    raise ValueError(f"The `{key}` should be the path to the `.csv` file.")
        # only save kwargs that can be passed to `FGClean`:
        allowed_kwarg_names = list(fgutils._get_all_kwargs(cls).keys())
        ignored_keys = [key for key in kwargs if (key not in allowed_kwarg_names)]
        if (len(ignored_keys) > 0) and mpi.is_rank0:
            warnings.warn(f"The following keys in {config_fname} or in"
                          " `kwargs` are not accepted by the FGClean"
                          f" class and will be ignored: {ignored_keys}")
        config = fgutils.dict_with_keys(kwargs, allowed_kwarg_names, copy=True)
        config['imaps'] = imaps
        config['beam_fwhms'] = beam_fwhms
        if output_dir is None:
            output_dir, _ = os.path.split(config_fname)
        config['output_dir'] = output_dir
        if mpi.is_rank0:
            config_dir, _ = os.path.split(config_fname)
            utils.mkdir(config_dir)
            utils.save_yaml(config_fname, config, overwrite=overwrite)
            print(f'Configuration file has been saved to {config_fname}')
        mpi.comm.barrier()


    def __init__(self, output_dir, imaps, beam_fwhms,
                 subtract_sources=True, subtract_clusters=True,
                 map_apod_width=0,
                 max_patch_size=fgi.max_patch_size,
                 patch_apod_width=fgi.patch_apod_width,
                 map_shape=None, map_wcs=None,
                 noise_maps_for_source_filters=None,
                 noise_maps_for_cluster_filters=None,
                 noise_maps_for_source_mask_filters=None,
                 hd_noise_sims_for_filters_dir=None,
                 noise_maps_kwargs={},
                 sources_rms_gw=fgi.rms_gw_sources,
                 sources_snr_threshold_list=fgi.sources_snr_threshold_list,
                 extrapolate_dim_sources=True,
                 remeasure_missubtracted_sources=True,
                 min_num_iter_sources_per_snr=fgi.min_num_iter_sources_per_snr,
                 remeasure_snr_threshold_list=fgi.remeasure_sources_snr_threshold_list,
                 max_ntimes_remeasure=fgi.max_ntimes_remeasure_sources,
                 min_snr_for_spectral_index=fgi.index_min_snr,
                 nsigma_for_spectral_index=fgi.index_nsigma_to_remove,
                 freq_for_radio=None, freq_for_cib=None,
                 calc_beam_solid_angle_funcs=True,
                 beam_solid_angle_funcs=None,
                 clusters_rms_gw=fgi.rms_gw_clusters,
                 clusters_snr_threshold_list=fgi.clusters_snr_threshold_list,
                 clusters_iter_match_radius=fgi.clusters_iter_match_radius,
                 cluster_profiles=None,
                 cluster_profiles_info=None,
                 smooth_p2d_npix=fgi.p2d_smooth_npix,
                 rms_niter=fgi.rms_niter, rms_nsigma=fgi.rms_nsigma,
                 sources_to_mask_before_fgclean_catalog=None,
                 clusters_to_mask_after_fgclean_catalog=None,
                 verbose=True, log=None):
        """Initialize the `FGClean` class.

        Parameters
        ----------
        output_dir : str
            Absolute path to the directory where files will be saved.
        imaps : dict of str or dict of pixell.enmap.ndmap
            A dictionary of the maps at each frequency. The dictionary
            must have a key (`int` or `float`; we recommend the former)
            for each map frequency (in GHz), and the corresponding value
            is the map at that frequency. This can be the actual map
            (as a `pixell.enmap.ndmap`), or the file name of the saved
            map (as a `.fits` file that can be read with
            `pixell.enmap.read_map`). These are the maps that will be FG
            cleaned; note that they should be apodized.
        beam_fwhms : dict of float
            A dictionary with the beam size at each frequency (full-width
            at half-maximum, in arcminutes, for a Gaussian beam profile).
            The keys must be the same as in the `imaps` dictionary.
        subtract_sources : bool, default=True
            Whether to iteratively detect, measure, and remove (CIB and
            radio) point sources from the maps.
        subtract_clusters : bool, default=True
            Whether to iteratively detect, measure, and remove tSZ
            clusters from the maps.
        map_apod_width : int or float, default=0
            The apodization width (in degrees) of the region along the
            map edges over which the maps have been apodized.
        max_patch_size : int or float, default=3
            The maximum allowed width or height (in degrees) of the
            smaller patches in the original maps, including any padding
            along the edges to account for the apodization width of the
            patch and of the full map. The FG cleaning is run on each
            patch individually, and the results are combined at the end.
        patch_apod_width : int or float, default=0.25
            The apodization width (in degrees) that will be used to
            apodize the smaller patches.

        Other Parameters
        ----------------
        map_shape : tuple of int, optional
            The shape `(Ny, Nx)` of the array holding the map data, where
            `Ny` and `Nx` are the number of pixels along the dec. and
            R.A. directions, respectively.
        map_wcs : astropy.wcs.wcs.WCS, optional
            An astropy World Coordinate System instance for the
            pixelization of the maps.
        noise_maps_for_source_filters : dict, optional
            A dictionary of maps at each frequency used to quantify the
            noise in the point source matched filters. The format is the
            same as the `imaps` dictionary. By default, the maps in
            `imaps` will be used for this purpose. Ignored if
            `hd_noise_sims_for_filters_dir` is passed.
        noise_maps_for_cluster_filters : dict, optional
            A dictionary of maps at each frequency used to quantify the
            noise in the tSZ cluster matched filters. The format is the
            same as the `imaps` dictionary. By default, the maps in
            `imaps` will be used for this purpose. Ignored if
            `hd_noise_sims_for_filters_dir` is passed.
        noise_maps_for_source_mask_filters : dict, optional
            A dictionary of maps at each frequency used to quantify the
            noise in the point source matched filters used when making the
            masks after FG cleaning. The format is the same as the `imaps`
            dictionary. By default, if `noise_maps_for_source_filters`
            was passed, those maps will be used for this purpose;
            otherwise, the maps in `imaps` will be used. Ignored if
            `hd_noise_sims_for_filters_dir` is passed.
        hd_noise_sims_for_filters_dir : str, optional
            If you have generated a set of maps used to quantify the
            noise in the matched filter with the `hdsims` package, pass
            the path to the directory where the simulations are saved
            (i.e., the `hd_sims_dir` passed to `hdsims.hdsims.HDSims`).
            If not provided, then `noise_maps_for_cluster_filters`,
            `noise_maps_for_cluster_filters`, and
            `noise_maps_for_source_mask_filters` will be used.
        noise_maps_kwargs : dict, optional
            Keyword arguments passed to `fgfilters.NoiseSimsForFilters`
            (which is derived from `hdsims.hdsims.HDSims`).
            If you have generated a set of maps used to quantify the
            noise in the matched filter with the `hdsims` package,
            this includes all keyword arguments that were passed to
            `hdsims.hdsims.HDSims` in order to generate the simulations.
            Ignored if `hd_noise_sims_for_filters_dir` is not passed.
        sources_rms_gw : int or float, default=10
            The width and height (in arcminutes) of the grid cells used
            for the RMS map calculation (see "Notes" below) when
            filtering the maps to detect point sources. Ignored if
            `subtract_sources=False`.
        sources_snr_threshold_list : list of float, optional
            A list of signal-to-noise ratio (SNR) thresholds used to
            iteratively detect point sources. The default list is
            `[250, 100, 75, 50, 40, 30, 25, 20, 15, 12.5, 10, 7.5, 5, 4]`.
            Ignored if `subtract_sources=False`.
        extrapolate_dim_sources : bool, default=True
            Whether to use the detected point source catalogs at
            different frequencies (`freq_for_cib` or `freq_for_radio` for
            CIB or radio sources, respectively) to remove sources not
            detected at a given frequency. See the "Notes" section for
            further information. Ignored if `subtract_sources=False`.
        remeasure_missubtracted_sources : bool, default=True
            If `True`, try to re-measure the fluxes of any mis-subtracted
            point sources after subtracting all detected sources from the
            map at a given frequency. See the "Notes" section for further
            information. Ignored if `subtract_sources=False`.
        min_num_iter_sources_per_snr : int, default=10
            When iteratively detecting point sources, once less than
            `min_num_iter_sources_per_snr` are detected (and removed),
            move on to the next (lower) SNR in the list of SNR
            thresholds. Not used for the last (lowest) SNR threshold.
            Ignored if `subtract_sources=False`.
        remeasure_snr_threshold_list : list of float, optional
            A list of SNR thresholds to use while re-measuring
            mis-subtracted sources. The default is `[250, 100, 50, 25,
            15, 10, 5, 4]`. Ignored if `subtract_sources=False` or
            `remeasure_missubtracted_sources=False`.
        max_ntimes_remeasure : int, default=25
            The maximum number of times to repeat the re-measurement of
            mis-subtracted sources. Ignored if `subtract_sources=False`
            or `remeasure_missubtracted_sources=False`.
        min_snr_for_spectral_index : int, default=10
            When calculating the average CIB or radio spectral index
            between two frequencies, only use sources detected with SNR
            above `min_snr_for_spectral_index` at both frequencies.
            Ignored if `subtract_sources=False` or
            `extrapolate_dim_sources=False`.
        nsigma_for_spectral_index : float, default=False
            The number of standard deviations to use to remove outliers
            from the sources used for the spectral index calculation.
            Ignored if `subtract_sources=False` or
            `extrapolate_dim_sources=False`.
        freq_for_radio, freq_for_cib : int or float or None, optional
            The second frequency used to measure the radio or CIB
            spectral index, respectively (see the "Notes" section).
            Each must be one of the map frequencies. By default, the
            `freq_for_cib` and `freq_for_radio` will be the highest and
            lowest map frequencies, respectively. Ignored if
            `subtract_sources=False` or `extrapolate_dim_sources=False`.
        calc_beam_solid_angle_funcs : bool, default=True
            At each frequency, calculate the beam solid angle as a
            function of declination (due to the varying pixel size) by
            measuring the variation of the beam solid angle in the map.
            If `calc_beam_solid_angle_funcs=True` and the
            `beam_solid_angle_funcs` dictionary was passed, the
            calculation will only be done for frequencies not in the
            dictionary. If `calc_beam_solid_angle_funcs=False` and the
            `beam_solid_angle_funcs` was not passed (or does not contain
            all frequencies), a constant beam solid angle will be used.
        beam_solid_angle_funcs : dict or None, default=None
            A dictionary, with the frequencies in `imaps` as the keys, of
            callable functions that return the beam solid angle
            (steradians) for that frequency at a given map declination
            (degrees).
        clusters_rms_gw : int or float, default=40
            The width and height (in arcminutes) of the grid cells used
            for the RMS map calculation (see "Notes" below) when
            filtering the maps to detect tSZ clusters. Ignored if
            `subtract_clusters=False`.
        clusters_snr_threshold_list : list of float, optional
            A list of signal-to-noise ratio (SNR) thresholds used to
            iteratively detect clusters.
            The default list is `[50, 25, 15, 12.5, 10, 7.5, 5, 4]`.
            Ignored if `subtract_clusters=False`.
        clusters_iter_match_radius : int or float, default=1
            On a given cluster subtraction iteration, all clusters
            detected (using different filters, corresponding to different
            radial cluster profiles) within a distance of
            `clusters_iter_match_radius` from the highest-SNR detection in
            that region are identified as the same cluster, and only the
            highest-SNR detection is kept.
            Ignored if `subtract_clusters=False`.
        cluster_profiles : dict of dict, optional
            A dictionary containing a set of radial cluster profiles to
            use when detecting tSZ clusters; a multi-frequency matched
            filter is calculated for each profile.
            Each (key, value) pair must be, respectively, a label (`str`)
            for a given profile and a dictionary with the following keys
            and values:
            - `'radial_prof_func'` : A callable `function` for the radial
                                     profile. The first argument must be
                                     the angular distance (in arcminutes)
                                     from the origin.
            - `'args'` : A list of any additional positional arguments to
                         pass to the profile function.
            - `'kwargs'` : A dictionary of any additional keyword
                           arguments to pass to the profile function.
            - `'rmax'` : A maximum angular distance (in arcminutes) from
                         the origin, beyond which the radial profile is
                         truncated.
            The default set of cluster profiles are 11 Gaussian profiles
            with standard deviations of 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
            0.55, 0.6, 0.65, 0.7, and 0.75 arcminutes. If a non-default
            set of `cluster_profiles` is passed, you must also pass
            `cluster_profiles_info`.
            Ignored if `subtract_clusters=False`.
        cluster_profiles_info : str or None, default=None
            If a `cluster_profiles` dictionary was passed, you must also
            pass a short, simple description of the set of profiles to
            `cluster_profiles_info` to use in the output file names.
            Ignored if `subtract_clusters=False` or
            `cluster_profiles=None`.
        smooth_p2d_npix : int, default=3
            The number of pixels used to smooth the 2D "noise" power
            spectrum in the matched filter calculations, by convolving
            with a Gaussian with a standard deviation of
            `smooth_p2d_npix` pixels.
        rms_niter : int, default=10
            The number of iterations used to estimate the RMS map in each
            RMS grid cell.
        rms_nsigma : int or float, default=3
            The number of standard deviations used to remove outliers
            when iteratively calculating the RMS map in each grid cell.
        sources_to_mask_before_fgclean_catalog : optional
            A catalog (`pandas.DataFrame`)  of known, very bright sources
            in the maps, which will be masked before FG cleaning. Must
            have columns `'RADeg'`, `'decDeg'` (R.A., dec. of each
            source, in degrees) and a column for flux in mJy at each
            frequency (keys of `imaps`) named
            `f'fluxmJy_{round(freq)}GHz'`.
        clusters_to_mask_after_fgclean_catalog : optional
            A catalog (`pandas.DataFrame`) of any known clusters (e.g.,
            nearby, large, massive) that you would like to include in the
            final mask; must have columns `'RADeg'`, `'decDeg'` (R.A.,
            dec. of each cluster) and `mask_radius` (radius of each hole
            in the mask, in arcminutes).
        verbose : bool, default=False
            Whether to print messages describing the progress of the FG
            cleaning.
        log : logging.Logger, optional
            A `logging.Logger` instance to use when `verbose=True`. If a
            `log` is passed, any messages will be passed to `log.info`.
            Otherwise, messages will be passed to the `print` function.

        See Also
        --------
        fgmaps.divide_map_area_into_patches, fgmaps.MapPatches :
            Used to divide the maps into smaller patches.
        fgfilters.calc_rms_map : The RMS map calculation.
        fgfilters.get_default_gauss_cluster_profiles_dict :
            The default `cluster_profiles` dictionary.

        Notes
        -----
        The FG cleaning procedure used here is described in detail in
        MacInnis et. al. (2026), arXiv:2609.16128

        You should use a new `output_dir` for each FG cleaning run.

        If the `imaps` dictionary contains map file names and the
        `map_shape` and `map_wcs` are not passed, one of the `imaps will
        be loaded in order to obtain its `shape` and `wcs` attributes.

        The maps used to quantify the noise in the matched filters must
        be at least as large as the `max_patch_size` on each side; they
        do not need to be as large as the original maps (if the original
        maps are larger than the `max_patch_size`).

        A "SNR map" is a map of the signal-to-noise ratio (SNR) in each
        pixel of a matched-filtered map. The filtered map is the "signal"
        part of the SNR, and the "RMS map" is the "noise" part.

        Extrapolaing dim sources:
        Here is a brief, simplified summary of the "extrapolation" done
        during the point source subtraction:
        1. Detect, measure, and remove sources at the `freq_for_cib`
           frequency above the minimum SNR threshold
        2. (a) Detect, measure, and remove sources at the
               `freq_for_radio` frequency above the minimum SNR threshold
           (b) Identify all sources that were measured at both
               `freq_for_radio` and `freq_for_cib`, and calculate an
               average CIB spectral index from the sources that are
               brighter at the higher frequency
           (c) Identify all sources detected at the `freq_for_cib` but
               not at the `freq_for_radio`, and use the average CIB
               spectral index to extrapolate their fluxes to the
               `freq_for_radio` and remove them from that map
        3. Make a catalog of all sources that were detected at at least
           one of `freq_for_cib` or `freq_for_radio`. Identify sources
           that are brighter/dimmer at the higher frequency as CIB/radio.
        4. For each remaining map frequency `freq`:
           (a) Detect, measure, and remove sources at the `freq` above
               the minimum SNR threshold
           (b) Identify all sources that were measured at both
               `freq_for_cib`/`freq_for_radio`, and calculate an average
               CIB/radio spectral index from the sources that are
               brighter/dimmer at the higher frequency
           (c) Identify all sources in the catalog from step 3 but not at
               the `freq`, and use the average CIB/radio spectral index
               to extrapolate the CIB/radio fluxes to the `freq` and
               remove them from that map.

        Re-measuring mis-subtracted sources:
        After removing all sources detected above the minimum SNR at a
        given frequency and dim sources detected at a different frequency
        (if applicable), we once again apply the matched filter to the
        map, and identify mis-subtracted sources as any point source that
        is located in a region with |SNR| exceeding the minimum
        threshold.
        This is done by "un-subtracting" those sources from the map (or,
        equivalently, removing them from the detected source catalog for
        that map, and then only subtracting the remaining sources in the
        catalog from the original map). We then repeat the process of
        iteratively detecting, measuring, and removing sources on this
        new map (using the `remeasure_snr_threshold_list`), in order to
        more accurately measure the positions and fluxes of the
        "mis-subtracted" sources.
        The entire "re-measuring" procedure is repeated a maximum of
        `max_ntimes_remeasure` times.
        Note that there are typically not many mis-subtracted sources
        at the `freq_for_cib`, but there are typically many
        mis-subtracted sources in the maps after the "extrapolation" step
        (due to, e.g., sources appearing more blended at lower
        frequencies).
        """
        self.freqs = list(imaps.keys())
        self.output_dir = output_dir
        self.verbose = verbose
        self.log = log

        # templates, etc. for error messages:
        freq_errmsg = ("The keys (frequencies, in GHz) of the `{}` dictionary must be the "
                       "same as the mapfrequencies used as keys in the  `imaps` dictionary. "
                       "The following keys are missing from the `{}` dictionary: {}")
        _missing_freqs = lambda freqs: [freq for freq in self.freqs if (freq not in freqs)]
        file_errmsg = "The following files that were passed as values in the `{}` dictionary do not exist: {}"
        _missing_files = lambda fnames: {freq: path for (freq, path) in fnames.items() if (not os.path.exists(path))}
        _files_msg = lambda fnames: '\n\t'.join(['', *[f"{freq} GHz: {path}" for (freq, path) in fnames.items()]])

        # input maps:
        self.map_shape = map_shape
        self.map_wcs = map_wcs
        if all([isinstance(imap, enmap.ndmap) for imap in imaps.values()]):
            self.imap_fnames = None
            self.imaps = deepcopy(imaps)
            self.map_shape = self.imaps[self.freqs[0]].shape
            self.map_wcs = self.imaps[self.freqs[0]].wcs
        else:
            missing_fnames = _missing_files(imaps) # files passed that don't exist
            if len(missing_fnames) > 0:
                raise FileNotFoundError(file_errmsg.format('imaps', _files_msg(missing_fnames)))
            self.imap_fnames = imaps
            self.imaps = None
            if (self.map_shape is None) or (self.map_wcs is None): # need to load a map anyway to get its geometry
                imap = enmap.read_map(self.imap_fnames[self.freqs[0]])
                self.map_shape = imap.shape
                self.map_wcs = imap.wcs

        # make sure we have a beam for each freq
        self.beam_fwhm = beam_fwhms
        missing_beam_freqs = _missing_freqs(beam_fwhms)
        if len(missing_beam_freqs) > 0:
            raise ValueError(freq_errmsg.format('beam_fwhms', 'beam_fwhms', missing_beam_freqs))

        # dividing the maps into smaller patches:
        self.patch_apod_width = patch_apod_width
        self.patches = fgmaps.MapPatches(self.map_shape, self.map_wcs, max_patch_size=max_patch_size,
                                  patch_apod_width=self.patch_apod_width, map_apod_width=map_apod_width)
        patch_width_info = f'{simutils.round_str(self.patches.patch_width, n=3)}'
        patch_height_info = f'{simutils.round_str(self.patches.patch_height, n=3)}'
        patch_area_info = f'{patch_width_info}x{patch_height_info}deg'
        self.fgclean_dir_for_patches = os.path.join(self.output_dir, f'fgclean_{patch_area_info}_patches')

        # FG cleaning: source and cluster subtraction
        self.subtract_sources = subtract_sources
        self.subtract_clusters = subtract_clusters
        if not (self.subtract_clusters or self.subtract_sources):
            raise ValueError(f"`{subtract_sources = }` and `{subtract_clusters = }`, so there is nothing to do.")

        self.sources_snr_threshold_list = sources_snr_threshold_list
        self.sources_to_mask_before_fgclean_catalog = sources_to_mask_before_fgclean_catalog
        self.sources_to_subtract_before_fgclean_maps = None # subtract these sources from input maps before masking
        self.beam_solid_angle_funcs = {} if (beam_solid_angle_funcs is None) else beam_solid_angle_funcs
        self.calc_beam_solid_angle_funcs = calc_beam_solid_angle_funcs
        self.sources_iter_info = None
        if self.subtract_sources:
            min_snr = np.min(self.sources_snr_threshold_list)
            max_snr = np.max(self.sources_snr_threshold_list)
            num_thresholds = len(self.sources_snr_threshold_list)
            self.sources_iter_info = f'{num_thresholds}SNRs{simutils.round_str(min_snr)}to{simutils.round_str(max_snr)}'
            if len(self.freqs) > 1:
                extrap_info_list = ['extrap']
                radio_freq = min(self.freqs) if (freq_for_radio is None) else freq_for_radio
                extrap_info_list.append(f'{round(radio_freq):03d}radio')
                cib_freq = max(self.freqs) if (freq_for_cib is None) else freq_for_cib
                extrap_info_list.append(f'{round(cib_freq):03d}cib')
                extrap_info = ''.join(extrap_info_list)
                self.sources_iter_info = f'{self.sources_iter_info}{extrap_info}'

        self.clusters_snr_threshold_list = clusters_snr_threshold_list
        self.clusters_to_mask_after_fgclean_catalog = clusters_to_mask_after_fgclean_catalog
        self.cluster_profiles = cluster_profiles
        self.clusters_iter_info = None
        if self.subtract_clusters:
            default_cluster_profiles = fgfilters.get_default_gauss_cluster_profiles_dict()
            default_cluster_profile_names = list(default_cluster_profiles.keys())
            if self.cluster_profiles is None: # use default set of profiles
                self.cluster_profiles = default_cluster_profiles
            min_snr = np.min(self.clusters_snr_threshold_list)
            max_snr = np.max(self.clusters_snr_threshold_list)
            num_thresholds = len(self.clusters_snr_threshold_list)
            self.clusters_iter_info = f'{num_thresholds}SNRs{simutils.round_str(min_snr)}to{simutils.round_str(max_snr)}'
            if cluster_profiles_info is not None:
                self.clusters_iter_info = f'{self.clusters_iter_info}{cluster_profiles_info}'

        # maps of noise used in the matched filter calculations:
        self.noise_sims_for_filters = None
        self.noise_maps_for_source_filters = None
        self.noise_maps_for_source_filters_fnames = None
        self.noise_maps_for_cluster_filters = None
        self.noise_maps_for_cluster_filters_fnames = None
        self.noise_maps_for_source_mask_filters = None
        self.noise_maps_for_source_mask_filters_fnames = None
        self.noise_maps_for_source_mask_filters_freqs = []
        if hd_noise_sims_for_filters_dir is not None:
            noise_maps_kwargs = {**noise_maps_kwargs, 'make_output_dirs': mpi.is_rank0}
            self.noise_sims_for_filters = fgfilters.NoiseSimsForFilters(hd_noise_sims_for_filters_dir, **noise_maps_kwargs)
        else:
            dict_name = lambda name : f'noise_maps_for_{name}_filters' # for error messages
            # noise maps used to calculate point source filter:
            if self.subtract_sources:
                if noise_maps_for_source_filters is not None:
                    missing_freqs = _missing_freqs(noise_maps_for_source_filters)
                    if len(missing_freqs) > 0:
                        raise ValueError(freq_errmsg.format(dict_name('source'), dict_name('source'), missing_freqs))
                    elif all([isinstance(nmap, enmap.ndmap) for nmap in noise_maps_for_source_filters.values()]):
                        self.noise_maps_for_source_filters = deepcopy(noise_maps_for_source_filters)
                    else:
                        self.noise_maps_for_source_filters = {}
                        self.noise_maps_for_source_filters_fnames = noise_maps_for_source_filters
                        missing_fnames = _missing_files(noise_maps_for_source_filters) # files passed that don't exist
                        if len(missing_fnames) > 0:
                            raise FileNotFoundError(file_errmsg.format(dict_name('source'), _files_msg(missing_fnames)))
                if noise_maps_for_source_mask_filters is not None:
                    self.noise_maps_for_source_mask_filters_freqs = list(noise_maps_for_source_mask_filters.keys())
                    if all([isinstance(nmap, enmap.ndmap) for nmap in noise_maps_for_source_mask_filters.values()]):
                        self.noise_maps_for_source_mask_filters = deepcopy(noise_maps_for_source_mask_filters)
                    elif noise_maps_for_source_mask_filters is not None:
                        self.noise_maps_for_source_mask_filters = {}
                        self.noise_maps_for_source_mask_filters_fnames = noise_maps_for_source_mask_filters
                        missing_fnames = _missing_files(noise_maps_for_source_mask_filters) # files passed that don't exist
                        if len(missing_fnames) > 0:
                            raise FileNotFoundError(file_errmsg.format(dict_name('source_mask'), _files_msg(missing_fnames)))
            # noise maps used to calculate cluster filters:
            if self.subtract_clusters and (noise_maps_for_cluster_filters is not None):
                missing_freqs = _missing_freqs(noise_maps_for_cluster_filters)
                if len(missing_freqs) > 0:
                    raise ValueError(freq_errmsg.format(dict_name('cluster'), dict_name('cluster'), missing_freqs))
                elif all([isinstance(nmap, enmap.ndmap) for nmap in noise_maps_for_cluster_filters.values()]):
                    self.noise_maps_for_cluster_filters = deepcopy(noise_maps_for_cluster_filters)
                else:
                    self.noise_maps_for_cluster_filters = {}
                    self.noise_maps_for_cluster_filters_fnames = noise_maps_for_cluster_filters
                    missing_fnames = _missing_files(noise_maps_for_cluster_filters) # files passed that don't exist
                    if len(missing_fnames) > 0:
                        raise FileNotFoundError(file_errmsg.format(dict_name('cluster'), _files_msg(missing_fnames)))

        # will hold instances of the fg-cleaning classes for each smaller patch:
        self.ptsrclibs = {}
        self.clusterlibs = {}
        # and instances of classes for fg-cleaning files (to obtain file names, etc. without needing to pass in maps):
        self.ptsrcfiles = {}
        self.clusterfiles = {}
        for patch_num in self.patches.patch_nums:
            self.ptsrcfiles[patch_num] = fgclean_sources.FGCleanMFSourcesFiles(self.patch_sources_dir(patch_num=patch_num, make_dir=False),
                                                                  self.freqs, verbose=self.verbose, log=self.log)
            self.clusterfiles[patch_num] = fgclean_clusters.FGCleanClustersFiles(self.patch_clusters_dir(patch_num=patch_num, make_dir=False),
                                                                   self.freqs, self.beam_fwhm,
                                                                   make_output_dirs=False, verbose=self.verbose, log=self.log)

        # store kwargs that will be passed to fg-cleaning classes
        self.sources_kwargs = {'smooth_p2d_npix': smooth_p2d_npix, 'rms_gw': sources_rms_gw, 'rms_niter': rms_niter, 'rms_nsigma': rms_nsigma,
                               'min_num_iter_sources_per_snr': min_num_iter_sources_per_snr, 'apply_apod': False,
                               'min_snr_for_spectral_index': min_snr_for_spectral_index, 'nsigma_for_spectral_index': nsigma_for_spectral_index,
                               'remeasure_snr_threshold_list': remeasure_snr_threshold_list, 'max_ntimes_remeasure': max_ntimes_remeasure,
                               'extrapolate_dim_sources': extrapolate_dim_sources, 'remeasure_missubtracted_sources': remeasure_missubtracted_sources,
                               'freq_for_radio': freq_for_radio, 'freq_for_cib': freq_for_cib,
                               'verbose': self.verbose, 'log': self.log}
        self.clusters_kwargs = {'iter_match_radius': clusters_iter_match_radius, 'smooth_p2d_npix': smooth_p2d_npix,
                                'rms_gw': clusters_rms_gw, 'rms_niter': rms_niter, 'rms_nsigma': rms_nsigma,
                                'verbose': self.verbose, 'log': self.log}

        # set defaults for mask
        self.default_mask_kwargs = {'mask_sources': True, 'mask_clusters': True, 'mask_apod_width': fgi.mask_apod_width,
                                    'mask_snr_threshold': fgi.mask_snr_threshold, 'mask_abs_snr': fgi.mask_abs_snr,
                                    'mask_snr_threshold_below': fgi.mask_snr_threshold_below, 'patch_num': None}

        # create directories:
        if mpi.is_rank0:
            utils.mkdir(self.output_dir)
            utils.mkdir(self.fgclean_dir_for_patches)
            self.results_dir(make_dir=True)
            self.fgcleaned_maps_dir(make_dir=True)
            self.mask_dir(make_dir=True)
        mpi.comm.barrier()


    ####################################################
    ############### running FG cleaning: ###############
    ####################################################

    def run_fgclean(self, patch_nums=None, combine_patches=True,
                    make_masks=True, mask_freqs=None,
                    save_maps_after_subtraction=True,
                    save_subtracted_sources_maps=True,
                    save_subtracted_clusters_maps=True,
                    **kwargs):
        '''
        note : if `make_masks=True`, masks will be saved but won't be returned - get them with `get_mask` or `get_masks`

        kwargs are for the mask
        '''
        self.run_fgclean_on_patches(patch_nums=patch_nums, make_masks=make_masks, mask_freqs=mask_freqs, **kwargs)
        if self.all_patches_fgcleaned() and combine_patches:
            if mpi.is_rank0:
                self.infomsg(f"all patches have been FG cleaned")
            # save catalogs (for full maps) after fg cleaning:
            self.save_combined_catalogs_from_patches()
            # get the (full) maps after fg cleaning:
            save_maps = any([save_maps_after_subtraction, save_subtracted_sources_maps, save_subtracted_clusters_maps])
            if save_maps and mpi.is_rank0:
                self.get_fgcleaned_maps(save=save_maps_after_subtraction,
                                        save_subtracted_sources_maps=save_subtracted_sources_maps,
                                        save_subtracted_clusters_maps=save_subtracted_clusters_maps)
            # masks (for the full maps):
            if make_masks and mpi.is_rank0:
                mask_freqs = self.freqs if (mask_freqs is None) else self._validate_freqs(mask_freqs)
                mask_kwargs = self._get_mask_kwargs(**kwargs)
                if not all([os.path.exists(self.get_mask_fname(freq, **mask_kwargs)) for freq in mask_freqs]):
                    self.infomsg(f"making masks")
                    masks = self.get_masks(freqs=mask_freqs, save=True, **mask_kwargs)
        mpi.comm.barrier()


    # ----- maps after FG cleaning: -----

    def fgcleaned_map_fname(self, freq, subtract_sources=True, subtract_clusters=True):
        freq = self._validate_freq(freq)
        fname_info = [f'{round(freq):03d}GHz', 'fgcleaned']
        if subtract_sources and self.subtract_sources:
            fname_info.append('sources')
        if subtract_clusters and self.subtract_clusters:
            fname_info.append('clusters')
        fname_root = '_'.join(fname_info)
        fname = os.path.join(self.fgcleaned_maps_dir(), f'{fname_root}.fits')
        return fname


    def get_fgcleaned_maps(self, freqs=None, subtract_sources=True, subtract_clusters=True, save=False,
                           save_subtracted_sources_maps=False, save_subtracted_clusters_maps=False,):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        fnames = {freq: self.fgcleaned_map_fname(freq, subtract_sources=subtract_sources, subtract_clusters=subtract_clusters) for freq in freqs}

        if all([os.path.exists(fnames[freq]) for freq in freqs]):
            fgcleaned_maps = {}
            for freq in freqs:
                fgcleaned_maps[freq] = enmap.read_map(fnames[freq])
        else:
            # if we are subtracting all detected sources, this will include any masked before FG cleaning;
            # otherwise, remove any sources that were masked before FG cleaning (and leave the rest in):
            subtract_sources_to_mask_before_fgclean = not (subtract_sources and self.subtract_sources)
            imaps = self.get_imaps(subtract_sources_to_mask_before_fgclean=subtract_sources_to_mask_before_fgclean)
            fgcleaned_maps = {freq: imaps[freq].copy() for freq in freqs}
            if subtract_sources and self.subtract_sources:
                subtracted_sources_maps = self.get_maps_of_subtracted_sources(freqs=freqs, pixwin=True, beam=True,
                                                                              save=save_subtracted_sources_maps)
                for freq in freqs:
                    fgcleaned_maps[freq] -= subtracted_sources_maps[freq]
            if subtract_clusters and self.subtract_clusters:
                subtracted_clusters_maps = self.get_maps_of_subtracted_clusters(freqs=freqs, pixwin=True, beam=True,
                                                                                save=save_subtracted_clusters_maps)
                for freq in freqs:
                    fgcleaned_maps[freq] -= subtracted_clusters_maps[freq]
            if save:
                for freq in freqs:
                    enmap.write_map(fnames[freq], fgcleaned_maps[freq])
                    self.infomsg(f"saved {simutils.round_str(freq)} GHz map after FG cleaning to {fnames[freq]}")
        return fgcleaned_maps


    # --- maps of sources/clusters subtracted from full maps: ---

    def get_map_of_subtracted_fgs(self, freq, pixwin=True, beam=True, save=False):
        freq = self._validate_freq(freq)
        omap = self.get_maps_of_subtracted_fgs(freqs=[freq], pixwin=pixwin, beam=beam, save=save)[freq]
        return omap


    def get_maps_of_subtracted_fgs(self, freqs=None, pixwin=True, beam=True, save=False):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        omaps = {freq: enmap.zeros(self.map_shape, self.map_wcs) for freq in freqs}
        if self.subtract_sources:
            subtracted_sources_maps = self.get_maps_of_subtracted_sources(freqs=freqs, pixwin=pixwin, beam=beam, save=save)
            for freq in freqs:
                omaps[freq] += subtracted_sources_maps[freq]
        if self.subtract_clusters:
            subtracted_clusters_maps = self.get_maps_of_subtracted_clusters(freqs=freqs, pixwin=pixwin, beam=beam, save=save)
            for freq in freqs:
                omaps[freq] += subtracted_clusters_maps[freq]
        return omaps


    def get_map_of_subtracted_sources(self, freq, pixwin=True, beam=True, save=False):
        fname = self.get_map_of_subtracted_sources_fname(freq, pixwin=pixwin, beam=beam)
        if os.path.exists(fname):
            omap = enmap.read_map(fname)
        else:
            catalog = self.get_catalog_of_subtracted_sources(freq)
            omap = maps.make_src_map(self.map_shape, self.map_wcs, [catalog], freq)
            # note: don't need to apodize, b/c srcs were found in apodized map
            if pixwin:
                omap = enmap.apply_window(omap)
            if beam:
                omap = maps.convolve_sim_with_beam(omap, self.beam_fwhm[freq])
            if save:
                enmap.write_map(fname, omap)
        return omap


    def get_maps_of_subtracted_sources(self, freqs=None, pixwin=True, beam=True, save=False):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        omaps = {}
        for freq in freqs:
            omaps[freq] = self.get_map_of_subtracted_sources(freq, pixwin=pixwin, beam=beam, save=save)
        return omaps


    def get_map_of_subtracted_sources_fname(self, freq, pixwin=True, beam=True):
        freq = self._validate_freq(freq)
        fname_info = [f'{int(freq):03d}GHz_subtracted_sources']
        if pixwin:
            fname_info.append('pixwin')
        if beam:
            fname_info.append(f'beam{simutils.round_str(self.beam_fwhm[freq])}arcmin')
        fname_root = '_'.join(fname_info)
        fname = os.path.join(self.results_dir(), f'{fname_root}.fits')
        return fname


    def get_map_of_subtracted_clusters(self, freq, pixwin=True, beam=True, save=False):
        fname = self.get_map_of_subtracted_clusters_fname(freq, pixwin=pixwin, beam=beam)
        if os.path.exists(fname):
            omap = enmap.read_map(fname)
        else:
            catalog = self.get_catalog_of_subtracted_clusters()
            omap = fgmaps.make_cluster_sim_from_catalog(self.map_shape, self.map_wcs, catalog, self.cluster_profiles, freq,
                                                 convolve_pixwin=pixwin, convolve_beam=beam, beam_fwhm=self.beam_fwhm[freq])
            if save:
                enmap.write_map(fname, omap)
        return omap


    def get_maps_of_subtracted_clusters(self, freqs=None, pixwin=True, beam=True, save=False):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        fnames = {freq: self.get_map_of_subtracted_clusters_fname(freq, pixwin=pixwin, beam=beam) for freq in freqs}
        saved_freqs = [freq for freq in freqs if os.path.exists(fnames[freq])]
        unsaved_freqs = [freq for freq in freqs if (not os.path.exists(fnames[freq]))]

        omaps = {}
        if len(unsaved_freqs) > 0:
            cluster_catalog = self.get_catalog_of_subtracted_clusters()
            omaps = fgmaps.make_cluster_sims_from_catalog(self.map_shape, self.map_wcs, cluster_catalog,
                                                   self.cluster_profiles, unsaved_freqs,
                                                   convolve_pixwin=pixwin, convolve_beam=beam, beam_fwhms=self.beam_fwhm)
            if save:
                for freq in unsaved_freqs:
                    enmap.write_map(fnames[freq], omaps[freq])
        for freq in saved_freqs:
            omaps[freq] = enmap.read_map(fnames[freq])
        return omaps


    def get_map_of_subtracted_clusters_fname(self, freq, pixwin=True, beam=True):
        freq = self._validate_freq(freq)
        fname_info = [f'{int(freq):03d}GHz_subtracted_clusters']
        if pixwin:
            fname_info.append('pixwin')
        if beam:
            fname_info.append(f'beam{simutils.round_str(self.beam_fwhm[freq])}arcmin')
        fname_root = '_'.join(fname_info)
        fname = os.path.join(self.results_dir(), f'{fname_root}.fits')
        return fname


    # ----- catalogs of all subtracted sources/clusters from full maps: -----

    def catalog_of_subtracted_sources_fname(self, freq):
        fname = os.path.join(self.results_dir(), f'{round(freq):03d}GHz_subtracted_sources.csv')
        return fname


    def get_catalog_of_subtracted_sources(self, freq):
        fname = self.catalog_of_subtracted_sources_fname(freq)
        if not os.path.exists(fname):
            self.save_combined_catalogs_from_patches()
        catalog = fgcatalogs.load_catalog(fname)
        return catalog


    def catalog_of_subtracted_clusters_fname(self):
        fname = os.path.join(self.results_dir(), 'subtracted_clusters.csv')
        return fname


    def get_catalog_of_subtracted_clusters(self):
        fname = self.catalog_of_subtracted_clusters_fname()
        if not os.path.exists(fname):
            self.save_combined_catalogs_from_patches()
        catalog = fgcatalogs.load_catalog(fname)
        return catalog


    # --- combining the catalogs from FG cleaning each patch: ---

    def _prepare_patch_catalog(self, catalog, patch_num, add_patch_idx_col=True, include_apodized_region=True):
        catalog = self.patches.trim_patch_catalog_to_use(catalog, patch_num, include_apodized_region=include_apodized_region)
        catalog['patch_num'] = patch_num
        if add_patch_idx_col:
            catalog['patch_idx'] = catalog['idx'].values.copy()
        return catalog


    def _combine_patch_catalogs(self, catalogs):
        catalog = fgcatalogs.combine_catalogs(catalogs)
        catalog = catalog.sort_values('SNR', ascending=False)
        catalog['idx'] = list(range(len(catalog)))
        return catalog


    def save_combined_catalogs_from_patches(self):
        if mpi.is_rank0:
            if self.subtract_sources:
                catalog_fnames = {freq: self.catalog_of_subtracted_sources_fname(freq) for freq in self.freqs}
                catalog_freqs = [freq for freq in self.freqs if not os.path.exists(catalog_fnames[freq])]
                if len(catalog_freqs) > 0:
                    self.patches._calc_pixel_info_for_patches()
                for freq in catalog_freqs:
                    source_catalogs = {}
                    for patch_num in self.patches.patch_nums:
                        patch_source_catalog = self._patch_catalog_of_subtracted_sources(freq, patch_num=patch_num)
                        source_catalogs[patch_num] = self._prepare_patch_catalog(patch_source_catalog, patch_num)
                    source_catalog = self._combine_patch_catalogs([source_catalogs[patch_num] for patch_num in self.patches.patch_nums])
                    source_catalog.to_csv(catalog_fnames[freq])
                    self.infomsg(f"saved catalog of subtracted {freq} GHz sources to {catalog_fnames[freq]}")
            if self.subtract_clusters:
                catalog_fname = self.catalog_of_subtracted_clusters_fname()
                if not os.path.exists(catalog_fname):
                    cluster_catalogs = {}
                    self.patches._calc_pixel_info_for_patches()
                    for patch_num in self.patches.patch_nums:
                        patch_cluster_catalog = self._patch_catalog_of_subtracted_clusters(patch_num=patch_num)
                        cluster_catalogs[patch_num] = self._prepare_patch_catalog(patch_cluster_catalog, patch_num)
                    cluster_catalog = self._combine_patch_catalogs([cluster_catalogs[patch_num] for patch_num in self.patches.patch_nums])
                    cluster_catalog.to_csv(catalog_fname)
                    self.infomsg(f"saved catalog of subtracted clusters to {catalog_fname}")
        mpi.comm.barrier()


    # ----- measured spectral index (for a given patch) -----

    def get_spectral_index(self, freq, component, patch_num=0):
        try:
            index_mean, index_std = self.ptsrcfiles[patch_num].load_spectral_index(freq, component)
        except FileNotFoundError:
            self._init_patch_ptsrclib(patch_num=patch_num)
            index_mean, index_std = self.ptsrclibs[patch_num].get_spectral_index(freq, component)
        return index_mean, index_std


    def get_spectral_index_catalog(self, freq, extrap_freq, component=None, patch_num=0):
        try:
            catalog = self.ptsrcfiles[patch_num].load_spectral_index_catalog(freq, extrap_freq, component=component)
        except FileNotFoundError:
            self._init_patch_ptsrclib(patch_num=patch_num)
            catalog = self.ptsrclibs[patch_num].get_spectral_index_catalog(freq, extrap_freq, component=component)
        return catalog



    ###############################################################
    ############### running FG cleaning on patches: ###############
    ###############################################################

    def run_fgclean_on_patches(self, patch_nums=None, make_masks=True, mask_freqs=None, **kwargs):
        '''kwargs are for mask'''
        patch_nums = self.patches.patch_nums if (patch_nums is None) else patch_nums
        patch_nums_indices = mpi.distribute(len(patch_nums), mpi.size, mpi.rank)
        self.infomsg(f"starting FG cleaning for patch numbers {[patch_nums[i] for i in patch_nums_indices]}")
        for i in patch_nums_indices:
            patch_num = patch_nums[i]
            self.run_fgclean_on_patch(patch_num=patch_num, make_masks=make_masks, mask_freqs=mask_freqs, **kwargs)
        mpi.comm.barrier()


    def run_fgclean_on_patch(self, patch_num=0, make_masks=True, mask_freqs=None, **kwargs):
        '''
        kwargs are for the masks
        '''
        source_catalogs_saved = all([os.path.exists(self._patch_catalog_of_subtracted_sources_fname(freq, patch_num=patch_num)) for freq in self.freqs])
        cluster_catalog_saved = os.path.exists(self._patch_catalog_of_subtracted_clusters_fname(patch_num=patch_num))
        if self.subtract_clusters and (not cluster_catalog_saved):
            # this will first run source-subtraction, if necessary:
            self.run_cluster_subtraction_on_patch(patch_num=patch_num)
        elif self.subtract_sources and (not source_catalogs_saved):
            self.run_source_subtraction_on_patch(patch_num=patch_num)
        if make_masks:
            mask_kwargs = {**self._get_mask_kwargs(**kwargs), 'patch_num': patch_num}
            self.get_patch_masks(freqs=mask_freqs, **mask_kwargs)
        self.infomsg(f"patch {patch_num} is FG cleaned")


    def run_source_subtraction_on_patch(self, patch_num=0):
        self.infomsg(f"{patch_num = :2d} : starting source subtraction")
        self._init_patch_ptsrclib(patch_num=patch_num)
        subtracted_sources_catalogs, sims_after_source_subtraction = self.ptsrclibs[patch_num].run_source_subtraction(save_subtracted_sources_maps=False,
                                                                                                                      save_intermediate_subtracted_sources_maps=False)
        self.infomsg(f"{patch_num = :2d} : done with source subtraction")
        return subtracted_sources_catalogs, sims_after_source_subtraction


    def run_cluster_subtraction_on_patch(self, patch_num=0):
        self.infomsg(f"{patch_num = :2d} : starting cluster subtraction")
        self._init_patch_clusterlib(patch_num=patch_num)
        subtracted_clusters_catalog, sims_after_cluster_subtraction = self.clusterlibs[patch_num].run_cluster_subtraction(save_subtracted_clusters_maps=False)
        self.infomsg(f"{patch_num = :2d} : done with cluster subtraction")
        return subtracted_clusters_catalog, sims_after_cluster_subtraction


    def _init_patch_ptsrclib(self, patch_num=0):
        if patch_num not in self.ptsrclibs:
            imaps = self.get_imaps_for_patch(patch_num=patch_num)
            noise_maps = self._get_patch_noise_maps_for_source_filters(patch_num=patch_num)
            beam_fwhms = self._patch_beam_fwhms(patch_num=patch_num)
            self.ptsrclibs[patch_num] = fgclean_sources.FGCleanMFSources(self.patch_sources_dir(patch_num=patch_num),
                                                            imaps, beam_fwhms, self.patch_apod_width,
                                                            noise_maps, self.sources_snr_threshold_list,
                                                            apod_window=self.get_apod_window_for_patch(patch_num=patch_num),
                                                            beam_solid_angle_funcs=self.beam_solid_angle_funcs,
                                                            calc_beam_solid_angle_funcs=self.calc_beam_solid_angle_funcs,
                                                            masks_for_filtered_maps=self._get_imap_masks(patch_num=patch_num),
                                                            **self.sources_kwargs)


    def _init_patch_clusterlib(self, patch_num=0):
        if patch_num not in self.clusterlibs:
            imaps = self._get_imaps_for_patch_cluster_subtraction(patch_num=patch_num)
            noise_maps = self._get_patch_noise_maps_for_cluster_filters(patch_num=patch_num)
            beam_fwhms = self._patch_beam_fwhms(patch_num=patch_num)
            self.clusterlibs[patch_num] = fgclean_clusters.FGCleanClusters(self.patch_clusters_dir(patch_num=patch_num),
                                                             imaps, beam_fwhms, self.patch_apod_width,
                                                             self.cluster_profiles, noise_maps,
                                                             self.clusters_snr_threshold_list,
                                                             mask_for_filtered_maps=self._get_sn_map_mask_for_clusters(patch_num=patch_num),
                                                             **self.clusters_kwargs)


    def _patch_is_fgcleaned(self, patch_num=0):
        '''check whether fgcleaning has been done for the patch'''
        if self.subtract_sources:
            catalog_fnames = {freq: self.ptsrcfiles[patch_num].catalog_of_subtracted_sources_fname(freq) for freq in self.freqs}
            done_with_sources = all([os.path.exists(catalog_fnames[freq]) for freq in self.freqs])
        else:
            done_with_sources = True
        if self.subtract_clusters:
            done_with_clusters = os.path.exists(self.clusterfiles[patch_num].catalog_of_subtracted_clusters_fname())
        else:
            done_with_clusters = True
        fgcleaned = done_with_sources and done_with_clusters
        return fgcleaned


    def all_patches_fgcleaned(self):
        return all([self._patch_is_fgcleaned(patch_num) for patch_num in self.patches.patch_nums])


    # ----- output of running FG cleaning on patches -----

    def _get_fgcleaned_patch_maps(self, patch_num=0, freqs=None):
        if self.subtract_clusters:
            try:
                omaps = self.clusterfiles[patch_num].load_cluster_subtracted_sims(freqs=freqs)
            except FileNotFoundError:
                self._init_patch_clusterlib(patch_num=patch_num)
                omaps = self.clusterlibs[patch_num].get_cluster_subtracted_sims(freqs=freqs)
        else:
            try:
                omaps = self.ptsrcfiles[patch_num].load_source_subtracted_sims(freqs=freqs)
            except FileNotFoundError:
                self._init_patch_ptsrclib(patch_num=patch_num)
                omaps = self.ptsrclibs[patch_num].get_source_subtracted_sims(freqs=freqs)
        return omaps


    def _patch_catalog_of_subtracted_sources_fname(self, freq, patch_num=0):
        fname = self.ptsrcfiles[patch_num].catalog_of_subtracted_sources_fname(freq)
        return fname


    def _patch_catalog_of_subtracted_sources(self, freq, patch_num=0):
        fname = self._patch_catalog_of_subtracted_sources_fname(freq, patch_num=patch_num)
        if os.path.exists(fname):
            catalog = fgcatalogs.load_catalog(fname)
        else:
            self._init_patch_ptsrclib(patch_num=patch_num)
            catalog = self.ptsrclibs[patch_num].get_catalog_of_subtracted_sources(freq)
        catalog = self._add_sources_masked_before_fgclean_to_catalog(catalog, freq, patch_num=patch_num)
        return catalog


    def _patch_catalog_of_subtracted_clusters_fname(self, patch_num=0):
        fname = self.clusterfiles[patch_num].catalog_of_subtracted_clusters_fname()
        return fname


    def _patch_catalog_of_subtracted_clusters(self, patch_num=0):
        fname = self._patch_catalog_of_subtracted_clusters_fname(patch_num=patch_num)
        if os.path.exists(fname):
            catalog = fgcatalogs.load_catalog(fname)
        else:
            self._init_patch_clusterlib(patch_num=patch_num)
            catalog = self.clusterlibs[patch_num].get_catalog_of_subtracted_clusters()
        return catalog


    ######################################################
    ############### inputs to FG cleaning: ###############
    ######################################################

    def get_imaps(self, subtract_sources_to_mask_before_fgclean=True):
        '''
        load the input maps into a dictionary
        '''
        if self.imaps is None:
            self.imaps = {}
            for freq in self.freqs:
                self.imaps[freq] = enmap.read_map(self.imap_fnames[freq])
        imaps = {freq: imap for (freq, imap) in self.imaps.items()}
        if (self.sources_to_mask_before_fgclean_catalog is not None) and subtract_sources_to_mask_before_fgclean:
            bright_srcs = self._get_sources_to_subtract_before_fgclean_maps()
            if bright_srcs is not None:
                for freq in self.freqs:
                    imaps[freq] = imaps[freq].copy() - bright_srcs[freq].copy()
        return imaps


    def get_imaps_for_patch(self, patch_num=0, apod=True, freqs=None):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        imaps = self.get_imaps()
        shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)
        masks = self._get_imap_masks(patch_num=patch_num)
        window = 1 if (not apod) else self.get_apod_window_for_patch(patch_num=patch_num)
        patch_imaps = {}
        for freq in freqs:
            patch_imaps[freq] = masks[freq] * enmap.project(imaps[freq].copy(), shape, wcs) * window
        return patch_imaps


    def get_apod_window_for_patch(self, patch_num=0, padded=True, save=False):
        width, height = self.patches.get_patch_dimensions(patch_num=patch_num, padded=padded)
        fname = simutils.get_apod_window_fname(self.patch_apod_width, width, height, maps_dir=self.patch_fgclean_dir(patch_num=patch_num))
        if os.path.exists(fname):
            window = enmap.read_map(fname)
        else:
            shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=padded)
            window = maps.make_apod_window(shape, wcs, self.patch_apod_width)
            if save:
                enmap.write_map(fname, window)
        return window


    def _get_imaps_for_patch_cluster_subtraction(self, patch_num=0):
        '''
        if we are also running source subtraction, need source-subtracted maps to input into cluster subtraction
        '''
        if self.subtract_sources:
            try:
                imaps = self.ptsrcfiles[patch_num].load_source_subtracted_sims()
            except FileNotFoundError:
                self._init_patch_ptsrclib(patch_num=patch_num)
                imaps = self.ptsrclibs[patch_num].get_source_subtracted_sims()
        else:
            imaps = self.get_imaps_for_patch(patch_num=patch_num)
        return imaps


    def _get_beam_solid_angle_funcs(self, freqs=None):
        freqs = self.freqs if (freqs is None) else  self._validate_freqs(freqs)
        delta_dec = round(250 * maps.get_map_resolution(self.map_shape, self.map_wcs), 3)
        for freq in freqs:
            if freq not in self.beam_solid_angle_funcs:
                if self.calc_beam_solid_angle_funcs:
                    self.beam_solid_angle_funcs[freq] = iterfgclean.get_beam_solid_angle_per_dec_func(self.beam_fwhm[freq], delta_dec,
                                                                                          self.map_shape, self.map_wcs,
                                                                                          save=mpi.is_rank0,
                                                                                          save_dir=self.fgclean_dir_for_patches,
                                                                                          verbose=self.verbose, log=self.log)
                else:
                    self.beam_solid_angle_funcs[freq] = iterfgclean.get_const_beam_solid_angle_func(self.beam_fwhm[freq])
        return self.beam_solid_angle_funcs


    def _patch_beam_fwhms(self, patch_num=0):
        # calculate an "effective" beam fwhm for each patch (due to variation of pixel size w/ dec.)
        beam_solid_angle_funcs = self._get_beam_solid_angle_funcs()
        _, patch_dec_ctr = self.patches.get_patch_center_coords(patch_num, padded=True)
        beam_fwhms = {}
        for freq in self.freqs:
            beam_fwhms[freq] = fgutils.beam_solid_angle_to_fwhm(beam_solid_angle_funcs[freq](patch_dec_ctr))
        return beam_fwhms


    # ----- maps used for 2D noise power calculation in matched filters: -----

    def _get_patch_noise_maps_for_filters(self, noise_maps_for_filters, patch_num=0, freqs=None):
        freqs = self.freqs if (freqs is None) else  self._validate_freqs(freqs)
        shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)
        patch_window = self.get_apod_window_for_patch(patch_num=patch_num)
        noise_maps = {}
        for freq in freqs:
            noise_map = noise_maps_for_filters[freq].copy()
            # get map of correct shape
            if not maps.map_shape_is_equal(noise_map.shape, shape):
                res = maps.get_map_resolution(noise_map.shape, noise_map.wcs)
                ra_ctr, dec_ctr, _, _ = maps.get_map_ctr_extent(noise_map.shape, noise_map.wcs)
                width, height = self.patches.get_patch_dimensions(patch_num=patch_num, padded=True)
                _, noise_map_wcs = maps.get_shape_wcs(res, ra_ctr, dec_ctr, width, height=height)
                noise_map = enmap.project(noise_map, shape, noise_map_wcs)
            noise_maps[freq] = noise_map[:] * patch_window.copy()
        return noise_maps


    def _get_patch_noise_maps_for_source_filters(self, patch_num=0, freqs=None):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        if self.noise_sims_for_filters is not None:
            shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)
            noise_maps = self.noise_sims_for_filters.noise_maps_for_source_filters(shape, wcs, freqs=freqs)
        elif self.noise_maps_for_source_filters is not None:
            freqs_to_load = [freq for freq in freqs if (freq not in self.noise_maps_for_source_filters)]
            for freq in freqs_to_load:
                self.noise_maps_for_source_filters[freq] = enmap.read_map(self.noise_maps_for_source_filters_fnames[freq])
            noise_maps = self._get_patch_noise_maps_for_filters(self.noise_maps_for_source_filters, patch_num=patch_num, freqs=freqs)
        else: # use input maps
            noise_maps = self.get_imaps_for_patch(patch_num=patch_num, freqs=freqs)
        return noise_maps


    def _get_patch_noise_maps_for_cluster_filters(self, patch_num=0, freqs=None):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        if self.noise_sims_for_filters is not None:
            shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)
            noise_maps = self.noise_sims_for_filters.noise_maps_for_cluster_filters(shape, wcs, subtract_sources=self.subtract_sources, freqs=freqs)
        elif self.noise_maps_for_cluster_filters is not None:
            freqs_to_load = [freq for freq in freqs if (freq not in self.noise_maps_for_cluster_filters)]
            for freq in freqs_to_load:
                self.noise_maps_for_cluster_filters[freq] = enmap.read_map(self.noise_maps_for_cluster_filters_fnames[freq])
            noise_maps = self._get_patch_noise_maps_for_filters(self.noise_maps_for_cluster_filters, patch_num=patch_num, freqs=freqs)
        else: # use input maps
            noise_maps = self._get_imaps_for_patch_cluster_subtraction(patch_num=patch_num)
        return noise_maps



    def _get_patch_noise_maps_for_source_mask_filters(self, freq, patch_num=0):
        freq = self._validate_freq(freq)
        if self.noise_sims_for_filters is not None:
            shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)
            noise_maps = self.noise_sims_for_filters.noise_maps_for_source_mask_filters(shape, wcs, subtract_clusters=self.subtract_clusters, freqs=[freq])
        elif freq in self.noise_maps_for_source_mask_filters_freqs: # use map passed by user for sources mask
            if freq not in self.noise_maps_for_source_mask_filters:
                self.noise_maps_for_source_mask_filters[freq] = enmap.read_map(self.noise_maps_for_source_mask_filters_fnames[freq])
            noise_maps = self._get_patch_noise_maps_for_filters(self.noise_maps_for_source_mask_filters, patch_num=patch_num, freqs=[freq])
        elif self.noise_maps_for_source_filters is not None: # use map passed by user for point source filter
            noise_maps = self._get_patch_noise_maps_for_source_filters(patch_num=patch_num, freqs=[freq])
        else: # use the fgcleaned map
            noise_maps = self._get_fgcleaned_patch_maps(patch_num=patch_num, freqs=[freq])
        noise_map = noise_maps[freq]
        return noise_map


    # ----- maps of and masks applied to known bright sources subtracted before FG cleaning : -----


    def _get_sources_to_subtract_before_fgclean_maps(self):
        make_maps = all([self.subtract_sources, self.sources_to_mask_before_fgclean_catalog is not None,
                         self.sources_to_subtract_before_fgclean_maps is None])
        if make_maps:
            self.sources_to_subtract_before_fgclean_maps = {}
            src_maps = maps.make_src_maps(self.map_shape, self.map_wcs,
                                          self.sources_to_mask_before_fgclean_catalog,
                                          [round(freq) for freq in self.freqs]) # make sure freqs match col names
            for freq, beam_fwhm in self.beam_fwhm.items():
                src_map = maps.convolve_sim_with_beam(enmap.apply_window(src_maps[freq]), beam_fwhm)
                self.sources_to_subtract_before_fgclean_maps[freq] = src_map.copy()
        return self.sources_to_subtract_before_fgclean_maps


    def _get_sources_to_mask_before_fgclean_catalog(self, patch_num=None):
        if self.sources_to_mask_before_fgclean_catalog is not None:
            catalog = self.sources_to_mask_before_fgclean_catalog.copy()
            if patch_num is not None:
                catalog = self.patches.trim_patch_catalog(catalog, patch_num, padded=True)
            return catalog


    def _get_imap_masks(self, patch_num=None, freqs=None, apodize=True):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        catalog_for_mask = self._get_sources_to_mask_before_fgclean_catalog(patch_num=patch_num)
        if patch_num is not None:
            shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)
        else:
            shape = self.map_shape
            wcs = self.map_wcs
        masks = {}
        for freq in freqs:
            if (catalog_for_mask is not None) and (len(catalog_for_mask) > 0):
                radius_col = f'mask_radius_{simutils.round_str(freq)}GHz'
                if radius_col not in catalog_for_mask.columns.values:
                    radius_col = 'mask_radius'
                if apodize:
                    masks[freq] = fgmasks.make_mask_from_catalog(shape, wcs, catalog_for_mask, radius_col=radius_col)
                else:
                    masks[freq] = fgmasks.make_binary_mask_from_catalog(shape, wcs, catalog_for_mask, radius_col=radius_col)
            else:
                masks[freq] = enmap.ones(shape, wcs)
        return masks


    def _get_sn_map_mask_for_clusters(self, patch_num=None, apodize=True):
        catalog_for_mask = self._get_sources_to_mask_before_fgclean_catalog(patch_num=patch_num)
        if patch_num is not None:
            shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)
        else:
            shape = self.map_shape
            wcs = self.map_wcs
        if (catalog_for_mask is not None) and (len(catalog_for_mask) > 0):
            # use largest mask radius for each source
            all_mask_radius_cols = ['mask_radius', *[f'mask_radius_{simutils.round_str(freq)}GHz' for freq in self.freqs]]
            mask_radius_cols = [col for col in all_mask_radius_cols if (col in catalog_for_mask.columns.values)]
            mask_radius_to_use = []
            for i, row in catalog_for_mask.iterrows():
                mask_radius_to_use.append(2 * max([row[col] for col in mask_radius_cols]))
            catalog_for_mask['mask_radius'] = mask_radius_to_use
            catalog_for_mask['apod_width'] = 3 * catalog_for_mask['apod_width'].values
            if apodize:
                mask = fgmasks.make_mask_from_catalog(shape, wcs, catalog_for_mask)
            else:
                mask = fgmasks.make_binary_mask_from_catalog(shape, wcs, catalog_for_mask)
        else:
            mask = enmap.ones(shape, wcs)
        return mask


    def _add_sources_masked_before_fgclean_to_catalog(self, catalog, freq, patch_num=None):
        if self.sources_to_mask_before_fgclean_catalog is not None:
            masked_sources = self._get_sources_to_mask_before_fgclean_catalog(patch_num=patch_num)
            if len(masked_sources) > 0:
                masked_sources['fluxmJy'] = masked_sources[f'fluxmJy_{round(freq)}GHz'].values.copy()
                masked_sources_cols = ['RADeg', 'decDeg', 'fluxmJy']
                if 'component' in masked_sources.columns.values:
                    masked_sources_cols.append('component')
                masked_sources = masked_sources[masked_sources_cols].copy()
                # make sure we have all of the necessary columns:
                cols = catalog.columns.values
                masked_sources['method'] = 'external'
                masked_sources['remeasured'] = False
                if 'component' not in masked_sources_cols:
                    masked_sources['component'] = 'unknown'
                # add the `patch_num` for each source:
                if 'patch_num' in cols:
                    if patch_num is not None:
                        masked_sources['patch_num'] = patch_num
                    else: # catalog is for full map
                        ras = masked_sources['RADeg'].values
                        decs = masked_sources['decDeg'].values
                        masked_sources['patch_num'] = [self.get_patch_num(ra=r, dec=d) for (r, d) in zip(ras, decs)]
                # add an `idx` for each:
                idx0 = int(np.max(catalog['idx'].values))
                masked_sources['idx'] = list(range(idx0, idx0 + len(masked_sources)))
                # approximate beam-convolved source amplitude in uK:
                fluxes = masked_sources['fluxmJy'].values
                beam_solid_angle = 2 * np.pi * utils.fwhm2sigma(utils.arcmin2rad(self.beam_fwhm[freq]))**2
                masked_sources['deltaT_c'] = utils.mJy_per_str_to_uK(fluxes / beam_solid_angle, freq)
                # fill in remaining columns with placeholders:
                for col in ['SNR', 'numSigPix', 'snr_threshold']:
                    masked_sources[col] = 0
                other_cols = [col for col in ['iter_num', 'patch_idx', 'catalog_idx'] if (col in cols)]
                for col in other_cols:
                    masked_sources[col] = -1
                catalog = fgcatalogs.combine_catalogs([catalog, masked_sources])
        return catalog



    ######################################
    ############### masks: ###############
    ######################################

    def _get_mask_kwargs(self, **kwargs):
        mask_kwargs = self.default_mask_kwargs.copy()
        for key in mask_kwargs:
            if key in kwargs:
                mask_kwargs[key] = kwargs[key]
        return mask_kwargs


    # ----- masks after fg cleaning for full map: -----

    def get_masks(self, freqs=None, patch_num=None,
                  mask_sources=True, mask_clusters=True, mask_apod_width=1,
                  mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None, save=True):
        if patch_num is not None:
            masks = self.get_patch_masks(patch_num=patch_num, freqs=freqs, mask_sources=mask_sources, mask_clusters=mask_clusters,
                                         mask_apod_width=mask_apod_width, mask_snr_threshold=mask_snr_threshold,
                                         mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below, save=save)
        else:
            masks = {}
            freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
            # check if masks are saved ; if not, make them
            mask_fnames = {}
            mask_is_saved = {}
            for freq in freqs:
                mask_fnames[freq] = self.get_mask_fname(freq=freq, mask_sources=mask_sources, mask_clusters=mask_clusters, mask_apod_width=mask_apod_width,
                                                        mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
                mask_is_saved[freq] = os.path.exists(mask_fnames[freq])
            freqs_with_mask = [freq for freq in freqs if mask_is_saved[freq]] # mask is already saved for these freqs
            freqs_without_mask = [freq for freq in freqs if (not mask_is_saved[freq])] # need to make mask for these freqs
            # make the masks that haven't been saved:
            if len(freqs_without_mask) > 0:
                masks = self.make_masks(freqs=freqs_without_mask, mask_sources=mask_sources, mask_clusters=mask_clusters, mask_apod_width=mask_apod_width,
                                        mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
                if save:
                    for freq in freqs_without_mask:
                        enmap.write_map(mask_fnames[freq], masks[freq])
                        self.infomsg(f"saved {simutils.round_str(freq)} GHz mask to {mask_fnames[freq]}")
            # load any that were already saved:
            for freq in freqs_with_mask:
                masks[freq] = enmap.read_map(mask_fnames[freq])
        return masks


    def get_mask(self, freq, patch_num=None, mask_sources=True, mask_clusters=True, mask_apod_width=1,
                  mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None, save=True):
        if mask_sources and self.subtract_sources and (freq is None):
            raise ValueError(f"`{freq = }`. You must pass the map frequency (in GHz) to mask point sources.")
        mask = self.get_masks(freqs=[freq], patch_num=patch_num, mask_sources=mask_sources, mask_clusters=mask_clusters,
                              mask_apod_width=mask_apod_width, mask_snr_threshold=mask_snr_threshold,
                              mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below, save=save)[freq]
        return mask


    def make_masks(self, freqs=None, mask_sources=True, mask_clusters=True, mask_apod_width=1,
                   mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None):
        '''
        NOTE:
         - will run fg cleaning on all patches if necessary
         - edges of mask are not apodized (only holes are)
        '''
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        masks = {freq: enmap.zeros(self.map_shape, self.map_wcs) for freq in freqs}
        for patch_num in self.patches.patch_nums:
            # get binary masks:
            patch_masks = self.get_patch_masks(patch_num=patch_num, freqs=freqs, mask_apod_width=0,
                                               mask_sources=mask_sources, mask_clusters=mask_clusters,
                                               mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr,
                                               mask_snr_threshold_below=mask_snr_threshold_below)
            # get geometry for non-overlapping region of patch:
            patch_shape, patch_wcs = self.patches.get_patch_geometry_to_use(patch_num=patch_num)
            for freq in freqs:
                # cut out the region to use:
                patch_mask = enmap.project(patch_masks[freq], patch_shape, patch_wcs)
                # place it on the larger map:
                masks[freq] += enmap.project(patch_mask.copy(), self.map_shape, self.map_wcs)

        src_masks_for_fgcleaning = {freq: None for freq in freqs}
        large_nearby_clusters_mask = None 
        if mask_clusters and self.subtract_clusters:
            src_mask_for_fgcleaning = self._get_sn_map_mask_for_clusters(apodize=(mask_apod_width > 0))
            src_masks_for_fgcleaning = {freq: src_mask_for_fgcleaning for freq in freqs}
            large_nearby_clusters_mask = self._get_large_clusters_mask(apodize=(mask_apod_width > 0))
        elif mask_sources and self.subtract_sources:
            src_masks_for_fgcleaning = self._get_imap_masks(apodize=(mask_apod_width > 0))

        # apodize the holes in each mask:
        for freq in freqs:
            masks[freq] = fgmasks.apodize_mask(masks[freq], mask_apod_width)
            if src_masks_for_fgcleaning[freq] is not None:
                masks[freq] *= src_masks_for_fgcleaning[freq]
            if large_nearby_clusters_mask is not None:
                masks[freq] *= large_nearby_clusters_mask
        return masks


    def calculate_masked_map_fractions(self, freqs=None, max_unmasked_value=1, shape=None, wcs=None, as_percent=True, save_masks=True, **kwargs):
        '''
        calculate fraction of the map at each freq. that has been masked after FG cleaning

        max_unmasked_value : any pixels in mask with value less than this are considered "masked"
            - by default, all pixels w/ value less than 1 are masked (instead of all pixels that equal zero)
            - since holes are apodized, could set this to, e.g., 0.95 instead
            - must greater than zero and less than one
              - if less than zero, will use max_unmasked_value = 0.01 instead

        can pass shape, wcs to look at a specific sub-region of map

        kwargs are for the masks

        returns dict w/ keys = freq, values = masked fraction
        '''
        if max_unmasked_value > 1:
            max_unmasked_value = 1
        elif max_unmasked_value <= 0:
            max_unmasked_value = 0.01
        mult_fact = 100 if as_percent else 1
        mask_kwargs = self._get_mask_kwargs(**kwargs)
        masks = self.get_masks(freqs=freqs, save=save_masks, **mask_kwargs)
        masked_fractions = {}
        for freq, mask in masks.items():
            if (shape is not None) and (wcs is not None):
                mask = enmap.project(mask, shape, wcs)
            num_masked_pixels = np.less(mask, max_unmasked_value).sum()
            tot_num_pixels = mask.shape[-1] * mask.shape[-2]
            masked_frac = num_masked_pixels / tot_num_pixels
            if mpi.is_rank0:
                self.infomsg(f"{simutils.round_str(freq)} GHz: {100*masked_frac:.2f}% of the map is masked")
            masked_fractions[freq] = mult_fact * masked_frac
        return masked_fractions


    # ----- masks after fg cleaning for each patch: -----

    def get_patch_masks(self, patch_num=0, freqs=None, mask_sources=True, mask_clusters=True,
                        mask_apod_width=0,
                        mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None, save=False):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        # check if masks are saved ; if not, make them
        mask_fnames = {}
        mask_is_saved = {}
        for freq in freqs:
            mask_fnames[freq] = self.get_patch_mask_fname(patch_num=patch_num, freq=freq, mask_sources=mask_sources, mask_clusters=mask_clusters, mask_apod_width=mask_apod_width,
                                                          mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
            mask_is_saved[freq] = os.path.exists(mask_fnames[freq])
        freqs_with_mask = [freq for freq in freqs if mask_is_saved[freq]] # mask is already saved for these freqs
        freqs_without_mask = [freq for freq in freqs if (freq not in freqs_with_mask)] # need to make mask for these freqs

        masks = {}
        # make the masks that haven't been saved:
        if len(freqs_without_mask) > 0:
            freq_info = ', '.join([f'{simutils.round_str(freq)} GHz'])
            self.infomsg(f"making masks for {patch_num = } for {freq_info}")
            masks = self.make_patch_masks(patch_num=patch_num, freqs=freqs_without_mask, mask_sources=mask_sources, mask_clusters=mask_clusters, mask_apod_width=mask_apod_width,
                                          mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
            if save:
                for freq in freqs_without_mask:
                    enmap.write_map(mask_fnames[freq], masks[freq])
        # load any that were already saved:
        for freq in freqs_with_mask:
            masks[freq] = enmap.read_map(mask_fnames[freq])

        return masks


    def get_patch_mask(self, freq, mask_sources=True, mask_clusters=True,
                       mask_apod_width=0,
                       mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None, save=False):
        if mask_sources and self.subtract_sources and (freq is None):
            raise ValueError(f"`{freq = }`. You must pass the map frequency (in GHz) to mask point sources.")
        mask = self.get_patch_masks(patch_num=patch_num, freqs=[freq], mask_sources=mask_sources, mask_clusters=mask_clusters,
                                    mask_apod_width=mask_apod_width, mask_snr_threshold=mask_snr_threshold,
                                    mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below, save=save)[freq]
        return mask


    def make_patch_masks(self, patch_num=0, freqs=None, mask_sources=True, mask_clusters=True,
                         mask_apod_width=0,
                         mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None):
        self.infomsg(f"{patch_num = :2d} : making mask(s)")
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)

        masks = {freq: enmap.ones(shape, wcs) for freq in freqs}
        if mask_sources and self.subtract_sources:
            for freq in freqs:
                masks[freq] *= self._get_binary_patch_sources_mask(freq, patch_num=patch_num, mask_snr_threshold=mask_snr_threshold,
                                                                   mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
        if mask_clusters and self.subtract_clusters:
            binary_clusters_mask = self._get_binary_patch_clusters_mask(patch_num=patch_num, mask_snr_threshold=mask_snr_threshold,
                                                                        mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
            for freq in freqs:
                masks[freq] *= binary_clusters_mask

        if mask_apod_width > 0:
            # include the mask that was applied to the map before FG cleaning:
            src_masks_for_fgcleaning = {freq: None for freq in freqs}
            large_nearby_clusters_mask = None 
            if mask_clusters and self.subtract_clusters:
                large_nearby_clusters_mask = self._get_large_clusters_mask(patch_num=patch_num)
                src_mask_for_fgcleaning = self._get_sn_map_mask_for_clusters(patch_num=patch_num)
                src_masks_for_fgcleaning = {freq: src_mask_for_fgcleaning for freq in freqs}
            elif mask_sources and self.subtract_sources:
                src_masks_for_fgcleaning = self._get_imap_masks(patch_num=patch_num)
            for freq in freqs:
                masks[freq] = fgmasks.apodize_mask(masks[freq], mask_apod_width)
                if src_masks_for_fgcleaning[freq] is not None:
                    masks[freq] *= src_masks_for_fgcleaning[freq]
                if large_nearby_clusters_mask is not None:
                    masks[freq] *= large_nearby_clusters_mask

        return masks


    def _make_binary_patch_mask(self, masked_pixel_nums, patch_num=0):
        shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)
        x, y = maps.get_pixel_from_num(np.atleast_1d(masked_pixel_nums).astype(int), shape)
        mask = enmap.ones(shape, wcs)
        mask[y, x] = 0
        return mask


    def _patch_source_masked_pixels_fname(self, freq, patch_num=0, mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None):
        fname_root = self._mask_fname_root(freq=freq, mask_sources=True, mask_clusters=False, mask_apod_width=0,
                                           mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr,
                                           mask_snr_threshold_below=mask_snr_threshold_below)
        fname_root = fname_root.replace('mask_', 'masked_pixels_')
        fname = os.path.join(self.patch_mask_dir(patch_num=patch_num), f'{fname_root}.txt')
        return fname


    def _get_binary_patch_sources_mask(self, freq, patch_num=0, mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None):
        masked_pixels_fname = self._patch_source_masked_pixels_fname(freq, patch_num=patch_num, mask_snr_threshold=mask_snr_threshold,
                                                                         mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
        if os.path.exists(masked_pixels_fname):
            masked_pixels = np.loadtxt(masked_pixels_fname).astype(int)
            mask = self._make_binary_patch_mask(masked_pixels, patch_num=patch_num)
        else:
            imap_mask = self._get_imap_masks(patch_num=patch_num, freqs=[freq])[freq] # mask applied before source subtraction
            fgcleaned_map = self._get_fgcleaned_patch_maps(patch_num=patch_num, freqs=[freq])[freq] * imap_mask
            # calculate a new filter for point sources using a new noise map, if possible
            # (use `noise_maps_for_source_mask_filters` if provided ; or `noise_maps_for_source_filters` if provided ; or the fgcleaned map itself)
            source_filter_kwargs = {key: self.sources_kwargs[key] for key in ['apply_apod', 'smooth_p2d_npix', 'rms_gw', 'rms_niter', 'rms_nsigma']}
            source_filter_kwargs['mask'] = imap_mask
            beam_fwhms = self._patch_beam_fwhms(patch_num=patch_num)
            noise_map_for_source_mask_filter = self._get_patch_noise_maps_for_source_mask_filters(freq, patch_num=patch_num)
            source_filter_for_mask = fgfilters.PointSourceFilter(beam_fwhms[freq], noise_map_for_source_mask_filter, self.patch_apod_width, **source_filter_kwargs)
            mask = fgmasks.make_source_mask(fgcleaned_map, source_filter_for_mask, apod_width=0,
                                    snr_threshold=mask_snr_threshold, use_abs_sn=mask_abs_snr,
                                    snr_threshold_below=mask_snr_threshold_below)
            # save which pixels are masked:
            _, _, masked_pixels = fgmasks.get_masked_pixels(mask)
            np.savetxt(masked_pixels_fname, masked_pixels)
        return mask


    def _patch_cluster_masked_pixels_fname(self, patch_num=0, mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None):
        fname_root = self._mask_fname_root(mask_sources=False, mask_clusters=True, mask_apod_width=0,
                                           mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr,
                                           mask_snr_threshold_below=mask_snr_threshold_below)
        fname_root = fname_root.replace('mask_', 'masked_pixels_')
        fname = os.path.join(self.patch_mask_dir(patch_num=patch_num), f'{fname_root}.txt')
        return fname


    def _get_binary_patch_clusters_mask(self, patch_num=0, mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None):
        masked_pixels_fname = self._patch_cluster_masked_pixels_fname(patch_num=patch_num, mask_snr_threshold=mask_snr_threshold,
                                                                          mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
        if os.path.exists(masked_pixels_fname):
            masked_pixels = np.loadtxt(masked_pixels_fname).astype(int)
            mask = self._make_binary_patch_mask(masked_pixels, patch_num=patch_num)
        else:
            imap_masks = self._get_imap_masks(patch_num=patch_num) # masks applied before source subtraction
            fgcleaned_maps = self._get_fgcleaned_patch_maps(patch_num=patch_num)
            for freq in self.freqs:
                fgcleaned_maps[freq] *= imap_masks[freq]
            self._init_patch_clusterlib(patch_num=patch_num)
            mask = fgmasks.make_cluster_mask(fgcleaned_maps, self.clusterlibs[patch_num].filters, apod_width=0,
                                     snr_threshold=mask_snr_threshold, use_abs_sn=mask_abs_snr, snr_threshold_below=mask_snr_threshold_below)
            # save which pixels are masked:
            _, _, masked_pixels = fgmasks.get_masked_pixels(mask)
            np.savetxt(masked_pixels_fname, masked_pixels)
        return mask


    # ----- masks applied to known large/massive/nearby clusters after FG cleaning : -----

    def _get_clusters_to_mask_after_fgclean_catalog(self, patch_num=None):
        if self.clusters_to_mask_after_fgclean_catalog is not None:
            catalog = self.clusters_to_mask_after_fgclean_catalog.copy()
            if patch_num is not None:
                catalog = self.patches.trim_patch_catalog(catalog, patch_num, padded=True)
            return catalog


    def _get_large_clusters_mask(self, patch_num=None, apodize=True):
        catalog_for_mask = self._get_clusters_to_mask_after_fgclean_catalog(patch_num=patch_num)
        if patch_num is not None:
            shape, wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)
        else:
            shape = self.map_shape
            wcs = self.map_wcs
        if (catalog_for_mask is not None) and (len(catalog_for_mask) > 0):
            if apodize:
                mask = fgmasks.make_mask_from_catalog(shape, wcs, catalog_for_mask)
            else:
                mask = fgmasks.make_binary_mask_from_catalog(shape, wcs, catalog_for_mask)
        else:
            mask = enmap.ones(shape, wcs)
        return mask


    # ----- file names: -----

    def _get_mask_info(self, mask_sources=True, mask_clusters=True, mask_apod_width=1, mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None, patch_num=None):
        masked_components = []
        if mask_sources and self.subtract_sources:
            masked_components.append('sources')
        if mask_clusters and self.subtract_clusters:
            clusters_info = 'clusters'
            if mask_apod_width > 1:
                clusters_to_mask = self._get_clusters_to_mask_after_fgclean_catalog(patch_num=patch_num)
                num_to_mask = 0 if (clusters_to_mask is None) else len(clusters_to_mask)
                if num_to_mask > 0:
                    clusters_info = f'{clusters_info}_plus{num_to_mask}fromcatalog'
            masked_components.append(clusters_info)
        if len(masked_components) > 0:
            if mask_abs_snr:
                snr_info = f'absSN{simutils.round_str(mask_snr_threshold)}'
            else:
                snr_info = f'SNabove{simutils.round_str(mask_snr_threshold)}'
                if mask_snr_threshold_below is not None:
                    snr_info = f'{snr_info}below{simutils.round_str(mask_snr_threshold_below)}'
            apod_info = f'apod{simutils.round_str(mask_apod_width)}arcmin'
            mask_info = '_'.join([snr_info, *masked_components, apod_info])
            return mask_info


    def _mask_fname_root(self, freq=None, mask_sources=True, mask_clusters=True, mask_apod_width=1, mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None):
        mask_info = self._get_mask_info(mask_sources=mask_sources, mask_clusters=mask_clusters, mask_apod_width=mask_apod_width,
                                        mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
        fname_root = f'mask_{mask_info}'
        if mask_sources and self.subtract_sources:
            if freq is None:
                raise ValueError(f"`{freq = }`. You must pass the map frequency (in GHz) to mask point sources.")
            fname_root = f'{round(freq):03d}GHz_{fname_root}'
        return fname_root


    def get_mask_fname(self, freq=None, patch_num=None, mask_sources=True, mask_clusters=True, mask_apod_width=1, mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None):
        if patch_num is not None:
            fname = self.get_patch_mask_fname(freq=freq, patch_num=patch_num, mask_sources=mask_sources, mask_clusters=mask_clusters, mask_apod_width=mask_apod_width,
                                              mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
        else:
            fname_root = self._mask_fname_root(freq=freq, mask_sources=mask_sources, mask_clusters=mask_clusters, mask_apod_width=mask_apod_width,
                                               mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
            fname = os.path.join(self.mask_dir(), f'{fname_root}.fits')
        return fname


    def get_patch_mask_fname(self, patch_num=0, freq=None, mask_sources=True, mask_clusters=True, mask_apod_width=0, mask_snr_threshold=5, mask_abs_snr=True, mask_snr_threshold_below=None):
        fname_root = self._mask_fname_root(freq=freq, mask_sources=mask_sources, mask_clusters=mask_clusters, mask_apod_width=mask_apod_width,
                                           mask_snr_threshold=mask_snr_threshold, mask_abs_snr=mask_abs_snr, mask_snr_threshold_below=mask_snr_threshold_below)
        fname = os.path.join(self.patch_mask_dir(patch_num=patch_num), f'{fname_root}.fits')
        return fname



    ########################################
    ###############  misc.:  ###############
    ########################################

    def get_patch_num(self, ra=None, dec=None, patch_col_num=None, patch_row_num=None):
        return self.patches.get_patch_num(ra=ra, dec=dec, patch_col_num=patch_col_num, patch_row_num=patch_row_num)


    def infomsg(self, msg):
        """Print out a message.

        Parameters
        ----------
        msg : str
            The message to print.

        Notes
        -----
        This will only print out the message if `verbose=True` was passed
        during initialization.
        """
        fgutils.print_msg(msg, verbose=self.verbose, log=self.log, stacklevel=3)


    def _validate_freq(self, freq):
        if freq not in self.freqs:
            raise ValueError(f"`{freq = }`. The frequency (in units of GHz) must be one of the input map frequencies: {self.freqs}.")
        return freq


    def _validate_freqs(self, freqs):
        # make sure `freqs` is a list instead of a single value:
        freqs = list(np.atleast_1d(freqs))
        invalid_freqs = [freq for freq in freqs if (freq not in self.freqs)]
        if len(invalid_freqs) > 0:
            raise ValueError(f"`{freqs = }`. Each frequency (in units of GHz) in `freqs` must be one of the input map frequencies: {self.freqs}.")
        return freqs


    # -----  output directories:  -----

    def results_dir(self, make_dir=True):
        results_dir_name_info = ['fgcleaned']
        if self.subtract_sources:
            results_dir_name_info.append(f"sources{self.sources_iter_info}")
        if self.subtract_clusters:
            results_dir_name_info.append(f"clusters{self.clusters_iter_info}")
        results_dir_name = '_'.join(results_dir_name_info)
        results_dir_path = os.path.join(self.output_dir, results_dir_name)
        if make_dir:
            utils.mkdir(results_dir_path)
        return results_dir_path


    def fgcleaned_maps_dir(self, make_dir=True):
        sims_dir = os.path.join(self.results_dir(), 'fgcleaned_maps')
        if make_dir:
            utils.mkdir(sims_dir)
        return sims_dir


    def patch_fgclean_dir(self, patch_num=0, make_dir=True):
        ra, dec = self.patches.get_patch_center_coords(patch_num, padded=False)
        patch_ctr_info = f'ra{simutils.round_str(ra, n=3)}_dec{simutils.round_str(dec, n=3)}'
        patch_dir = os.path.join(self.fgclean_dir_for_patches, f'patch{patch_num:02d}_{patch_ctr_info}')
        if make_dir:
            patch_dir = utils.mkdir(patch_dir)
        return patch_dir


    def patch_sources_dir(self, patch_num=0, make_dir=True):
        patches_fgclean_dir = self.patch_fgclean_dir(patch_num=patch_num, make_dir=make_dir)
        patch_dir = os.path.join(patches_fgclean_dir, f'sources_{self.sources_iter_info}')
        if make_dir:
            patch_dir = utils.mkdir(patch_dir)
        return patch_dir


    def patch_clusters_dir(self, patch_num=0, make_dir=True):
        patches_fgclean_dir = self.patch_fgclean_dir(patch_num=patch_num, make_dir=make_dir)
        if self.subtract_sources:
            clusters_dir_name = f'clusters_{self.clusters_iter_info}_after_sources_{self.sources_iter_info}'
        else:
            clusters_dir_name = f'clusters_{self.clusters_iter_info}'
        patch_dir = os.path.join(patches_fgclean_dir, clusters_dir_name)
        if make_dir:
            patch_dir = utils.mkdir(patch_dir)
        return patch_dir


    def mask_dir(self, make_dir=True):
        mask_dir_path = os.path.join(self.results_dir(), 'masks')
        if make_dir:
            utils.mkdir(mask_dir_path)
        return mask_dir_path


    def patch_mask_dir(self, patch_num=0, make_dir=True):
        mask_dir_name_info = ['masks']
        if self.subtract_sources:
            mask_dir_name_info.append(f"sources{self.sources_iter_info}")
        if self.subtract_clusters:
            mask_dir_name_info.append(f"clusters{self.clusters_iter_info}")
        mask_dir_name = '_'.join(mask_dir_name_info)
        patch_dir = os.path.join(self.patch_fgclean_dir(patch_num=patch_num, make_dir=make_dir), mask_dir_name)
        if make_dir:
            patch_dir = utils.mkdir(patch_dir)
        return patch_dir


