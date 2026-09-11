import os
from copy import deepcopy
import numpy as np
from pixell import enmap
from hdsims import utils, simutils, fgcatalogs, maps
from . import fgclean_info as fgi, fgutils, fgfilters, iterfgclean


class FGCleanSourcesFiles:
    def __init__(self, output_dir, freq, extrapolate_dim_sources=True, make_output_dirs=False, verbose=True, log=None):
        self.freq = freq
        self.extrap = extrapolate_dim_sources
        self.verbose = verbose
        self.log = log
        self.make_output_dirs = make_output_dirs
        self.output_dir = output_dir
        self.iter_catalog_dir = os.path.join(self.output_dir, 'iter_catalogs')
        if self.make_output_dirs:
            utils.mkdir(self.output_dir)
            utils.mkdir(self.iter_catalog_dir)
        
    
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
    
    
    def iter_catalog_fname(self, iter_num, snr_threshold, extrap=False):
        return iterfgclean.iter_sources_catalog_fname(iter_num, snr_threshold, iter_catalog_dir=self.iter_catalog_dir, extrap=extrap)
    
    
    def measured_positions_catalog_dir(self):
        odir = os.path.join(self.output_dir, 'sources_measured_at_other_freqs')
        if self.make_output_dirs:
            utils.mkdir(odir)
        return odir
    
    
    def measured_positions_catalog_fname(self):
        return os.path.join(self.measured_positions_catalog_dir(), 'sources_measured_at_other_freqs.csv')
    
    
    def catalog_idxs_to_exclude_fname(self):
        return os.path.join(self.measured_positions_catalog_dir(), 'source_idxs_missubtracted_at_other_freqs.txt')
    
    
    
    # --- initial step of iteratively finding bright sources above some SNR threshold: ---
    
    def measured_bright_srcs_dir(self):
        odir = os.path.join(self.output_dir, 'step1_measure_bright_sources')
        if self.make_output_dirs:
            utils.mkdir(odir)
        return odir
    
    
    def measured_bright_srcs_catalog_fname(self):
        """catalog of all sources subtracted after finding/measuring bright sources above SNR"""
        return os.path.join(self.measured_bright_srcs_dir(), 'subtracted_sources.csv')
    
    
    def measured_bright_srcs_map_fname(self):
        """map of all sources subtracted after finding/measuring bright sources above SNR
        (units of uK; no beam or pixwin convolved)
        """
        return os.path.join(self.measured_bright_srcs_dir(), 'subtracted_sources.fits')
    
    
    def load_measured_bright_srcs_catalog(self):
        self.infomsg(f"loading {self.measured_bright_srcs_catalog_fname()}")
        catalog = fgcatalogs.load_catalog(self.measured_bright_srcs_catalog_fname())
        return catalog
    
    
    def load_measured_bright_srcs_map(self):
        src_map_fname = self.measured_bright_srcs_map_fname()
        self.infomsg(f"loading {src_map_fname}")
        measured_srcs_map = enmap.read_map(src_map_fname)
        return measured_srcs_map
    
    
    # --- extrapolating & subtracting remaining dim sources: ---
    
    def extrapolated_dim_srcs_dir(self):
        odir = os.path.join(self.output_dir, 'step2_extrapolate_dim_sources')
        if self.make_output_dirs:
            utils.mkdir(odir)
        return odir
    
    
    def measured_and_extrap_srcs_catalog_fname(self):
        """catalog of all sources subtracted after finding/measuring bright sources above SNR & extrapolating dimmer sources"""
        return os.path.join(self.extrapolated_dim_srcs_dir(), 'subtracted_sources.csv')
    
    
    def measured_and_extrap_srcs_map_fname(self):
        """map of all sources subtracted after finding/measuring bright sources above SNR  & extrapolating dimmer sources
        (units of uK; no beam or pixwin convolved)
        """
        return os.path.join(self.extrapolated_dim_srcs_dir(), 'subtracted_sources.fits')
    
    
    def load_measured_and_extrap_srcs_catalog(self):
        self.infomsg(f"loading {self.measured_and_extrap_srcs_catalog_fname()}")
        catalog = fgcatalogs.load_catalog(self.measured_and_extrap_srcs_catalog_fname())
        return catalog
    
    
    def load_measured_and_extrap_srcs_map(self):
        src_map_fname = self.measured_and_extrap_srcs_map_fname()
        self.infomsg(f"loading {src_map_fname}")
        subtracted_srcs_map = enmap.read_map(src_map_fname)
        return subtracted_srcs_map
    
    
    def spectral_index_catalog_fname(self, extrap_freq, component=None):
        freq_info = f'{simutils.round_str(self.freq)}and{simutils.round_str(extrap_freq)}GHz'
        if component is not None: # catalog of only sources used for final index calc
            fname = f'{freq_info}_sources_used_for_{component}_index.csv'
        else:
            fname = f'all_{freq_info}_sources_for_spectral_index_calc.csv'
        fname = os.path.join(self.extrapolated_dim_srcs_dir(), fname)
        return fname
    
    
    def load_spectral_index_catalog(self, extrap_freq, component=None):
        '''just writing this so we don't have to init. the whole src sub class to get catalog if it's already been saved'''
        return fgcatalogs.load_catalog(self.spectral_index_catalog_fname(extrap_freq, component=component))
    
    
    def spectral_index_fname(self, component):
        """`component` is 'cib' or 'radio'"""
        if component.lower() not in ['cib', 'radio']:
            raise ValueError(f"`{component = }`. The `component` name must be either `'cib'` or `'radio'`.")
        fname = os.path.join(self.extrapolated_dim_srcs_dir(), f'{component}_spectral_index.txt')
        return fname
        
        
    def load_spectral_index(self, component):
        index_mean, index_std_dev = np.loadtxt(self.spectral_index_fname(component), unpack=True)
        return index_mean, index_std_dev
    
    
    # --- finding, re-measuring, and re-subtracting missubtracted sources: ---
    
    def remeasured_srcs_dir(self):
        step_num = 3 if self.extrap else 2
        odir = os.path.join(self.output_dir, f'step{step_num}_remeasure_missubtracted_sources')
        if self.make_output_dirs:
            utils.mkdir(odir)
        return odir
    
    
    def missubtracted_sources_catalog_fname(self):
        return os.path.join(self.remeasured_srcs_dir(), 'missubtracted_sources.csv')
    
    
    def load_missubtracted_sources_catalog(self):
        self.infomsg(f"loading {self.missubtracted_sources_catalog_fname()}")
        catalog = fgcatalogs.load_catalog(self.missubtracted_sources_catalog_fname())
        return catalog
    
    
    def remeasured_srcs_catalog_fname(self):
        """catalog of all sources subtracted after re-measuring missubtracted sources"""
        return os.path.join(self.remeasured_srcs_dir(), 'subtracted_sources.csv')
    
    
    def remeasured_srcs_map_fname(self):
        """map of all sources subtracted after re-measuring missubtracted sources
        (units of uK; no beam or pixwin convolved)
        """
        return os.path.join(self.remeasured_srcs_dir(), 'subtracted_sources.fits')
    
    
    def load_remeasured_catalog(self):
        self.infomsg(f"loading {self.remeasured_srcs_catalog_fname()}")
        catalog = fgcatalogs.load_catalog(self.remeasured_srcs_catalog_fname())
        return catalog
    
    
    def load_remeasured_src_map(self):
        src_map_fname = self.remeasured_srcs_map_fname()
        self.infomsg(f"loading {src_map_fname}")
        remeasured_srcs_map = enmap.read_map(src_map_fname)
        return remeasured_srcs_map
    
    
    # --- after running all steps to produce final catalog & source-subtracted sim ---
    
    def catalog_of_subtracted_sources_fname(self):
        """final catalog of all subtracted sources"""
        return os.path.join(self.output_dir, 'subtracted_sources.csv')
    
    
    def map_of_subtracted_sources_fname(self):
        """final map of all subtracted sources (no beam or pixwin)"""
        return os.path.join(self.output_dir, 'subtracted_sources.fits')
    
    
    def source_subtracted_sim_fname(self):
        """final source-subtracted sim"""
        return os.path.join(self.output_dir, 'sim_after_source_subtraction.fits')
    
    
    def load_catalog_of_subtracted_sources(self):
        self.infomsg(f"loading {self.catalog_of_subtracted_sources_fname()}")
        return fgcatalogs.load_catalog(self.catalog_of_subtracted_sources_fname())
    
    
    def load_map_of_subtracted_sources(self):
        """NOTE: maps include apod. region, and are NOT convolved w/ pixwin or beam"""
        self.infomsg(f"loading {self.map_of_subtracted_sources_fname()}")
        subtracted_srcs_map = enmap.read_map(self.map_of_subtracted_sources_fname())
        return subtracted_srcs_map
    
    
    def load_source_subtracted_sim(self):
        if os.path.exists(self.source_subtracted_sim_fname()):
            self.infomsg(f"loading {self.source_subtracted_sim_fname()}")
        return enmap.read_map(self.source_subtracted_sim_fname())
    




class FGCleanSources(FGCleanSourcesFiles):
    def __init__(self, output_dir,
                 imap, freq, beam_fwhm, apod_width,
                 noise_map_for_filter,
                 snr_threshold_list,
                 smooth_p2d_npix=fgi.p2d_smooth_npix,
                 rms_gw=fgi.rms_gw_sources, rms_niter=fgi.rms_niter, rms_nsigma=fgi.rms_nsigma,
                 apply_apod=False, apod_window=None,
                 beam_solid_angle_func=None,
                 calc_beam_solid_angle_func=True,
                 min_num_iter_sources_per_snr=fgi.min_num_iter_sources_per_snr,
                 measured_positions_catalog=None,
                 extrapolate_dim_sources=True, remeasure_missubtracted_sources=True,
                 freq_for_radio_extrap=None, freq_for_cib_extrap=None,
                 catalog_idxs_to_exclude_from_extrap=[],
                 min_snr_for_spectral_index=fgi.index_min_snr, nsigma_for_spectral_index=fgi.index_nsigma_to_remove,
                 min_snr_for_extrap=fgi.min_snr_for_extrap,
                 remeasure_snr_threshold_list=None,
                 max_ntimes_remeasure=fgi.max_ntimes_remeasure_sources,
                 mask_for_filtered_maps=None,
                 verbose=True, log=None):
        '''
        apply_apod refers to input map: if it isn't already apodized, need to apply apod. before filtering
        '''
        super().__init__(output_dir, freq, extrapolate_dim_sources=extrapolate_dim_sources, make_output_dirs=True, verbose=verbose, log=log)

        self.imap = imap.copy()
        self.shape = self.imap.shape
        self.wcs = self.imap.wcs

        self.beam_fwhm = beam_fwhm
        self.apod_width = apod_width
        self.apod_window = apod_window

        self.filter = fgfilters.PointSourceFilter(self.beam_fwhm, noise_map_for_filter, self.apod_width,
                                                  rms_gw=rms_gw, rms_niter=rms_niter, rms_nsigma=rms_nsigma,
                                                  apod_width=self.apod_width, apply_apod=apply_apod,
                                                  deconvolve_pixwin=True, smooth_p2d_npix=smooth_p2d_npix,
                                                  mask=mask_for_filtered_maps)
        self.snr_threshold_list = snr_threshold_list.copy()
        self.min_num_iter_sources_per_snr = min_num_iter_sources_per_snr
        self.beam_solid_angle_func = beam_solid_angle_func


        if os.path.exists(self.measured_positions_catalog_fname()):
            self.measured_positions_catalog = fgcatalogs.load_catalog(self.measured_positions_catalog_fname())
        elif measured_positions_catalog is not None:
            if os.path.exists(self.catalog_of_subtracted_sources_fname()):
                errmsg = (f"Cannot use the provided `measured_positions_catalog`: the catalog of {self.freq} GHz"
                          f" sources has already been measured without using a `measured_positions_catalog` and saved"
                          f" in the `output_dir` to {self.catalog_of_subtracted_sources_fname()}. You must provide a"
                          " different `output_dir` if you would like to use the `measured_positions_catalog`.")
                raise ValueError(errmsg)
            self.measured_positions_catalog = measured_positions_catalog.copy()
        else:
            self.measured_positions_catalog = None


        self.extrap = extrapolate_dim_sources and (self.measured_positions_catalog is not None)
        self.remeasure = remeasure_missubtracted_sources

        if self.extrap: # determine which frequency to use to measure CIB and radio spectral indices
            extrap_freqs = list(set(self.measured_positions_catalog['freq'].values))
            # if `freq_for_radio_extrap` and/or `freq_for_radio_extrap` were passed, make sure we have information from that freq. in the `measured_positions_catalog`
            if freq_for_radio_extrap not in [None, *extrap_freqs]:
                raise ValueError(f"You passed `{freq_for_radio_extrap = }`, but this frequency was not found in the `'freq'` column of the `measured_positions_catalog`.")
            if freq_for_cib_extrap not in [None, *extrap_freqs]:
                raise ValueError(f"You passed `{freq_for_cib_extrap = }`, but this frequency was not found in the `'freq'` column of the `measured_positions_catalog`.")
            # # if `freq_for_radio_extrap` and/or `freq_for_radio_extrap` were not passed, use the min/max available frequency for radio/cib
            if len(extrap_freqs) > 1:
                radio_extrap_freq = min(extrap_freqs)
                cib_extrap_freq = max(extrap_freqs)
            else:
                extrap_freq = extrap_freqs[0]
                if extrap_freq > self.freq:
                    radio_extrap_freq = self.freq
                    cib_extrap_freq = extrap_freq
                else:
                    radio_extrap_freq = extrap_freq
                    cib_extrap_freq = self.freq
            self.freq_for_radio_extrap = radio_extrap_freq if (freq_for_radio_extrap is None) else freq_for_radio_extrap
            self.freq_for_cib_extrap = cib_extrap_freq if (freq_for_cib_extrap is None) else freq_for_cib_extrap

            if os.path.exists(self.catalog_idxs_to_exclude_fname()):
                self.catalog_idxs_to_exclude_from_extrap = np.loadtxt(self.catalog_idxs_to_exclude_fname())
            else:
                self.catalog_idxs_to_exclude_from_extrap = catalog_idxs_to_exclude_from_extrap.copy()
        else:
            self.freq_for_radio_extrap = None
            self.freq_for_cib_extrap = None
            self.catalog_idxs_to_exclude_from_extrap = []
        self.min_snr_for_spectral_index = min_snr_for_spectral_index
        self.nsigma_for_spectral_index = nsigma_for_spectral_index
        self.min_snr_for_extrap = min(self.snr_threshold_list) if (min_snr_for_extrap is None) else min_snr_for_extrap

        if os.path.exists(self.output_dir) and (self.measured_positions_catalog is not None):
            if not os.path.exists(self.measured_positions_catalog_fname()):
                self.measured_positions_catalog.to_csv(self.measured_positions_catalog_fname())
            if self.extrap and (not os.path.exists(self.catalog_idxs_to_exclude_fname())):
                np.savetxt(self.catalog_idxs_to_exclude_fname(), self.catalog_idxs_to_exclude_from_extrap)


        if remeasure_snr_threshold_list is None:
            self.remeasure_snr_threshold_list = self.snr_threshold_list
        else:
            self.remeasure_snr_threshold_list = remeasure_snr_threshold_list.copy()
        self.max_ntimes_remeasure = max_ntimes_remeasure


        if self.beam_solid_angle_func is None: 
            delta_dec = round(250 * maps.get_map_resolution(self.shape, self.wcs), 3)
            beam_solid_angle_vs_dec_fname = iterfgclean.get_beam_solid_angle_per_dec_fname(self.beam_fwhm, delta_dec, self.shape, self.wcs, save_dir=self.output_dir)
            if os.path.exists(beam_solid_angle_vs_dec_fname) or calc_beam_solid_angle_func:
                self.beam_solid_angle_func = iterfgclean.get_beam_solid_angle_per_dec_func(self.beam_fwhm, delta_dec, self.shape, self.wcs,
                                                                               save=os.path.exists(self.output_dir), save_dir=self.output_dir,
                                                                               verbose=self.verbose, log=self.log)




    def get_apod_window(self):
        if self.apod_window is None:
            self.apod_window = maps.make_apod_window(self.shape, self.wcs, self.apod_width)
        return self.apod_window


    def convolve_map(self, imap, pixwin=False, beam=False, apod=False):
        '''
        note : `apod` can be false b/c `imap` may already be apodized/zero at edges
        '''
        if apod:
            imap *= self.get_apod_window()
        if pixwin:
            imap = enmap.apply_window(imap)
        if beam:
            imap = maps.convolve_sim_with_beam(imap, self.beam_fwhm)
        return imap


    def make_src_map(self, catalog, convolve_pixwin=False, convolve_beam=False, apply_apod=False):
        '''
        note: `apply_apod` can be false, b/c typically measuring them in apodized map (so no srcs at edges)
        but option for it to be true, in case, e.g., making a map from sim catalog (w/ srcs at edges)

        note: the `catalog` should have a column named `'fluxmJy'` for flux at this freq
        '''
        src_map = maps.make_src_map(self.shape, self.wcs, [catalog], self.freq)
        src_map = self.convolve_map(src_map, pixwin=convolve_pixwin, beam=convolve_beam, apod=apply_apod)
        return src_map


    def subtract_catalog_sources_from_input_map(self, catalog, apply_apod=False):
        """subtracts the sources in the `catalog` from the input map passed during initialization

        note: `apply_apod` can be false, b/c typically measuring them in apodized map (so no srcs at edges)
        but option for it to be true, in case, e.g., making a map from sim catalog (w/ srcs at edges)

        note: the `catalog` should have a column named `'fluxmJy'` for flux at this freq
        """
        srcs_sim = self.make_src_map(catalog, convolve_pixwin=True, convolve_beam=True, apply_apod=apply_apod)
        omap = self.imap - srcs_sim
        return omap



    # --- initial step of iteratively finding bright sources above some SNR threshold: ---


    def measure_bright_sources(self, save_subtracted_sources_map=False):
        src_catalog_fname = self.measured_bright_srcs_catalog_fname()
        src_map_fname = self.measured_bright_srcs_map_fname()

        if os.path.exists(src_catalog_fname):
            catalog = self.load_measured_bright_srcs_catalog()
            if os.path.exists(src_map_fname):
                measured_srcs_map = self.load_measured_bright_srcs_map()
            else:
                measured_srcs_map = self.make_src_map(catalog)
                if save_subtracted_sources_map:
                    enmap.write_map(src_map_fname, measured_srcs_map)
                    self.infomsg(f"saved {src_map_fname}")
            sub_sim = self.imap.copy() - self.convolve_map(measured_srcs_map.copy(), pixwin=True, beam=True)

        else:
            self.infomsg(f"iteratively finding, measuring, and subtracting sources from the {self.freq} GHz map down to SNR = {simutils.round_str(np.min(self.snr_threshold_list))}")
            catalog, sub_sim, measured_srcs_map = iterfgclean.iteratively_measure_sources(self.imap, self.freq, self.beam_fwhm, self.filter,
                                                                              self.snr_threshold_list, self.iter_catalog_dir,
                                                                              min_num_iter_sources_per_snr=self.min_num_iter_sources_per_snr,
                                                                              beam_solid_angle_func=self.beam_solid_angle_func,
                                                                              measured_positions_catalog=self.measured_positions_catalog,
                                                                              verbose=self.verbose, log=self.log)
            catalog.to_csv(src_catalog_fname)
            self.infomsg(f'saved {src_catalog_fname}')
            if save_subtracted_sources_map:
                enmap.write_map(src_map_fname, measured_srcs_map)
                self.infomsg(f"saved {src_map_fname}")

        return catalog, sub_sim, measured_srcs_map




    # --- extrapolating & subtracting remaining dim sources: ---


    def _check_if_can_extrap(self):
        if not self.extrap:
            if self.measured_positions_catalog is not None:
                errmsg = ("To extrapolate dim sources in the `measured_positions_catalog` from a different frequency, "
                          "you must pass `extrapolate_dim_sources=True` when initializing the `FGCleanSources` class.")
            else:
                errmsg = ("To extrapolate dim sources from a different frequency, you must pass"
                          " `extrapolate_dim_sources=True` and a `measured_positions_catalog` of the sources measured at"
                          " another frequency when initializing the `FGCleanSources` class.")
            raise ValueError(errmsg)


    def get_spectral_index_catalog(self, extrap_freq, component=None):
        fname = self.spectral_index_catalog_fname(extrap_freq, component=component)
        if not os.path.exists(fname):
            self.measure_spectral_indices()
        catalog = fgcatalogs.load_catalog(fname)
        return catalog


    def save_spectral_index(self, component, index_mean, index_std_dev, extrap_freq):
        fname = self.spectral_index_fname(component)
        header = f"measured {extrap_freq}-to-{self.freq} {component} index mean and standard deviation"
        cols = np.column_stack([[index_mean], [index_std_dev]])
        np.savetxt(fname, cols, header=header)
        self.infomsg(f"saved {fname}")


    def get_spectral_index(self, component):
        fname = self.spectral_index_fname(component)
        if not os.path.exists(fname):
            self.measure_spectral_indices()
        index_mean, index_std_dev = self.load_spectral_index(component)
        return index_mean, index_std_dev


    def measure_spectral_indices(self):
        self._check_if_can_extrap()
        # get catalog of measured & subtracted sources above SNR at this freq:
        catalog = self.load_measured_bright_srcs_catalog()
        # only keep sources in un-apodized region:
        ra_ctr, dec_ctr, map_width, map_height = maps.get_map_ctr_extent(self.shape, self.wcs)
        width = map_width - 2 * self.apod_width
        height = map_height - 2 * self.apod_width
        catalog = fgcatalogs.trim_catalog_positions(catalog, ra_ctr=ra_ctr, dec_ctr=dec_ctr, width=width, height=height)
        # also trim the catalog of measured sources at other freq(s):
        measured_positions_catalog = fgcatalogs.trim_catalog_positions(self.measured_positions_catalog.copy(),
                                                                       ra_ctr=ra_ctr, dec_ctr=dec_ctr, width=width, height=height)
        measured_positions_catalog = measured_positions_catalog[~measured_positions_catalog['idx'].isin(self.catalog_idxs_to_exclude_from_extrap)]

        spectral_index_kwargs = {'min_snr4index': self.min_snr_for_spectral_index, 'nsigma_for_index': self.nsigma_for_spectral_index,
                                 'return_catalogs': True, 'verbose': False}

        if self.freq not in [self.freq_for_radio_extrap, self.freq_for_cib_extrap]:
            # cib:
            self.infomsg(f"measuring {self.freq_for_cib_extrap}-to-{self.freq} CIB spectral index")
            _, _, cib_index_mean, cib_index_std, cib_index_cats = iterfgclean.measure_avg_spectral_indices(self.freq, self.freq_for_cib_extrap,
                                                                                               catalog, measured_positions_catalog,
                                                                                               **spectral_index_kwargs)
            # save index mean and standard deviation
            self.save_spectral_index('cib', cib_index_mean, cib_index_std, self.freq_for_cib_extrap)
            # save catalogs of (1) all sources input to calculation and (2) only sources used (after removing outliers)
            cib_index_cats['all'].to_csv(self.spectral_index_catalog_fname(self.freq_for_cib_extrap))
            cib_index_cats['cib'].to_csv(self.spectral_index_catalog_fname(self.freq_for_cib_extrap, component='cib'))

            # radio:
            self.infomsg(f"measuring {self.freq_for_radio_extrap}-to-{self.freq} radio spectral index")
            radio_index_mean, radio_index_std, _, _, radio_index_cats = iterfgclean.measure_avg_spectral_indices(self.freq, self.freq_for_radio_extrap,
                                                                                                     catalog, measured_positions_catalog,
                                                                                                     **spectral_index_kwargs)
            # save index mean and standard deviation
            self.save_spectral_index('radio', radio_index_mean, radio_index_std, self.freq_for_radio_extrap)
            # save catalogs of (1) all sources input to calculation and (2) only sources used (after removing outliers)
            radio_index_cats['all'].to_csv(self.spectral_index_catalog_fname(self.freq_for_radio_extrap))
            radio_index_cats['radio'].to_csv(self.spectral_index_catalog_fname(self.freq_for_radio_extrap, component='radio'))

        else:
            extrap_freq = self.freq_for_cib_extrap if (self.freq_for_cib_extrap != self.freq) else self.freq_for_radio_extrap
            self.infomsg(f"measuring {extrap_freq}-to-{self.freq} CIB and radio spectral indices")
            radio_index_mean, radio_index_std, cib_index_mean, cib_index_std, index_cats = iterfgclean.measure_avg_spectral_indices(self.freq, extrap_freq,
                                                                                                                        catalog, measured_positions_catalog,
                                                                                                                        **spectral_index_kwargs)
            # save index mean and standard deviation
            self.save_spectral_index('cib', cib_index_mean, cib_index_std, extrap_freq)
            self.save_spectral_index('radio', radio_index_mean, radio_index_std, extrap_freq)
            # save catalogs of (1) all sources input to calculation and (2) only sources used (after removing outliers)
            index_cats['all'].to_csv(self.spectral_index_catalog_fname(extrap_freq))
            index_cats['cib'].to_csv(self.spectral_index_catalog_fname(extrap_freq, component='cib'))
            index_cats['radio'].to_csv(self.spectral_index_catalog_fname(extrap_freq, component='radio'))

        self.infomsg(f"average radio spectral index = {radio_index_mean:5.2f} +/- {radio_index_std:5.2f}")
        self.infomsg(f"average CIB spectral index   = {cib_index_mean:5.2f} +/- {cib_index_std:5.2f}")

        return radio_index_mean, radio_index_std, cib_index_mean, cib_index_std



    def identify_sources(self, catalog):
        radio_index_mean, radio_index_std = self.get_spectral_index('radio')
        cib_index_mean, cib_index_std = self.get_spectral_index('cib')
        catalog = iterfgclean.identify_sources(self.freq, catalog, self.measured_positions_catalog,
                                   radio_index_mean, radio_index_std, cib_index_mean, cib_index_std)
        return catalog



    def extrapolate_and_subtract_dim_sources(self, save_subtracted_sources_map=False):
        self._check_if_can_extrap()
        src_catalog_fname = self.measured_and_extrap_srcs_catalog_fname()
        src_map_fname = self.measured_and_extrap_srcs_map_fname()

        if os.path.exists(src_catalog_fname):
            catalog = self.load_measured_and_extrap_srcs_catalog()
            if os.path.exists(src_map_fname):
                subtracted_srcs_map = self.load_measured_and_extrap_srcs_map()
            else:
                subtracted_srcs_map = self.make_src_map(catalog)
                if save_subtracted_sources_map:
                    enmap.write_map(src_map_fname, subtracted_srcs_map)
                    self.infomsg(f"saved {src_map_fname}")
            sub_sim = self.imap.copy() - self.convolve_map(subtracted_srcs_map.copy(), pixwin=True, beam=True)

        else:
            extrap_freqs = list(set(self.measured_positions_catalog['freq'].values))
            extrap_freq_info = ', '.join([str(extrap_freq) for extrap_freq in extrap_freqs])
            self.infomsg(f"extrapolating the fluxes of remaining dim sources found at {extrap_freq_info} GHz and subtracting them from the {self.freq} GHz map")

            # get initial catalog/map of measured & subtracted sources above SNR and the source-subtracted map:
            catalog, sub_sim, measured_srcs_map = self.measure_bright_sources()

            # measure CIB / radio spectral indices
            if self.freq_for_radio_extrap == self.freq:
                radio_index_mean = None
            else:
                radio_index_mean, _ = self.get_spectral_index('radio')
            if self.freq_for_cib_extrap == self.freq:
                cib_index_mean = None
            else:
                cib_index_mean, _ = self.get_spectral_index('cib')

            # determine whether each source already measured at this freq is CIB, radio, or unknown
            catalog = self.identify_sources(catalog)

            # exclude any missubtracted sources at other freq(s)
            measured_positions_catalog = self.measured_positions_catalog[~self.measured_positions_catalog['idx'].isin(self.catalog_idxs_to_exclude_from_extrap)]

            # extrapolate & subtract remaining dim sources:
            catalog, sub_sim, subtracted_srcs_map = iterfgclean.extrapolate_and_subtract_unmeasured_sources(sub_sim, measured_srcs_map, catalog,
                                                                                                measured_positions_catalog,
                                                                                                self.freq, self.beam_fwhm, self.filter,
                                                                                                radio_index=radio_index_mean, cib_index=cib_index_mean,
                                                                                                min_extrap_snr=self.min_snr_for_extrap,
                                                                                                iter_catalog_dir=self.iter_catalog_dir)

            # save catalog of only extrapolated & subtracted sources
            extrap_cat = catalog[catalog['method'].eq('extrap')]
            extrap_inum = np.max(catalog['iter_num'].values)
            extrap_cat_fname = self.iter_catalog_fname(extrap_inum, 0, extrap=True)
            extrap_cat.to_csv(extrap_cat_fname)
            self.infomsg(f"saved {extrap_cat_fname}")

            # save catalog & map of all (measured + extrapolated) subtracted sources so far:
            catalog.to_csv(src_catalog_fname)
            self.infomsg(f'saved {src_catalog_fname}')
            if save_subtracted_sources_map:
                enmap.write_map(src_map_fname, subtracted_srcs_map)
                self.infomsg(f"saved {src_map_fname}")

        return catalog, sub_sim, subtracted_srcs_map


    # --- finding, re-measuring, and re-subtracting missubtracted sources: ---

    def remeasure_sources(self, save_subtracted_sources_map=False):
        src_catalog_fname = self.remeasured_srcs_catalog_fname()
        src_map_fname = self.remeasured_srcs_map_fname()

        if os.path.exists(src_catalog_fname):
            catalog = self.load_remeasured_catalog()
            if os.path.exists(src_map_fname):
                remeasured_srcs_map = self.load_remeasured_src_map()
            else:
                remeasured_srcs_map = self.make_src_map(catalog)
                if save_subtracted_sources_map:
                    enmap.write_map(src_map_fname, remeasured_srcs_map)
                    self.infomsg(f"saved {src_map_fname}")
            sub_sim = self.imap.copy() - self.convolve_map(remeasured_srcs_map.copy(), pixwin=True, beam=True)

        else:
            self.infomsg(f"re-measuring and re-subtracting sources that were mis-subtracted from the {self.freq} GHz map")

            # get catalog/map of measured (+ extrapolated) & subtracted sources and the source-subtracted map:
            if self.extrap:
                catalog, sub_sim, measured_srcs_map = self.extrapolate_and_subtract_dim_sources()
            else:
                catalog, sub_sim, measured_srcs_map = self.measure_bright_sources()

            # load catalog of missubtracted sources, if it exists:
            if os.path.exists(self.missubtracted_sources_catalog_fname()):
                missubtracted_srcs = self.load_missubtracted_sources_catalog()
            else:
                missubtracted_srcs = None


            catalog, missubtracted_srcs, sub_sim, remeasured_srcs_map = iterfgclean.remeasure_missubtracted_sources(sub_sim, measured_srcs_map, catalog,
                                                                                                       self.remeasure_snr_threshold_list,
                                                                                                       self.freq, self.beam_fwhm, self.apod_width,
                                                                                                       self.filter, self.iter_catalog_dir,
                                                                                                       beam_solid_angle_func=self.beam_solid_angle_func,
                                                                                                       missubtracted_srcs_catalog=missubtracted_srcs,
                                                                                                       measured_positions_catalog=self.measured_positions_catalog,
                                                                                                       min_num_iter_sources_per_snr=self.min_num_iter_sources_per_snr,
                                                                                                       max_ntimes_remeasure=self.max_ntimes_remeasure,
                                                                                                       verbose=self.verbose, log=self.log)
            missubtracted_srcs.to_csv(self.missubtracted_sources_catalog_fname())
            self.infomsg(f"saved {self.missubtracted_sources_catalog_fname()}")
            catalog.to_csv(src_catalog_fname)
            self.infomsg(f'saved {src_catalog_fname}')
            if save_subtracted_sources_map:
                enmap.write_map(src_map_fname, measured_srcs_map)
                self.infomsg(f"saved {src_map_fname}")

        return catalog, sub_sim, remeasured_srcs_map



    # --- run all steps to produce final catalog & source-subtracted sim ---


    def run_source_subtraction(self, save_map_after_subtraction=False, save_subtracted_sources_map=False, save_intermediate_subtracted_sources_maps=False):
        subtracted_srcs_catalog_fname = self.catalog_of_subtracted_sources_fname()
        subtracted_srcs_map_fname = self.map_of_subtracted_sources_fname()
        sub_sim_fname = self.source_subtracted_sim_fname()

        if os.path.exists(subtracted_srcs_catalog_fname):
            catalog = self.load_catalog_of_subtracted_sources()
            if os.path.exists(sub_sim_fname):
                sub_sim = self.load_source_subtracted_sim()
            else:
                if os.path.exists(subtracted_srcs_map_fname):
                    subtracted_srcs_map = self.load_map_of_subtracted_sources()
                else:
                    subtracted_srcs_map = self.make_src_map(catalog)
                    if save_subtracted_sources_map:
                        enmap.write_map(subtracted_srcs_map_fname, subtracted_srcs_map)
                        self.infomsg(f"saved {subtracted_srcs_map_fname}")
                sub_sim = self.imap.copy() - self.convolve_map(subtracted_srcs_map, pixwin=True, beam=True)
                if save_map_after_subtraction:
                    enmap.write_map(sub_sim_fname, sub_sim)
                    self.infomsg(f"saved {sub_sim_fname}")

        else:
            # iteratively find, measure, and subtract sources down to the min. SNR threshold:
            catalog, sub_sim, subtracted_srcs_map = self.measure_bright_sources(save_subtracted_sources_map=save_intermediate_subtracted_sources_maps)
            if self.extrap:
                # extrapolate remaining dim sources that were previously found at another freq. & subtract them:
                catalog, sub_sim, subtracted_srcs_map = self.extrapolate_and_subtract_dim_sources(save_subtracted_sources_map=save_intermediate_subtracted_sources_maps)
            if self.remeasure:
                # re-measure and re-subtract any missubtracted sources:
                catalog, sub_sim, subtracted_srcs_map = self.remeasure_sources(save_subtracted_sources_map=save_intermediate_subtracted_sources_maps)

            # if we used measurements from different freq(s), then we can try to identify which sources are CIB vs. radio:
            if self.measured_positions_catalog is not None:
                catalog = self.identify_sources(catalog)
            elif 'component' not in catalog.columns.values:
                catalog['component'] = 'unknown'

            catalog.to_csv(subtracted_srcs_catalog_fname)
            self.infomsg(f"saved {subtracted_srcs_catalog_fname}")
            if save_subtracted_sources_map:
                enmap.write_map(subtracted_srcs_map_fname, subtracted_srcs_map)
                self.infomsg(f"saved {subtracted_srcs_map_fname}")
            if save_map_after_subtraction:
                enmap.write_map(sub_sim_fname, sub_sim)
                self.infomsg(f"saved {sub_sim_fname}")

        return catalog, sub_sim


    def get_catalog_of_subtracted_sources(self):
        if not os.path.exists(self.catalog_of_subtracted_sources_fname()):
            self.run_source_subtraction()
        return self.load_catalog_of_subtracted_sources()


    def get_map_of_subtracted_sources(self, convolve_pixwin=False, convolve_beam=False, save=False):
        if os.path.exists(self.map_of_subtracted_sources_fname()):
            subtracted_srcs_map = self.load_map_of_subtracted_sources()
        else:
            subtracted_srcs_catalog = self.get_catalog_of_subtracted_sources()
            subtracted_srcs_map = self.make_src_map(subtracted_srcs_catalog)
            # if the catalog had already been saved, then we still need to save the map (if requested):
            if save and (not os.path.exists(self.map_of_subtracted_sources_fname())):
                enmap.write_map(self.map_of_subtracted_sources_fname(), subtracted_srcs_map)
                self.infomsg(f"saved {self.map_of_subtracted_sources_fname()}")
        subtracted_srcs_map = self.convolve_map(subtracted_srcs_map, pixwin=convolve_pixwin, beam=convolve_beam)
        return subtracted_srcs_map



    def get_source_subtracted_sim(self, save=False):
        fname = self.source_subtracted_sim_fname()
        if os.path.exists(fname):
            sub_sim = self.load_source_subtracted_sim()
        else:
            sub_sim = self.imap.copy() - self.get_map_of_subtracted_sources(convolve_pixwin=True, convolve_beam=True)
            if save:
                enmap.write_map(fname, sub_sim)
                self.infomsg(f"saved {fname}")
        return sub_sim




class FGCleanMFSourcesFiles:
    def __init__(self, output_dir, freqs,
                 extrapolate_dim_sources=True, 
                 freq_for_radio=None, freq_for_cib=None,
                 verbose=True, log=None):

        self.output_dir = output_dir
        self.verbose = verbose
        self.log = log

        # run highest freq first (to find CIB), then lowest (to find radio), then others
        map_freqs = sorted(freqs)
        self.freqs = [map_freqs[-1], *map_freqs[:-1]]
        self.nfreq = len(self.freqs)

        self.extrap = extrapolate_dim_sources and (self.nfreq > 1)

        if self.nfreq > 1: # (note: this is still necessary to pass correct `measured_positions_catalog` for each freq, even if not extrapolating)
            if freq_for_cib not in [None, *self.freqs]:
                raise ValueError(f"You passed `{freq_for_cib = }`, but this frequency was not found in the keys of the `imaps` dictionary.")
            if freq_for_radio not in [None, *self.freqs]:
                raise ValueError(f"You passed `{freq_for_radio = }`, but this frequency was not found in the keys of the `imaps` dictionary.")
            self.freq_for_cib = max(self.freqs) if (freq_for_cib is None) else freq_for_cib
            self.freq_for_radio = min(self.freqs) if (freq_for_radio is None) else freq_for_radio
            # make sure list of freqs is in correct order
            other_freqs = [freq for freq in self.freqs if (freq not in [self.freq_for_cib, self.freq_for_radio])]
            self.freqs = [self.freq_for_cib, self.freq_for_radio, *other_freqs]
        else:
            self.freq_for_cib = None
            self.freq_for_radio = None

        if self.extrap:
            # key = freq, value = list of other freqs to use for finding bright sources or extrapolating dim sources
            self.extrap_freqs = {self.freq_for_cib: [], self.freq_for_radio: [self.freq_for_cib]}
            for freq in self.freqs[2:]:
                self.extrap_freqs[freq] = [self.freq_for_cib, self.freq_for_radio]


        # make instances of `FGCleanSources` to use for filenames, without actually creating any output files:
        self.ptsrcfiles = {}
        for freq in self.freqs:
            ptsrc_dir = os.path.join(self.output_dir, f'{round(freq):03d}GHz_sources')
            extrap = (len(self.extrap_freqs[freq]) > 0)
            self.ptsrcfiles[freq] = FGCleanSourcesFiles(ptsrc_dir, freq, extrapolate_dim_sources=extrap, make_output_dirs=False, verbose=self.verbose, log=self.log)



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


    def catalog_of_subtracted_sources_fname(self, freq):
        self._validate_freq(freq)
        fname = self.ptsrcfiles[freq].catalog_of_subtracted_sources_fname()
        return fname

    def load_catalog_of_subtracted_sources(self, freq):
        self._validate_freq(freq)
        subtracted_srcs_catalog = self.ptsrcfiles[freq].load_catalog_of_subtracted_sources()
        return subtracted_srcs_catalog


    def map_of_subtracted_sources_fname(self, freq):
        self._validate_freq(freq)
        fname = self.ptsrcfiles[freq].map_of_subtracted_sources_fname()
        return fname

    def load_map_of_subtracted_sources(self, freq):
        """NOTE: maps include apod. region, and are NOT convolved w/ pixwin or beam"""
        self._validate_freq(freq)
        subtracted_srcs_map = self.ptsrcfiles[freq].load_map_of_subtracted_sources()
        return subtracted_srcs_map


    def load_maps_of_subtracted_sources(self, freqs=None):
        """NOTE: maps include apod. region, and are NOT convolved w/ pixwin or beam"""
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        subtracted_srcs_maps = {}
        for freq in freqs:
            subtracted_srcs_maps[freq] = self.load_map_of_subtracted_sources(freq)
        return subtracted_srcs_maps


    def source_subtracted_sim_fname(self, freq):
        self._validate_freq(freq)
        fname = self.ptsrcfiles[freq].source_subtracted_sim_fname()
        return fname

    def load_source_subtracted_sim(self, freq):
        self._validate_freq(freq)
        sub_sim = self.ptsrcfiles[freq].load_source_subtracted_sim()
        return sub_sim


    def load_source_subtracted_sims(self, freqs=None):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        sub_sims = {}
        for freq in freqs:
            sub_sims[freq] = self.load_source_subtracted_sim(freq)
        return sub_sims


    def load_spectral_index(self, freq, component):
        freq = self._validate_freq(freq)
        component = simutils.validate_sim_component_name(component, valid_components=['cib', 'radio'])

        if component.lower() == 'cib':
            if freq == self.freq_for_cib:
                raise ValueError(f"`{freq = }` and `{component = }`: The CIB spectral index for other frequencies is calculated using the sources measured at {freq} GHz.")
            elif self.freq_for_cib in self.extrap_freqs[freq]:
                index_mean, index_std_dev = self.ptsrcfiles[freq].load_spectral_index(component)
            elif freq in self.extrap_freqs[self.freq_for_cib]:
                index_mean, index_std_dev = self.ptsrcfiles[self.freq_for_cib].load_spectral_index(component)
            else:
                raise ValueError(f"The {self.freq_for_cib}-to-{freq} CIB spectral index was not calculated.")

        elif component.lower() == 'radio':
            if freq == self.freq_for_radio:
                raise ValueError(f"`{freq = }` and `{component = }`: The radio spectral index for other frequencies is calculated using the sources measured at {freq} GHz.")
            elif self.freq_for_radio in self.extrap_freqs[freq]:
                index_mean, index_std_dev = self.ptsrcfiles[freq].load_spectral_index(component)
            elif freq in self.extrap_freqs[self.freq_for_radio]:
                # e.g. if `freq` is the CIB frequency, we may not extapolate radio sources to this freq, but we did calculate the index when running at the `freq_for_radio`
                index_mean, index_std_dev = self.ptsrcfiles[self.freq_for_radio].load_spectral_index(component)
            else:
                raise ValueError(f"The {self.freq_for_radio}-to-{freq} radio spectral index was not calculated.")

        return index_mean, index_std_dev


    def load_spectral_index_catalog(self, freq, extrap_freq, component=None):
        freq = self._validate_freq(freq)
        extrap_freq = self._validate_freq(extrap_freq)
        component = simutils.validate_sim_component_name(component, valid_components=['cib', 'radio', None])
        if extrap_freq not in self.extrap_freqs[freq]:
            raise ValueError(f"The {extrap_freq}-to-{freq} GHz spectral indices were never calculated.")
        else:
            catalog = self.ptsrcfiles[freq].load_spectral_index_catalog(extrap_freq, component=component)
        return catalog




class FGCleanMFSources(FGCleanMFSourcesFiles):
    def __init__(self, output_dir, imaps, beam_fwhms, apod_width, noise_maps_for_filters, snr_threshold_list,
                 smooth_p2d_npix=fgi.p2d_smooth_npix,
                 rms_gw=fgi.rms_gw_sources, rms_niter=fgi.rms_niter, rms_nsigma=fgi.rms_nsigma,
                 apply_apod=False, apod_window=None,
                 beam_solid_angle_funcs=None, calc_beam_solid_angle_funcs=True,
                 min_num_iter_sources_per_snr=fgi.min_num_iter_sources_per_snr,
                 extrapolate_dim_sources=True, remeasure_missubtracted_sources=True,
                 freq_for_radio=None, freq_for_cib=None,
                 min_snr_for_spectral_index=fgi.index_min_snr, nsigma_for_spectral_index=fgi.index_nsigma_to_remove,
                 min_snr_for_extrap=fgi.min_snr_for_extrap,
                 remeasure_snr_threshold_list=None, max_ntimes_remeasure=fgi.max_ntimes_remeasure_sources,
                 masks_for_filtered_maps=None,
                 verbose=True, log=None):
        super().__init__(output_dir, list(imaps.keys()), extrapolate_dim_sources=extrapolate_dim_sources,
                         freq_for_radio=freq_for_radio, freq_for_cib=freq_for_cib, verbose=verbose, log=log)
        self.imaps = deepcopy(imaps)
        self.shape = self.imaps[self.freqs[0]].shape
        self.wcs = self.imaps[self.freqs[0]].wcs
        self.noise_maps_for_filters = deepcopy(noise_maps_for_filters)
        if not all([(freq in self.noise_maps_for_filters) for freq in self.freqs]):
            raise ValueError("You must pass a noise map for each map frequency "
                             "in the `noise_maps_for_filters` dictionary.")

        if masks_for_filtered_maps is not None:
            self.masks_for_filtered_maps = deepcopy(masks_for_filtered_maps)
        else:
            self.masks_for_filtered_maps = masks_for_filtered_maps

        self.apod_width = apod_width
        self.beam_fwhm = beam_fwhms
        if beam_solid_angle_funcs is None:
            self.beam_solid_angle_funcs = {freq: None for freq in self.freqs}
        else:
            self.beam_solid_angle_funcs = beam_solid_angle_funcs
        self.snr_threshold_list = snr_threshold_list

        # dict of kwargs passed to `FGCleanSources` that are the same for each freq:
        self.ptsrclib_kwargs = {'min_num_iter_sources_per_snr': min_num_iter_sources_per_snr,
                                'smooth_p2d_npix': smooth_p2d_npix, 'rms_gw': rms_gw, 'rms_niter': rms_niter, 'rms_nsigma': rms_nsigma,
                                'apply_apod': apply_apod, 'apod_window': apod_window,

                                'calc_beam_solid_angle_func': calc_beam_solid_angle_funcs,

                                'extrapolate_dim_sources': self.extrap, 'remeasure_missubtracted_sources': remeasure_missubtracted_sources,

                                'min_snr_for_spectral_index': min_snr_for_spectral_index, 'nsigma_for_spectral_index': nsigma_for_spectral_index,
                                'min_snr_for_extrap': min_snr_for_extrap,
                                'remeasure_snr_threshold_list': remeasure_snr_threshold_list, 'max_ntimes_remeasure': max_ntimes_remeasure,

                                'verbose': verbose, 'log': log}



        self.ptsrclibs = {} # instance of `FGCleanSources` for each freq



    def _init_ptsrclib(self, freq):
        self._validate_freq(freq)
        if freq not in self.ptsrclibs:
            ptsrc_kwargs = {**self.ptsrclib_kwargs, 'beam_solid_angle_func': self.beam_solid_angle_funcs[freq]}
            if (self.masks_for_filtered_maps is not None) and (freq in self.masks_for_filtered_maps):
                ptsrc_kwargs['mask_for_filtered_maps'] = self.masks_for_filtered_maps[freq]
            ptsrc_dir = utils.mkdir(os.path.join(self.output_dir, f'{round(freq):03d}GHz_sources'))

            # if final catalog is already saved, or if we don't need catalog from a different freq, we can initialize the class;
            # otherwise, check if we need to pass the `measured_positions_catalog` from the other freq(s)
            if (not os.path.exists(self.catalog_of_subtracted_sources_fname(freq))) and (len(self.extrap_freqs[freq]) > 0):
                # check if we have already saved the input `measured_positions_catalog`; need to initialize a class for this freq to do this
                measured_positions_catalog_fname = self.ptsrcfiles[freq].measured_positions_catalog_fname()
                if not os.path.exists(measured_positions_catalog_fname):
                    # we need to make the catalog;
                    # NOTE that this may mean the entire source-subtraction procedure will be run (if it hasn't been already)
                    #      for each freq in the `extrap_freqs[freq]` list

                    catalogs, sub_sims = self.run_source_subtraction(freqs=self.extrap_freqs[freq]) # make sure class is initialized & catalog saved for each needed freq

                    if self.freq_for_cib in self.extrap_freqs[freq]:
                        # identify any sources that have been missubtracted from the map at the CIB frequency:
                        residual_missubtracted_srcs_catalog = iterfgclean.get_missubtracted_sources_catalog(sub_sims[self.freq_for_cib], catalogs[self.freq_for_cib],
                                                                                             np.min(self.snr_threshold_list), self.ptsrclibs[self.freq_for_cib].filter)
                        cib_freq_missubtracted_idxs = residual_missubtracted_srcs_catalog['idx'].values
                    else:
                        catalogs[self.freq_for_cib] = None
                        cib_freq_missubtracted_idxs = []

                    if self.freq_for_radio in self.extrap_freqs[freq]:
                        # identify any sources that have been missubtracted from the map at the radio frequency:
                        residual_missubtracted_srcs_catalog = iterfgclean.get_missubtracted_sources_catalog(sub_sims[self.freq_for_radio], catalogs[self.freq_for_radio],
                                                                                             np.min(self.snr_threshold_list), self.ptsrclibs[self.freq_for_radio].filter)
                        radio_freq_missubtracted_idxs = residual_missubtracted_srcs_catalog['idx'].values
                    else:
                        catalogs[self.freq_for_radio] = None
                        radio_freq_missubtracted_idxs = []

                    measured_positions_catalog = self._make_source_positions_catalog(catalog_at_cib_freq=catalogs[self.freq_for_cib], catalog_at_radio_freq=catalogs[self.freq_for_radio])
                    ptsrc_kwargs['measured_positions_catalog'] = measured_positions_catalog
                    cib_catalog_idxs_to_exclude = measured_positions_catalog[measured_positions_catalog['freq'].eq(self.freq_for_cib) & measured_positions_catalog['orig_idx'].isin(cib_freq_missubtracted_idxs)]['idx'].values
                    radio_catalog_idxs_to_exclude = measured_positions_catalog[measured_positions_catalog['freq'].eq(self.freq_for_radio) & measured_positions_catalog['orig_idx'].isin(radio_freq_missubtracted_idxs)]['idx'].values
                    ptsrc_kwargs['catalog_idxs_to_exclude_from_extrap'] = [*cib_catalog_idxs_to_exclude, *radio_catalog_idxs_to_exclude]

            self.ptsrclibs[freq] = FGCleanSources(ptsrc_dir, self.imaps[freq], freq, self.beam_fwhm[freq], self.apod_width, self.noise_maps_for_filters[freq], self.snr_threshold_list,  **ptsrc_kwargs)




    def _make_source_positions_catalog(self, catalog_at_cib_freq=None, catalog_at_radio_freq=None):
        '''
        NOTE: assumes that `catalog_at_radio_freq` was made using info from the `catalog_at_cib_freq`
        '''
        if (catalog_at_cib_freq is None) and (catalog_at_radio_freq is None):
            measured_positions_catalog = None

        elif (catalog_at_cib_freq is not None) and (catalog_at_radio_freq is not None):
            # make catalog of sources measured at the CIB frequency (assume they're all CIB for now):
            cib_srcs_catalog = catalog_at_cib_freq.copy()
            cib_srcs_catalog = cib_srcs_catalog.sort_values('SNR', ascending=False).reset_index(drop=True)
            cib_srcs_catalog['orig_idx'] = cib_srcs_catalog['idx'].values.copy()
            cib_srcs_catalog['idx'] = list(range(len(cib_srcs_catalog)))
            cib_srcs_catalog['freq'] = self.freq_for_cib
            cib_srcs_catalog = cib_srcs_catalog[['RADeg', 'decDeg', 'fluxmJy', 'SNR', 'freq', 'idx', 'orig_idx']].copy()
            max_cib_idx = np.max(cib_srcs_catalog['idx'].values)

            # make catalog of sources measured at the radio frequency and identified as radio sources (using info from catalog measured at CIB freq):
            radio_srcs_catalog = catalog_at_radio_freq[catalog_at_radio_freq['component'].eq('radio')].copy()
            radio_srcs_catalog = radio_srcs_catalog.sort_values('SNR', ascending=False).reset_index(drop=True)
            radio_srcs_catalog['orig_idx'] = radio_srcs_catalog['idx'].values.copy()
            # identify which radio sources were found at both frequencies, and remove those sources from the CIB freq. catalog
            radio_src_idxs_in_cib_cat = list(set(radio_srcs_catalog['catalog_idx'].values))
            cib_srcs_catalog = cib_srcs_catalog[~cib_srcs_catalog['idx'].isin(radio_src_idxs_in_cib_cat)]
            # for radio sources measured at both freqs, use the same source `idx`:
            radio_srcs_at_one_freq = radio_srcs_catalog[radio_srcs_catalog['method'].eq('map')].copy()
            radio_srcs_at_both_freqs = radio_srcs_catalog[~radio_srcs_catalog['orig_idx'].isin(radio_srcs_at_one_freq['orig_idx'].values)].copy()
            radio_srcs_at_one_freq['idx'] = list(range(max_cib_idx+1, max_cib_idx+1+len(radio_srcs_at_one_freq)))
            radio_srcs_at_both_freqs['idx'] = radio_srcs_at_both_freqs['catalog_idx'].values
            radio_srcs_catalog = fgcatalogs.combine_catalogs([radio_srcs_at_both_freqs, radio_srcs_at_one_freq])
            radio_srcs_catalog['freq'] = self.freq_for_radio
            radio_srcs_catalog = radio_srcs_catalog[['RADeg', 'decDeg', 'fluxmJy', 'SNR', 'freq', 'idx', 'orig_idx']].copy()

            measured_positions_catalog = fgcatalogs.combine_catalogs([cib_srcs_catalog, radio_srcs_catalog])
            measured_positions_catalog = measured_positions_catalog.sort_values('idx').reset_index(drop=True)

        else:
            if catalog_at_cib_freq is not None:
                measured_positions_catalog = catalog_at_cib_freq.copy()
                freq = self.freq_for_cib
            else:
                measured_positions_catalog = catalog_at_radio_freq.copy()
                freq = self.freq_for_radio
            measured_positions_catalog = measured_positions_catalog.sort_values('SNR', ascending=False).reset_index(drop=True)
            measured_positions_catalog['orig_idx'] = measured_positions_catalog['idx'].values.copy()
            measured_positions_catalog['idx'] = list(range(len(measured_positions_catalog)))
            measured_positions_catalog['freq'] = freq
            measured_positions_catalog = measured_positions_catalog[['RADeg', 'decDeg', 'fluxmJy', 'SNR', 'freq', 'idx', 'orig_idx']].copy()

        measured_positions_catalog = maps.add_pixel_coords_to_catalog(measured_positions_catalog, self.shape, self.wcs, add_pixel_num=True)
        return measured_positions_catalog


    def run_source_subtraction(self, freqs=None, save_maps_after_subtraction=False, save_subtracted_sources_maps=False, save_intermediate_subtracted_sources_maps=False):
        if freqs is None:
            freqs_to_run = self.freqs
        else:
            self._validate_freqs(freqs)
            freqs_to_run = []
            if (self.freq_for_cib in freqs) or any([self.freq_for_cib in self.extrap_freqs[freq] for freq in freqs]):
                freqs_to_run.append(self.freq_for_cib)
            if (self.freq_for_radio in freqs) or any([self.freq_for_radio in self.extrap_freqs[freq] for freq in freqs]):
                freqs_to_run.append(self.freq_for_radio)
            other_freqs = sorted([freq for freq in freqs if (freq not in freqs_to_run)])
            for freq in other_freqs:
                freqs_to_run.append(freq)

        catalogs = {}
        sub_sims = {}

        for freq in freqs_to_run:
            self._init_ptsrclib(freq)
            catalogs[freq], sub_sims[freq] = self.ptsrclibs[freq].run_source_subtraction(save_map_after_subtraction=save_maps_after_subtraction, save_subtracted_sources_map=save_subtracted_sources_maps, save_intermediate_subtracted_sources_maps=save_intermediate_subtracted_sources_maps)

        return catalogs, sub_sims



    def get_catalog_of_subtracted_sources(self, freq):
        if os.path.exists(self.catalog_of_subtracted_sources_fname(freq)):
            subtracted_srcs_catalog = self.load_catalog_of_subtracted_sources(freq)
        else:
            self._init_ptsrclib(freq)
            subtracted_srcs_catalog = self.ptsrclibs[freq].get_catalog_of_subtracted_sources()
        return subtracted_srcs_catalog




    def get_map_of_subtracted_sources(self, freq, convolve_pixwin=False, convolve_beam=False, save=False):
        if os.path.exists(self.map_of_subtracted_sources_fname(freq)):
            subtracted_srcs_map = self.load_map_of_subtracted_sources(freq)
            if convolve_pixwin:
                subtracted_srcs_map = enmap.apply_window(subtracted_srcs_map)
            if convolve_beam:
                subtracted_srcs_map = maps.convolve_sim_with_beam(subtracted_srcs_map, self.beam_fwhm[freq])
        else:
            self._init_ptsrclib(freq)
            subtracted_srcs_map = self.ptsrclibs[freq].get_map_of_subtracted_sources(convolve_pixwin=convolve_pixwin, convolve_beam=convolve_beam, save=save)
        return subtracted_srcs_map


    def get_maps_of_subtracted_sources(self, freqs=None, convolve_pixwin=False, convolve_beam=False, save=False):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        subtracted_srcs_maps = {}
        for freq in freqs:
            subtracted_srcs_maps[freq] = self.get_map_of_subtracted_sources(freq, convolve_pixwin=convolve_pixwin, convolve_beam=convolve_beam, save=save)
        return subtracted_srcs_maps


    def get_source_subtracted_sim(self, freq, save=False):
        if os.path.exists(self.source_subtracted_sim_fname(freq)):
            sub_sim = self.load_source_subtracted_sim(freq)
        elif os.path.exists(self.map_of_subtracted_sources_fname(freq)):
            sub_sim = self.imaps[freq] - self.get_map_of_subtracted_sources(freq, convolve_pixwin=True, convolve_beam=True)
            if save:
                enmap.write_map(self.source_subtracted_sim_fname(freq), sub_sim)
                self.infomsg(f"saved {self.source_subtracted_sim_fname(freq)}")
        else:
            self._init_ptsrclib(freq)
            sub_sim = self.ptsrclibs[freq].get_source_subtracted_sim(save=save)
        return sub_sim




    def get_source_subtracted_sims(self, freqs=None, save=False):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        sub_sims = {}
        for freq in freqs:
            sub_sims[freq] = self.get_source_subtracted_sim(freq, save=save)
        return sub_sims





    def get_spectral_index(self, freq, component):
        try:
            index_mean, index_std_dev = self.load_spectral_index(freq, component)

        except FileNotFoundError:
            freq = self._validate_freq(freq)
            component = simutils.validate_sim_component_name(component, valid_components=['cib', 'radio'])

            if component.lower() == 'cib':
                if freq == self.freq_for_cib:
                    raise ValueError(f"`{freq = }` and `{component = }`: The CIB spectral index for other frequencies is calculated using the sources measured at {freq} GHz.")
                elif self.freq_for_cib in self.extrap_freqs[freq]:
                    self._init_ptsrclib(freq)
                    index_mean, index_std_dev = self.ptsrclibs[freq].get_spectral_index(component)
                elif freq in self.extrap_freqs[self.freq_for_cib]:
                    self._init_ptsrclib(self.freq_for_cib)
                    index_mean, index_std_dev = self.ptsrclibs[self.freq_for_cib].get_spectral_index(component)
                else:
                    raise ValueError(f"The {self.freq_for_cib}-to-{freq} CIB spectral index was not calculated.")

            elif component.lower() == 'radio':
                if freq == self.freq_for_radio:
                    raise ValueError(f"`{freq = }` and `{component = }`: The radio spectral index for other frequencies is calculated using the sources measured at {freq} GHz.")
                elif self.freq_for_radio in self.extrap_freqs[freq]:
                    self._init_ptsrclib(freq)
                    index_mean, index_std_dev = self.ptsrclibs[freq].get_spectral_index(component)
                elif freq in self.extrap_freqs[self.freq_for_radio]:
                    # e.g. if `freq` is the CIB frequency, we may not extapolate radio sources to this freq, but we did calculate the index when running at the `freq_for_radio`
                    self._init_ptsrclib(self.freq_for_radio)
                    index_mean, index_std_dev = self.ptsrclibs[self.freq_for_radio].get_spectral_index(component)
                else:
                    raise ValueError(f"The {self.freq_for_radio}-to-{freq} radio spectral index was not calculated.")

        return index_mean, index_std_dev


    def get_spectral_index_catalog(self, freq, extrap_freq, component=None):
        freq = self._validate_freq(freq)
        extrap_freq = self._validate_freq(extrap_freq)
        component = simutils.validate_sim_component_name(component, valid_components=['cib', 'radio', None])

        if extrap_freq not in self.extrap_freqs[freq]:
            raise ValueError(f"The {extrap_freq}-to-{freq} GHz spectral indices were never calculated.")
        elif os.path.exists(self.ptsrcfiles[freq].spectral_index_catalog_fname(extrap_freq, component=component)):
            catalog = self.ptsrcfiles[freq].load_spectral_index_catalog(extrap_freq, component=component)
        else:
            self._init_ptsrclib(freq)
            catalog = self.ptsrclibs[freq].get_spectral_index_catalog(extrap_freq, component=component)
        return catalog


