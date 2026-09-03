import os
import numpy as np
from pixell import enmap
from hdsims import utils, simutils, fgcatalogs
from . import fgclean_info as fgi, fgutils, fgmaps, fgfilters, iterfgclean


class FGCleanClustersFiles:
    def __init__(self, output_dir, freqs, beam_fwhms, make_output_dirs=False, verbose=True, log=None):
        self.verbose = verbose
        self.log = log

        self.freqs = sorted(freqs.copy())
        self.beam_fwhms = beam_fwhms

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



    def iter_catalog_fname(self, iter_num, snr_threshold, include_all_profiles=False):
        if include_all_profiles:
            fname = iterfgclean.all_iter_clusters_catalog_fname(iter_num, snr_threshold, iter_catalog_dir=self.iter_catalog_dir)
        else:
            fname = iterfgclean.iter_clusters_catalog_fname(iter_num, snr_threshold, iter_catalog_dir=self.iter_catalog_dir)
        return fname


    def catalog_of_subtracted_clusters_fname(self):
        """final catalog of all subtracted clusters"""
        return os.path.join(self.output_dir, 'subtracted_clusters.csv')

    def map_of_subtracted_clusters_fname(self, freq=None, map_units='y', pixwin=False, beam=False):
        """final map of all subtracted clusters (beam and pixwin convolved)"""
        if map_units == 'y':
            fname = 'subtracted_clusters_ymap.fits'
        elif map_units == 'uK':
            if freq is None:
                raise ValueError(f"`{freq = }`. You must provide the map frequency (in GHz) for a map in units of uK.")
            freq = self._validate_freq(freq)
            fname_info = [f'{round(freq):03d}GHz_subtracted_clusters_uK']
            if pixwin or beam:
                if beam:
                    fname_info.append(f'beam{simutils.round_str(self.beam_fwhms[freq])}arcmin')
                if pixwin:
                    fname_info.append('pixwin')
                fname_info.append('convolved')
            fname_root = '_'.join(fname_info)
            fname = f'{fname_root}.fits'
        else:
            raise ValueError(f"`{map_units = }`. The `map_units` must be either `'y'` or `'uK'`.")
        return os.path.join(self.output_dir, fname)


    def cluster_subtracted_sim_fname(self, freq):
        """final cluster-subtracted sim"""
        freq = self._validate_freq(freq)
        return os.path.join(self.output_dir, f'{round(freq):03d}GHz_sim_after_cluster_subtraction.fits')


    def load_catalog_of_subtracted_clusters(self):
        fname = self.catalog_of_subtracted_clusters_fname()
        self.infomsg(f"loading {fname}")
        return fgcatalogs.load_catalog(fname)


    def load_map_of_subtracted_clusters(self, freq=None, map_units='y', pixwin=False, beam=False):
        fname = self.map_of_subtracted_clusters_fname(freq=freq, map_units=map_units, pixwin=pixwin, beam=beam)
        self.infomsg(f"loading {fname}")
        return enmap.read_map(fname)

    def load_cluster_subtracted_sim(self, freq):
        freq = self._validate_freq(freq)
        fname = self.cluster_subtracted_sim_fname(freq)
        if os.path.exists(fname):
            self.infomsg(f"loading {fname}")
        return enmap.read_map(fname)


    def load_maps_of_subtracted_clusters(self, freqs=None, pixwin=True, beam=True):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        subtracted_clusters_maps = {}
        for freq in freqs:
            subtracted_clusters_maps[freq] = self.load_map_of_subtracted_clusters(freq=freq, beam=beam, pixwin=pixwin, map_units='uK')
        return subtracted_clusters_maps

    def load_cluster_subtracted_sims(self, freqs=None):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        cluster_subtracted_sims = {}
        for freq in freqs:
            cluster_subtracted_sims[freq] = self.load_cluster_subtracted_sim(freq)
        return cluster_subtracted_sims




class FGCleanClusters(FGCleanClustersFiles):
    def __init__(self, output_dir, imaps, beam_fwhms, apod_width,
                 cluster_profiles, noise_maps_for_filter, snr_threshold_list,
                 iter_match_radius=fgi.clusters_iter_match_radius,
                 apply_apod=False, deconvolve_pixwin=True,
                 smooth_p2d_npix=fgi.p2d_smooth_npix,
                 rms_gw=fgi.rms_gw_clusters, rms_niter=fgi.rms_niter, rms_nsigma=fgi.rms_nsigma,
                 save_inv_noise_power2d=False,
                 mask_for_filtered_maps=None,
                 verbose=True, log=None):
        '''
        imaps and noise_maps should be dict
        '''
        super().__init__(output_dir, list(imaps.keys()), beam_fwhms, make_output_dirs=True, verbose=verbose, log=log)

        self.imaps = fgmaps._multifreq_mapdict_to_maps(imaps, freqs=self.freqs)
        self.shape = self.imaps[0].shape
        self.wcs = self.imaps[0].wcs
        self.apod_width = apod_width
        if any([freq not in self.beam_fwhms for freq in self.freqs]):
            freqs_without_beam = [freq for freq in self.freqs if (freq not in self.beam_fwhms)]
            raise ValueError(f"No key(s) in the `beam_fwhms` dictionary for the following frequencies (in GHz):"
                             f" {freqs_without_beam}. You must pass the beam full-width at half-maximum"
                             f" (in arcminutes) for each map frequency in the `input_sims` dictionary.")
        if any([freq not in noise_maps_for_filter for freq in self.freqs]):
            freqs_without_noise_map = [freq for freq in self.freqs if (freq not in noise_maps_for_filter)]
            raise ValueError(f"No key(s) in the `noise_maps_for_filter` dictionary for the following frequencies"
                             f" (in GHz): {freqs_without_noise_map}. For each map frequency in the `input_sims`"
                             f" dictionary, you must pass a corresponding map of the expected noise (i.e., the"
                             f" components in the `input_sims` map that are not the tSZ) at that frequency.")


        self.snr_threshold_list = snr_threshold_list.copy()
        self.iter_match_radius = iter_match_radius

        if save_inv_noise_power2d:
            inv_noise_power2d_dir = utils.mkdir(os.path.join(self.output_dir, 'inv_noise_power2d_for_filters'))
        else:
            inv_noise_power2d_dir = None
        self.filters = fgfilters.ClusterFilters(self.freqs, self.beam_fwhms, fgmaps._multifreq_mapdict_to_maps(noise_maps_for_filter, freqs=self.freqs),
                                      cluster_profiles, apod_width=self.apod_width, apply_apod=apply_apod,
                                      deconvolve_pixwin=deconvolve_pixwin, smooth_p2d_npix=smooth_p2d_npix,
                                      rms_gw=rms_gw, rms_niter=rms_niter, rms_nsigma=rms_nsigma,
                                      save_inv_noise_power2d=save_inv_noise_power2d, save_dir=inv_noise_power2d_dir,
                                      mask=mask_for_filtered_maps,
                                      verbose=self.verbose, log=self.log,
                                     )



    def make_cluster_ymap_from_catalog(self, catalog, profiles=None):
        '''
        profiles is dict of radial profile functions

        note: assumes default `ra_col='RADeg', dec_col='decDeg', amplitude_col='y_c', profile_name_col='template'`
        '''
        profiles = self.filters.profiles if (profiles is None) else profiles
        ymap = fgmaps.make_cluster_ymap_from_catalog(self.shape, self.wcs, catalog, profiles)
        return ymap


    def make_cluster_sim_from_catalog(self, catalog, freq, convolve_pixwin=True, convolve_beam=True,
                                      beam_fwhm=None, apply_apod=False, profiles=None):
        '''
        profiles is dict of radial profile functions

        note: assumes default `ra_col='RADeg', dec_col='decDeg', amplitude_col='y_c', profile_name_col='template'`
        '''
        if convolve_beam and (beam_fwhm is None):
            if freq in self.freqs:
                beam_fwhm = self.beam_fwhms[freq]
            else:
                raise ValueError(f"`{freq = }`, `{beam_fwhm = }`, `{convolve_beam = }`: You must provide the"
                                 f" {simutils.round_str(freq)} GHz beam full-width at half-maximum (in arcminutes).")
        profiles = self.filters.profiles if (profiles is None) else profiles
        omap = fgmaps.make_cluster_sim_from_catalog(self.shape, self.wcs, catalog, profiles, freq,
                                             apod_width=self.apod_width, apply_apod=apply_apod,
                                             convolve_pixwin=convolve_pixwin,
                                             convolve_beam=convolve_beam, beam_fwhm=beam_fwhm)
        return omap


    def make_cluster_sims_from_catalog(self, catalog, convolve_pixwin=True, convolve_beam=True, apply_apod=False, profiles=None):
        '''
        profiles is dict of radial profile functions

        note: assumes default `ra_col='RADeg', dec_col='decDeg', amplitude_col='y_c', profile_name_col='template'`
        '''
        profiles = self.filters.profiles if (profiles is None) else profiles
        omaps = fgmaps.make_cluster_sims_from_catalog(self.shape, self.wcs, catalog, profiles, self.freqs,
                                               apod_width=self.apod_width, apply_apod=apply_apod,
                                               convolve_pixwin=convolve_pixwin,
                                               convolve_beam=convolve_beam, beam_fwhms=self.beam_fwhms)
        return omaps





    def _save_maps_of_subtracted_clusters(self, subtracted_clusters_maps):
        '''will save pixwin- and beam-convolved maps (in uK) for each freq'''
        for freq in subtracted_clusters_maps.keys():
            fname = self.map_of_subtracted_clusters_fname(freq=freq, map_units='uK', beam=True, pixwin=True)
            if not os.path.exists(fname):
                enmap.write_map(fname, subtracted_clusters_maps[freq])
                self.infomsg(f"saved {fname}")


    def _save_cluster_subtracted_sims(self, cluster_subtracted_sims):
        for freq in cluster_subtracted_sims.keys():
            fname = self.cluster_subtracted_sim_fname(freq)
            if not os.path.exists(fname):
                enmap.write_map(fname, cluster_subtracted_sims[freq])
                self.infomsg(f"saved {fname}")




    def run_cluster_subtraction(self, save_maps_after_subtraction=False, save_subtracted_clusters_maps=False):
        catalog_fname = self.catalog_of_subtracted_clusters_fname()
        if os.path.exists(catalog_fname):
            # catalog of subtracted clusters:
            catalog = self.load_catalog_of_subtracted_clusters()
            # input maps after subtraction:
            if all([os.path.exists(self.cluster_subtracted_sim_fname(freq)) for freq in self.freqs]):
                sub_sims = self.load_cluster_subtracted_sims()
            else:
                # get map of subtracted clusters:
                cluster_map_fnames = {freq: self.map_of_subtracted_clusters_fname(freq=freq, map_units='uK', pixwin=True, beam=True) for freq in self.freqs}
                if all([os.path.exists(cluster_map_fnames[freq]) for freq in self.freqs]):
                    measured_cluster_sims = self.load_maps_of_subtracted_clusters()
                else:
                    measured_cluster_sims = self.make_cluster_sims_from_catalog(catalog, convolve_pixwin=True, convolve_beam=True)
                    if save_subtracted_clusters_maps:
                        self._save_maps_of_subtracted_clusters(measured_cluster_sims)
                sub_sims = {freq: self.imaps[i] - measured_cluster_sims[freq] for (i, freq) in enumerate(self.freqs)}
                if save_maps_after_subtraction:
                    self._save_cluster_subtracted_sims(sub_sims)
        else:
            catalog, sub_sims, measured_cluster_sims = iterfgclean.iteratively_measure_clusters(self.imaps, self.freqs, self.beam_fwhms,
                                                                                    self.filters, self.snr_threshold_list,
                                                                                    self.iter_catalog_dir,
                                                                                    iter_match_radius=self.iter_match_radius,
                                                                                    verbose=self.verbose, log=self.log)
            catalog.to_csv(catalog_fname)
            self.infomsg(f"saved {catalog_fname}")
            if save_maps_after_subtraction:
                self._save_cluster_subtracted_sims(sub_sims)
            if save_subtracted_clusters_maps:
                self._save_maps_of_subtracted_clusters(measured_cluster_sims)
        return catalog, sub_sims



    def get_catalog_of_subtracted_clusters(self):
        if os.path.exists(self.catalog_of_subtracted_clusters_fname()):
            catalog = self.load_catalog_of_subtracted_clusters()
        else:
            catalog, _ = self.run_cluster_subtraction()
        return catalog




    def get_maps_of_subtracted_clusters(self, freqs=None, beam=True, pixwin=True, save=False):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        cluster_map_fnames = {freq: self.map_of_subtracted_clusters_fname(freq=freq, map_units='uK', pixwin=pixwin, beam=beam) for freq in freqs}
        if all([os.path.exists(cluster_map_fnames[freq]) for freq in freqs]):
            subtracted_clusters_maps = self.load_maps_of_subtracted_clusters(freqs=freqs, pixwin=pixwin, beam=beam)
        else:
            catalog = self.get_catalog_of_subtracted_clusters()
            all_subtracted_clusters = self.make_cluster_sims_from_catalog(catalog, convolve_pixwin=pixwin, convolve_beam=beam)
            subtracted_clusters_maps = {freq: all_subtracted_clusters[freq] for freq in freqs}
            if save:
                for freq in freqs:
                    enmap.write_map(cluster_map_fnames[freq], subtracted_clusters_maps[freq])
                    self.infomsg(f"saved {cluster_map_fnames[freq]}")
        return subtracted_clusters_maps



    def get_map_of_subtracted_clusters(self, freq=None, map_units='y', pixwin=False, beam=False, save=False):
        fname = self.map_of_subtracted_clusters_fname(freq=freq, map_units=map_units, pixwin=pixwin, beam=beam)
        if os.path.exists(fname):
            omap = self.load_map_of_subtracted_clusters(freq=freq, map_units=map_units, pixwin=pixwin, beam=beam)
        else:
            catalog = self.get_catalog_of_subtracted_clusters()
            if map_units == 'y':
                omap = self.make_cluster_ymap_from_catalog(catalog)
            else:
                omap = self.make_cluster_sim_from_catalog(catalog, freq, convolve_pixwin=pixwin, convolve_beam=beam)
            if save:
                enmap.write_map(fname, omap)
                self.infomsg(f"saved {fname}")
        return omap



    def get_cluster_subtracted_sims(self, freqs=None):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        if all([os.path.exists(self.cluster_subtracted_sim_fname(freq)) for freq in freqs]):
            cluster_subtracted_sims = self.load_cluster_subtracted_sims(freqs=freqs)
        else:
            _, all_sub_sims = self.run_cluster_subtraction()
            cluster_subtracted_sims = {freq: all_sub_sims[freq] for freq in freqs}
        return cluster_subtracted_sims


    def get_cluster_subtracted_sim(self, freq):
        freq = self._validate_freq(freq)
        return self.get_cluster_subtracted_sims(freqs=[freq])[freq]


