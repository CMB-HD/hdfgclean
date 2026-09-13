import os
import warnings
import numpy as np
import pandas as pd
import yaml
from pixell import enmap
from hdsims import hdsims, siminfo as si, utils, simutils, fgcatalogs, maps
from . import mpi, fgclean_info as fgi, fgutils, fgfilters, fgclean


class HDFGCleanMaps(hdsims.HDSims, fgclean.FGClean):
    
    @classmethod
    def from_config(cls, config_fname, **kwargs):
        '''
        config_fname is path to yaml file
        kwargs can be passed to overwrite anything in config file
            (but note we don't check to make sure changes make sense....)

        NOTE : not using `yaml.safe_load` so we can load in, e.g., arrays, funcs, etc.
               see pyyaml docs for why you should be careful doing this
        '''
        with open(config_fname, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        # update `config` dict with any `kwargs`:
        config = {**config, **kwargs}

        # get required args:
        default_output_dir, _ = os.path.split(config_fname)
        output_dir = config.get('output_dir', default_output_dir)
        if 'hd_sims_dir' not in config:
            raise ValueError("You must provide the path to your `hd_sims_dir` by passing it "
                             "as a keyword argument, or saving it in the config file.")
        else:
            hd_sims_dir = config['hd_sims_dir']

        # get kwargs:
        required_arg_names = ['output_dir', 'hd_sims_dir']
        ignored_kwargs = ['pol', 'make_output_dirs', 'map_shape', 'map_wcs', 'calc_beam_solid_angle_funcs',
                          'sources_to_mask_before_fgclean_catalog', 'clusters_to_mask_after_fgclean_catalog']
        all_hdfgclean_kwargs = fgutils.dict_without_keys(fgutils._get_all_kwargs(cls), ignored_kwargs)
        allowed_kwarg_names = list(all_hdfgclean_kwargs.keys())
        # warn the user if any kwargs were removed:
        ignored_keys = [key for key in kwargs if (key not in [*required_arg_names, *allowed_kwarg_names])]
        if (len(ignored_keys) > 0) and mpi.is_rank0:
            warnings.warn(f"The following keys in the file {config_fname} "
                          f"or in `kwargs` will be ignored: {ignored_keys}")
        hdfgclean_kwargs = fgutils.dict_with_keys(config, allowed_kwarg_names)

        return cls(output_dir, hd_sims_dir, **hdfgclean_kwargs)


    @classmethod
    def save_config(cls, config_fname, hd_sims_dir, output_dir=None, overwrite=False, **kwargs):
        if os.path.exists(config_fname) and (not overwrite):
            raise FileExistsError(f"The file {config_fname} already exists. You may "
                                  "pass a new `config_fname`, or you may pass "
                                  "`overwrite=True` to overwrite the existing file.")
        # make sure any dict of noise maps is not dict of actual `pixell.enmap.ndmap`s:
        noise_map_kwarg_names = ['noise_maps_for_source_filters', 'noise_maps_for_cluster_filters',
                                 'noise_maps_for_source_mask_filters']
        for key in noise_map_kwarg_names:
            if key in kwargs:
                if any([isinstance(imap, enmap.ndmap) for imap in kwargs[key].values()]):
                    raise ValueError(f"The values in the `{key}` dictionary should be "
                                     "file names for each map, not the maps themselves.")
        # only save kwargs that can be passed to `HDFGClean`:
        ignored_kwargs = ['pol', 'make_output_dirs', 'map_shape', 'map_wcs', 'calc_beam_solid_angle_funcs',
                          'sources_to_mask_before_fgclean_catalog', 'clusters_to_mask_after_fgclean_catalog']
        all_hdfgclean_kwargs = fgutils.dict_without_keys(fgutils._get_all_kwargs(cls), ignored_kwargs)
        allowed_kwarg_names = list(all_hdfgclean_kwargs.keys())
        ignored_keys = [key for key in kwargs if (key not in allowed_kwarg_names)]
        if (len(ignored_keys) > 0) and mpi.is_rank0:
            warnings.warn(f"The following keys in `kwargs` will be ignored: {ignored_keys}")
        config = fgutils.dict_with_keys(kwargs, allowed_kwarg_names, copy=True)
        config['hd_sims_dir'] = hd_sims_dir
        if output_dir is None:
            output_dir, _ = os.path.split(config_fname)
        config['output_dir'] = output_dir
        if mpi.is_rank0:
            config_dir, _ = os.path.split(config_fname)
            utils.mkdir(config_dir)
            utils.save_yaml(config_fname, config, overwrite=overwrite)
            print(f'Configuration file has been saved to {config_fname}')
        mpi.comm.barrier()


    def __init__(self, output_dir, hd_sims_dir,
                 freqs=fgi.freqs, subtract_sources=True, subtract_clusters=True,
                 mask_bright_sources_before_fgclean=True, min_masked_flux=fgi.min_masked_flux, 
                 source_mask_apod_width=fgi.source_mask_apod_width, source_mask_radius=None,
                 mask_large_clusters_after_fgclean=True,
                 min_M500_to_mask=fgi.min_M500_to_mask, max_z_to_mask=fgi.max_z_to_mask, 
                 min_cluster_mask_radius=fgi.min_cluster_mask_radius, cluster_mask_apod_width=fgi.cluster_mask_apod_width,
                 **kwargs,
                 ):
        """
        kwargs are passed to hdsims and FGClean classes
          - NOTE : should include the `lowres_sims_dir` if the sims have not been generated
        """
        # HD sims:
        all_hdsims_kwargs = fgutils._get_all_kwargs(hdsims.HDSims)
        hdsims_kwargs = fgutils.dict_with_keys(kwargs, list(all_hdsims_kwargs.keys()))
        hdsims_kwargs = {**hdsims_kwargs, 'freqs': freqs, 'pol': False, #'verbose': verbose, 'log': log, 
                         'make_output_dirs': mpi.is_rank0}
        super().__init__(hd_sims_dir, **hdsims_kwargs)

        # FG cleaning:
        self.fgclean_dir = self.get_fgclean_dir(output_dir=output_dir, make_dir=mpi.is_rank0)
        self.subtract_sources = subtract_sources and (('cib' in self.map_components) or ('radio' in self.map_components))
        self.subtract_clusters = subtract_clusters and ('tsz' in self.map_components)
        if not (self.subtract_sources or self.subtract_clusters):
            raise ValueError(f"`{subtract_sources = }`, `{subtract_clusters = }`, and the map components are {self.map_components}, so there is nothing to do.")
        
        # pass dict of file names for input maps:
        imap_fnames = {}
        for freq in self.freqs:
            imap_fnames[freq] = self.get_sim_fname(freq=freq, beam=True, noise=True, shape=self.padded_shape, wcs=self.padded_wcs)
        # make sure all maps are saved:
        if mpi.is_rank0:
            for freq in self.freqs:
                if not os.path.exists(imap_fnames[freq]):
                    self.get_sim(freq=freq, beam=True, noise=True, shape=self.padded_shape, wcs=self.padded_wcs, save=True)
        mpi.comm.barrier()
        # and beam sizes for input maps:
        beam_fwhms = {freq: si.beam_fwhm[freq] for freq in self.freqs}
        
        # additional keyword arguments:
        all_fgclean_kwargs = fgutils._get_all_kwargs(fgclean.FGClean)
        fgclean_kwargs = fgutils.dict_with_keys(kwargs, list(all_fgclean_kwargs.keys()))
        noise_maps_kwargs = kwargs.get('noise_maps_kwargs', {})  
        # override / add our defaults
        noise_map_kwarg_names = ['hd_noise_sims_for_filters_dir', 'noise_maps_for_source_filters',
                                 'noise_maps_for_cluster_filters', 'noise_maps_for_source_mask_filters']
        if not any([key in fgclean_kwargs for key in noise_map_kwarg_names]):
            fgclean_kwargs['hd_noise_sims_for_filters_dir'] = hd_sims_dir
            if self.lowres_sims_dir is not None: # required to generate sims for noise maps if they don't exist
                noise_maps_kwargs['lowres_sims_dir'] = self.lowres_sims_dir
        noise_maps_kwargs = {**noise_maps_kwargs, 'freqs': self.freqs, 'components': self.map_components, 
                             'verbose': self.verbose, 'log': self.log}
        fgclean_kwargs = {**fgclean_kwargs, 'map_shape': self.padded_shape, 'map_wcs': self.padded_wcs,
                          'map_apod_width': self.map_apod_width, 
                          'subtract_sources': self.subtract_sources,
                          'subtract_clusters': self.subtract_clusters,
                          'calc_beam_solid_angle_funcs': True,
                          'noise_maps_kwargs': noise_maps_kwargs} 
        # if subtracting clusters and not using default set of filters, need to provide `cluster_profiles_info`
        cluster_profiles = fgclean_kwargs.get('cluster_profiles', None)
        if self.subtract_clusters and (cluster_profiles is not None):
            cluster_profiles_info = fgclean_kwargs.get('cluster_profiles_info', None)
            default_cluster_profile_names = list(fgfilters.get_default_gauss_cluster_profiles_dict().keys())
            if (set(cluster_profiles.keys()) != set(default_cluster_profile_names)) and (cluster_profiles_info is None):
                raise ValueError(f"A dictionary of `cluster_profiles` was passed but `{cluster_profiles_info=}`."
                                 " When using a non-default set of cluster profiles, `cluster_profiles_info` must also"
                                 " be passed; this is a (short) unique string used in the output file names which"
                                 " describes the set of cluster profiles.")
        # masking known bright sources before FG cleaning:
        if mask_bright_sources_before_fgclean:
            fgclean_kwargs['sources_to_mask_before_fgclean_catalog'] = self.get_sources_masked_before_fgclean_catalog(min_masked_flux=min_masked_flux, 
                                                                                                                      source_mask_apod_width=source_mask_apod_width, 
                                                                                                                      source_mask_radius=source_mask_radius)
        # masking known large/massive/nearby clusters after FG cleaning:
        if self.subtract_clusters and mask_large_clusters_after_fgclean:
            fgclean_kwargs['clusters_to_mask_after_fgclean_catalog'] = self.get_large_clusters_masked_after_fgclean_catalog(min_M500_to_mask=min_M500_to_mask, 
                                                                                                                            max_z_to_mask=max_z_to_mask, 
                                                                                                                            min_cluster_mask_radius=min_cluster_mask_radius, 
                                                                                                                            cluster_mask_apod_width=cluster_mask_apod_width)
        
        fgclean.FGClean.__init__(self, self.fgclean_dir, imap_fnames, beam_fwhms, **fgclean_kwargs)
        
        
    
    
    
    
    def get_fgclean_dir(self, output_dir=None, make_dir=True):
        patch_ctr_info = f'ra{simutils.round_str(self.ra_ctr)}dec{simutils.round_str(self.dec_ctr)}'
        patch_size_info = f'{simutils.round_str(self.width)}x{simutils.round_str(self.height)}deg'
        sim_component_info = self._map_component_list2str()
        if simutils.has_cmb(self.map_components):
            cmb_info = self._map_component_list2str(components=[simutils.cmb_component_name(self.map_components)])
            sim_component_info = sim_component_info.replace(cmb_info, f'cmbseed{self.cmb_seed}')
        freq_info = '_'.join([f'{round(freq):03d}' for freq in sorted(self.freqs)])
        dir_name = '_'.join(['hdfgclean', patch_ctr_info, patch_size_info, sim_component_info, freq_info])
        if output_dir is not None:
            dir_name = os.path.join(output_dir, dir_name)
        if make_dir:
            dir_name = utils.mkdir(dir_name)
        return dir_name
    

    def remove_objects_in_apodized_region(self, catalog, patch_num=None):
        if patch_num is None:
            # make sure we have same items in catalog for (un-apodized region of) full map 
            # as we do for combination of catalogs from all patches
            icat_cols = catalog.columns.values
            ocat = catalog.copy()
            ocat = maps.add_pixel_coords_to_catalog(catalog, self.patches.shape, self.patches.wcs)
            xmin, xmax, ymin, ymax = self.patches._calc_pixel_lims_for_region(self.shape, self.wcs)
            ocat = ocat[ocat['x_pixel'].between(xmin, xmax) & ocat['y_pixel'].between(ymin, ymax)][icat_cols].copy()
        else:
            ocat = self.patches.trim_patch_catalog_to_use(catalog, patch_num, include_apodized_region=False)
        return ocat
    
    
    # --- sims after FG cleaning ---

    def get_sim(self, freq=None, beam=False, noise=False, 
                subtract_sources=False, subtract_clusters=False, 
                save=False, **kwargs):
        if subtract_sources or subtract_clusters:
            return self.get_fgcleaned_sim(freq, beam=beam, noise=noise, subtract_sources=subtract_sources, 
                                                        subtract_clusters=subtract_clusters, save=save, **kwargs)
        else:
            return super().get_sim(freq=freq, beam=beam, noise=noise, save=save, **kwargs)
        
        
    def get_fgcleaned_sim(self, freq, beam=False, noise=False, 
                                  subtract_sources=True, subtract_clusters=True, 
                                  save=False, **kwargs):
        '''
        kwargs are passed to `get_sim` method
        
        note: if `save=True`, will also save maps of subtracted srcs/clusters
        '''
        sim_fname = self.get_fgcleaned_sim_fname(freq=freq, beam=beam, noise=noise, 
                                                         subtract_sources=subtract_sources, subtract_clusters=subtract_clusters, **kwargs)
        if os.path.exists(sim_fname):
            shape = self.get_kwarg('shape', **kwargs)
            wcs = self.get_kwarg('wcs', **kwargs)
            sim = enmap.project(enmap.read_map(sim_fname), shape, wcs)
        else:
            # get sim before subtraction:
            sim = super().get_sim(freq=freq, beam=beam, noise=noise, save=save, **kwargs)
            # get map of sources and/or clusters to subtract:
            map_to_subtract = enmap.zeros(sim.shape[-2:], sim.wcs)
            if self._can_subtract_sources(**kwargs) and subtract_sources:
                sources_to_subtract = self.get_map_of_subtracted_sources(freq, pixwin=True, beam=beam, include_apodized_region=True, save=save)
                map_to_subtract += enmap.project(sources_to_subtract, map_to_subtract.shape, map_to_subtract.wcs)
            if self._can_subtract_clusters(**kwargs) and subtract_clusters:
                clusters_to_subtract = self.get_map_of_subtracted_clusters(freq=freq, pixwin=True, beam=beam, include_apodized_region=True, save=save)
                map_to_subtract += enmap.project(clusters_to_subtract, map_to_subtract.shape, map_to_subtract.wcs)
            # subtract them:
            if len(sim.shape) > 2: # sim has T, Q, and U
                sim[0] -= map_to_subtract
            else:
                sim -= map_to_subtract
            # save the sim
            if save:
                enmap.write_map(sim_fname, sim)
        return sim
    
    
    def get_fgcleaned_sims(self, freqs=None, beam=False, noise=False, 
                                  subtract_sources=True, subtract_clusters=True, 
                                  save=False, **kwargs):
        freqs = self.freqs if (freqs is None) else self._validate_freqs(freqs)
        sims = {}
        for freq in freqs:
            sims[freq] = self.get_fgcleaned_sim(freq, beam=beam, noise=noise, subtract_sources=subtract_sources, 
                                                        subtract_clusters=subtract_clusters, save=save, **kwargs)
        return sims
    
    
    def get_fgcleaned_sim_fname(self, freq, beam=False, noise=False, 
                                        subtract_sources=True, subtract_clusters=True, 
                                        sources_sub_info=None, clusters_sub_info=None,
                                        **kwargs):
        '''kwargs are passed to `get_sim_fname` method of `HDSims`'''
        sim_before_subtraction_fname = self.get_sim_fname(freq=freq, beam=beam, noise=noise, **kwargs)
        sub_info = self._get_subtracted_components_info(subtract_sources=subtract_sources, subtract_clusters=subtract_clusters, 
                                                        sources_sub_info=sources_sub_info, clusters_sub_info=clusters_sub_info, **kwargs)
        if sub_info is None:
            fname = sim_before_subtraction_fname
        else:
            # remove path and file extension
            sim_before_subtraction_fname = os.path.split(sim_before_subtraction_fname)[1]
            fname_root = os.path.splitext(sim_before_subtraction_fname)[0]
            # add info about what was subtracted and save to the results directory
            fname = os.path.join(self.fgcleaned_maps_dir(), f'{fname_root}_sub_{sub_info}.fits')
        return fname
    
    
    def _get_subtracted_components_info(self, subtract_sources=True, subtract_clusters=True, sources_sub_info=None, clusters_sub_info=None, **kwargs):
        sub_component_info = []
        if self._can_subtract_sources(**kwargs) and subtract_sources:
            if sources_sub_info is not None:
                sub_component_info.append(sources_sub_info)
            sub_component_info.append('sources')
        if self._can_subtract_clusters(**kwargs) and subtract_clusters:
            if clusters_sub_info is not None:
                sub_component_info.append(clusters_sub_info)
            sub_component_info.append('clusters')
        if len(sub_component_info) > 0:
            sub_info = '_'.join(sub_component_info)
        else:
            sub_info = None
        return sub_info
    
    
    def _can_subtract_sources(self, **kwargs):
        '''
        note : kwargs are allowed (so they can be easily passed around), but only the `components` kwarg is used
        '''
        sim_components = self.get_kwarg('components', **kwargs)
        sim_srcs = [c for c in sim_components if (c in ['cib', 'radio'])]
        imap_srcs = [c for c in self.map_components if (c in ['cib', 'radio'])]
        can_subtract_sources = (len(sim_srcs) > 0) and self.subtract_sources
        if can_subtract_sources and (len(sim_srcs) != len(imap_srcs)):
            sim_info = " and ".join([f"'{c}'" for c in sim_srcs]) if (len(sim_srcs) > 1) else f"only '{sim_srcs[0]}'"
            imap_info = " and ".join([f"'{c}'" for c in imap_srcs]) if (len(imap_srcs) > 1) else f"only '{imap_srcs[0]}'"
            raise ValueError("Cannot subtract point sources that were found in a map with"
                             f" {imap_info} from a map with {sim_info}.")
        return can_subtract_sources
    
    
    def _can_subtract_clusters(self, **kwargs):
        '''
        note : kwargs are allowed (so they can be easily passed around), but only the `components` kwarg is used
        '''
        sim_components = self.get_kwarg('components', **kwargs)
        can_subtract_clusters = ('tsz' in sim_components) and self.subtract_clusters
        return can_subtract_clusters
        
    
    def sources_masked_before_fgclean_catalog_fname(self):
        return os.path.join(self.fgclean_dir, 'sources_masked_before_fgclean.csv')
    
    
    def get_sources_masked_before_fgclean_catalog(self, min_masked_flux=1000, source_mask_apod_width=1, source_mask_radius=None):
        srcs_to_mask_fname = self.sources_masked_before_fgclean_catalog_fname()
        srcs_to_mask = None
        if self.subtract_sources:
            if not os.path.exists(srcs_to_mask_fname):
                if mpi.is_rank0:
                    # load in catalogs of all sources in the maps:
                    all_true_radio = self.get_catalog('radio')
                    all_true_radio['idx'] = list(range(len(all_true_radio)))
                    all_true_radio['component'] = 'radio'
                    all_true_cib = self.get_catalog('cib')
                    all_true_cib['idx'] = list(range(len(all_true_cib)))
                    all_true_cib['component'] = 'cib'
                    # identify radio/CIB sources with flux > min_masked_flux at any frequency:
                    radio_src_idxs_to_mask = []
                    cib_src_idxs_to_mask = []
                    for freq in self.freqs:
                        radio_src_idxs_to_mask = [*radio_src_idxs_to_mask, *all_true_radio[all_true_radio[f'fluxmJy_{freq}GHz'].ge(min_masked_flux)]['idx'].values]
                        cib_src_idxs_to_mask = [*cib_src_idxs_to_mask, *all_true_cib[all_true_cib[f'fluxmJy_{freq}GHz'].ge(min_masked_flux)]['idx'].values]
                    radio_srcs_to_mask = all_true_radio[all_true_radio['idx'].isin(radio_src_idxs_to_mask)]
                    cib_srcs_to_mask = all_true_cib[all_true_cib['idx'].isin(cib_src_idxs_to_mask)]
                    # combine into a single catalog:
                    flux_cols = [f'fluxmJy_{freq}GHz' for freq in self.freqs]
                    cols = ['RADeg', 'decDeg', *flux_cols, 'component']
                    if (len(radio_srcs_to_mask) > 0) or (len(cib_srcs_to_mask) > 0):
                        srcs_to_mask = fgcatalogs.combine_catalogs([radio_srcs_to_mask[cols], cib_srcs_to_mask[cols]])
                    else:
                        srcs_to_mask = pd.DataFrame({col: [] for col in cols})
                    # add info about radius and apodization width for holes in the mask:
                    for freq in self.freqs:
                        if np.isscalar(source_mask_radius):
                            srcs_to_mask[f'mask_radius_{freq}GHz'] = source_mask_radius
                        elif (source_mask_radius is None) or (freq not in source_mask_radius):
                            srcs_to_mask[f'mask_radius_{freq}GHz'] = 4 * utils.fwhm2sigma(si.beam_fwhm[freq])
                        else:
                            srcs_to_mask[f'mask_radius_{freq}GHz'] = source_mask_radius[freq]
                    srcs_to_mask['apod_width'] = source_mask_apod_width
                    # save the catalog (even if it's empty):
                    srcs_to_mask.to_csv(srcs_to_mask_fname)
                mpi.comm.barrier()
            srcs_to_mask = fgcatalogs.load_catalog(srcs_to_mask_fname)
        return srcs_to_mask
    
    
    def large_clusters_masked_after_fgclean_catalog_fname(self):
        return os.path.join(self.fgclean_dir, 'large_clusters_masked_after_fgclean.csv')
    
    
    def get_large_clusters_masked_after_fgclean_catalog(self, min_M500_to_mask=2e14, max_z_to_mask=0.2, 
                                                        min_cluster_mask_radius=5, cluster_mask_apod_width=5):
        clusters_to_mask_fname = self.large_clusters_masked_after_fgclean_catalog_fname()
        clusters_to_mask = None
        if self.subtract_clusters and (min_M500_to_mask is not None):
            if not os.path.exists(clusters_to_mask_fname):
                if mpi.is_rank0:
                    pars = simutils.get_default_cambparams_for_sim()
                    all_true_sz = self.get_catalog('tsz')
                    all_true_sz['angular_radius_R500'] = fgutils.angular_radius(all_true_sz['z'].values, all_true_sz['R500'].values, pars)

                    clusters_to_mask = all_true_sz[all_true_sz['M500'].ge(min_M500_to_mask) &  all_true_sz['z'].le(max_z_to_mask) & all_true_sz['angular_radius_R500'].ge(min_cluster_mask_radius)].copy()
                    clusters_to_mask['mask_radius'] = clusters_to_mask['angular_radius_R500'].values.copy()
                    clusters_to_mask['apod_width'] = cluster_mask_apod_width
                    clusters_to_mask = clusters_to_mask[['RADeg', 'decDeg', 'mask_radius', 'apod_width', 'M500', 'z']]
                    
                    clusters_to_mask.to_csv(clusters_to_mask_fname)
                mpi.comm.barrier()
            clusters_to_mask = fgcatalogs.load_catalog(clusters_to_mask_fname)
        return clusters_to_mask
    
    
    
    ############### overriding inherited FGClean methods ###############
    
    # note : re-defining this here so other hdsims classes use it
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
        
    
    # same as method in FGClean, but uses inner, un-apodized region
    def calculate_masked_map_fractions(self, freqs=fgi.spectra_freqs, max_unmasked_value=1, shape=None, wcs=None, as_percent=True, save_masks=True, **kwargs):
        masked_fractions = super().calculate_masked_map_fractions(freqs=freqs, max_unmasked_value=max_unmasked_value, 
                                                     shape=self.shape, wcs=self.wcs, as_percent=as_percent, 
                                                     save_masks=save_masks, **kwargs)
        return masked_fractions
    
    
    def get_catalog_of_subtracted_sources(self, freq, patch_num=None, include_apodized_region=True):
        if patch_num is None:
            catalog = super().get_catalog_of_subtracted_sources(freq)
        else:
            catalog = self._patch_catalog_of_subtracted_sources(freq, patch_num=patch_num)
        if not include_apodized_region:
            catalog = self.remove_objects_in_apodized_region(catalog, patch_num=patch_num)
        return catalog
    
    def get_catalog_of_subtracted_clusters(self, patch_num=None, include_apodized_region=True):
        if patch_num is None:
            catalog = super().get_catalog_of_subtracted_clusters()
        else:
            catalog = self._patch_catalog_of_subtracted_clusters(patch_num=patch_num)
        if not include_apodized_region:
            catalog = self.remove_objects_in_apodized_region(catalog, patch_num=patch_num)
        return catalog
    
    
    def get_map_of_subtracted_sources(self, freq, pixwin=True, beam=True, save=False, include_apodized_region=True):
        omap = super().get_map_of_subtracted_sources(freq, pixwin=pixwin, beam=beam, save=save)
        if not include_apodized_region:
            omap = enmap.project(omap, self.shape, self.wcs)
        return omap
    
    def get_maps_of_subtracted_sources(self, freqs=None, pixwin=True, beam=True, save=False, include_apodized_region=True):
        omaps = super().get_maps_of_subtracted_sources(freqs=freqs, pixwin=pixwin, beam=beam, save=save)
        if not include_apodized_region:
            for freq in omaps:
                omaps[freq] = enmap.project(omaps[freq], self.shape, self.wcs)
        return omaps
    
    
    def get_map_of_subtracted_clusters(self, freq, pixwin=True, beam=True, save=False, include_apodized_region=True):
        omap = super().get_map_of_subtracted_clusters(freq, pixwin=pixwin, beam=beam, save=save)
        if not include_apodized_region:
            omap = enmap.project(omap, self.shape, self.wcs)
        return omap
    
    def get_maps_of_subtracted_clusters(self, freqs=None, pixwin=True, beam=True, save=False, include_apodized_region=True):
        omaps = super().get_maps_of_subtracted_clusters(freqs=freqs, pixwin=pixwin, beam=beam, save=save)
        if not include_apodized_region:
            for freq in omaps:
                omaps[freq] = enmap.project(omaps[freq], self.shape, self.wcs)
        return omaps
    
    
    def get_map_of_subtracted_fgs(self, freq, pixwin=True, beam=True, save=False, include_apodized_region=True):
        omap = super().get_map_of_subtracted_fgs(freq, pixwin=pixwin, beam=beam, save=save)
        if not include_apodized_region:
            omap = enmap.project(omap, self.shape, self.wcs)
        return omap
    
    def get_maps_of_subtracted_fgs(self, freqs=None, pixwin=True, beam=True, save=False, include_apodized_region=True):
        omaps = super().get_maps_of_subtracted_fgs(freqs=freqs, pixwin=pixwin, beam=beam, save=save)
        if not include_apodized_region:
            for freq in omaps:
                omaps[freq] = enmap.project(omaps[freq], self.shape, self.wcs)
        return omaps
    
    
    def fgcleaned_map_fname(self, freq, subtract_sources=True, subtract_clusters=True):
        fname = self.get_fgcleaned_sim_fname(freq, beam=True, noise=True, 
                                                     subtract_sources=subtract_sources, 
                                                     subtract_clusters=subtract_clusters)
        return fname
   
    
    
        

