import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from hdsims import utils, simutils, fgcatalogs, maps
from hd_mock_data import hd_data
from . import mpi, fgclean_info as fgi, fgmasks, fgresults, fgplots, hdfgclean_results

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['figure.dpi'] = 250
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.xmargin'] = 0.025
plt.rcParams['axes.ymargin'] = 0.025
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['figure.figsize'] = (5, 3)
if 'planck' not in mpl.colormaps:
    colorize.mpl_setdefault('planck')
plt.rcParams['image.cmap'] = 'planck'


class HDFGClean(hdfgclean_results.HDFGCleanResults):

    def run_hdfgclean(self, patch_nums=None, combine_patches=True,
                      make_masks=True, mask_freqs=fgi.spectra_freqs,
                      take_power=True, take_power_of_patches=False, 
                      spectra_freqs=fgi.spectra_freqs, bin_cl=False, bin_dl=True,
                      match_to_true_catalogs=True, freqs_to_match=None, min_snr_to_match=None,
                      source_match_radius_dict=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, 
                      cluster_match_radius=fgi.cluster_match_radius,
                      make_plots=True, show_plots=True, save_plots=True, use_fig_settings=False, 
                      save_maps_after_subtraction=True, save_subtracted_sources_maps=True, 
                      save_subtracted_clusters_maps=True, **kwargs):
        '''
        if `make_plots=True`, just makes whichever plots are possible (e.g., if `take_power=False`, won't plot spectra)
        kwargs are for the mask and power spectra
        '''
        # make sure we have everything we need to make the plots
        for freq in fgi.spectra_freqs:
            if make_plots and take_power and (freq not in mask_freqs):
                mask_freqs.append(freq)
            if make_plots and take_power and (freq not in spectra_freqs):
                spectra_freqs.append(freq)
            if make_plots and match_to_true_catalogs and (freqs_to_match is not None) and (freq not in freqs_to_match):
                freqs_to_match.append(freq)
            
        # kwargs to pass to `run_fgclean_on_patch`:
        fgclean_kwargs = {**kwargs, 'take_power': take_power_of_patches, 'spectra_freqs': spectra_freqs, 
                          'bin_cl': bin_cl, 'bin_dl': bin_dl, 'match_to_true_catalogs': match_to_true_catalogs,
                          'min_snr_to_match': min_snr_to_match,
                          'freqs_to_match': freqs_to_match, 'cluster_match_radius': cluster_match_radius, 
                          'source_match_radius_dict': source_match_radius_dict,
                          'max_abs_flux_diff_vs_err': max_abs_flux_diff_vs_err}
        self.run_fgclean(patch_nums=patch_nums, combine_patches=combine_patches,
                         make_masks=make_masks, mask_freqs=mask_freqs,
                         save_maps_after_subtraction=save_maps_after_subtraction,
                         save_subtracted_sources_maps=save_subtracted_sources_maps, 
                         save_subtracted_clusters_maps=save_subtracted_clusters_maps,
                         **fgclean_kwargs)
        
        if self.all_patches_fgcleaned() and combine_patches:
            if match_to_true_catalogs and mpi.is_rank0:
                self.save_combined_matched_patch_catalogs(freqs_to_match=freqs_to_match, min_snr=min_snr_to_match,
                                                          source_match_radius_dict=source_match_radius_dict, 
                                                          max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err, 
                                                          cluster_match_radius=cluster_match_radius)
            if take_power:
                spectra_kwargs = {**kwargs, 'patch_num': None, 'shape': self.shape, 'wcs': self.wcs}
                self.get_sim_power_before_and_after_fgclean(spectra_freqs=spectra_freqs, bin_cl=bin_cl, bin_dl=bin_dl, 
                                                            apply_mask=make_masks, mask_freqs=mask_freqs, 
                                                            **spectra_kwargs)
                # if making the plots, also save power spectra of sources on a smaller patch:
                if make_plots:
                    self._save_sources_spectra_for_plot()

            if make_plots:
                self.hdfgclean_plots(show=show_plots, save=save_plots, plot_spectra=take_power, 
                                     plot_match_to_true=match_to_true_catalogs, use_fig_settings=use_fig_settings,
                                     min_snr_to_match=min_snr_to_match, 
                                     source_match_radius_dict=source_match_radius_dict, 
                                     max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err, 
                                     cluster_match_radius=cluster_match_radius, **kwargs)
        mpi.comm.barrier()
    
    
    def run_fgclean_on_patch(self, patch_num=0, make_masks=True, mask_freqs=fgi.spectra_freqs, 
                             take_power=False, spectra_freqs=fgi.spectra_freqs, bin_cl=False, bin_dl=True,
                             match_to_true_catalogs=False, freqs_to_match=None, min_snr_to_match=None,
                             source_match_radius_dict=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, 
                             cluster_match_radius=fgi.cluster_match_radius, **kwargs):
        '''
        NOTE: recommended to match catalogs for each patch, but NOT to take power on patches
        
        kwargs are for the masks + spectra
        '''
        super().run_fgclean_on_patch(patch_num=patch_num, make_masks=make_masks, mask_freqs=mask_freqs, **kwargs)
        if take_power:
            self.get_patch_power_before_and_after_fgclean(patch_num, spectra_freqs=spectra_freqs, 
                                                          bin_cl=bin_cl, bin_dl=bin_dl, 
                                                          apply_mask=make_masks, mask_freqs=mask_freqs, **kwargs)
        if match_to_true_catalogs:
            if self.subtract_sources:
                if freqs_to_match is None:
                    match_freqs = self.freqs
                else:
                    match_freqs = [freq for freq in freqs_to_match if (freq in self.freqs)]
                    if len(match_freqs) == 0:
                        raise ValueError(f"`{match_to_true_catalogs = }` and `{freqs_to_match = }`, but the map "
                                         f"frequencies used for FG cleaning are {self.freqs}. Each frequency (in "
                                         "GHz) in the `freqs_to_match` list must correspond to a map frequency.")
                for freq in match_freqs:
                    match_radius = None # will be automatically set based on beam size at this freq
                    if source_match_radius_dict is not None:
                        if freq in source_match_radius_dict:
                            match_radius = source_match_radius_dict[freq]
                    self.match_to_true_sources_catalog(freq, patch_num=patch_num, match_radius=match_radius, 
                                                       max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err, 
                                                       min_snr=min_snr_to_match)
            if self.subtract_clusters:
                self.match_to_true_clusters_catalog(patch_num=patch_num, match_radius=cluster_match_radius, 
                                                    min_snr=min_snr_to_match)
    
    
    def run_fgclean_on_patches(self, patch_nums=None, make_masks=True, mask_freqs=fgi.spectra_freqs, **kwargs):
        '''kwargs are for mask + power spectra (lmax, pol)'''
        # if running in parallel, make sure all output directories, etc.
        # are created by a single mpi process before doing the FG cleaning:
        if mpi.size > 1:
            # when only taking power of full map, output files/directories 
            # can be generated after FG cleaning; if taking power of each 
            # patch, need to create output directories and save apod 
            # windows and  inverse mode-coupling matrices for each row of 
            # patches (i.e., patches centered at same dec. with same height)
            take_power_of_patches = kwargs.get('take_power', False)
            if take_power_of_patches:
                # make sure output directories for power spectra are created
                self._make_spectra_dirs_and_save_files(apply_mask=make_masks, **kwargs)
                # save an apod window and inverse mode-coupling matrices for patches in each row
                self._calculate_mode_coupling_for_patches(patch_nums=patch_nums, **kwargs)
            # if matching to true catalogs, save the true catalogs for the full map:
            match_to_true_catalogs = kwargs.get('match_to_true_catalogs', False)
            if match_to_true_catalogs and mpi.is_rank0:
                if self.subtract_sources:
                    self.matched_sources_catalogs_dir(make_dir=True)
                    self.get_true_sources_catalog(save=True)
                if self.subtract_clusters:
                    self.matched_clusters_catalogs_dir(make_dir=True)
                    self.get_true_clusters_catalog(save=True)
            mpi.comm.barrier()
        super().run_fgclean_on_patches(patch_nums=patch_nums, make_masks=make_masks, mask_freqs=mask_freqs, **kwargs)


    # ----- plots : -----
        
    def hdfgclean_plots(self, show=True, save=True, plot_spectra=True, 
                        plot_match_to_true=True, use_fig_settings=True, 
                        min_snr_to_match=None, source_match_radius_dict=None, 
                        max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, 
                        cluster_match_radius=fgi.cluster_match_radius, **kwargs):
        plt_patch_num = self.get_patch_num(ra=self.ra_ctr, dec=self.dec_ctr)
        if plot_spectra and plot_match_to_true: # need to take power spectra of the patch
            # save an apod window and inverse mode-coupling matrices for patches in each row
            kwargs = {**kwargs, 'bin_dl': True, 'bin_cl': False}
            self._calculate_mode_coupling_for_patches(patch_nums=[plt_patch_num], **kwargs)
        if mpi.is_rank0:
            plot_dir = self.plots_dir(make_dir=True) # make the plot dir
            self.plot_fgcleaned_maps_with_cmb(patch_num=plt_patch_num, show=show, save=save)
            self.plot_fgcleaned_maps_without_cmb(patch_num=plt_patch_num, show=show, save=save)
            if plot_spectra:
                self.plot_fgcleaned_spectra(show=show, save=save, **kwargs)
                coadd_label = 'Sim-based forecast'
                if use_fig_settings:
                    coadd_label = f'{coadd_label} (this work)'
                self.plot_coadded_fgcleaned_spectra(label=coadd_label, show=show, save=save, **kwargs)
            if self.subtract_sources:
                # flux measurements for spectral index and/or measured vs true:
                has_cib = ('cib' in self.map_components)
                can_make_spectral_index_plot = (148 in self.freqs) and (277 in self.freqs) and has_cib
                if plot_match_to_true and (source_match_radius_dict is not None):
                    source_match_radius = source_match_radius_dict[90]
                else:
                    source_match_radius = None
                if plot_match_to_true and can_make_spectral_index_plot and use_fig_settings: # combine them
                    self.flux_measurement_plot(freq_for_index_plot=148, component_for_index_plot='cib', 
                                               freq_for_flux_plot=90, min_match_snr=min_snr_to_match,
                                               match_radius=source_match_radius, 
                                               max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err,
                                               patch_num=plt_patch_num, show=show, save=save)
                else:
                    if plot_match_to_true:
                        self.measured_vs_true_flux_plot(90, patch_num=plt_patch_num, show=show, save=save,
                                                        min_match_snr=min_snr_to_match,
                                                        match_radius=source_match_radius, 
                                                        max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
                    if can_make_spectral_index_plot:
                        self.spectral_index_plot(148, 'cib', patch_num=plt_patch_num, show=show, save=save)
                # histograms and spectra of sources:
                if plot_match_to_true:
                    if plot_spectra:
                        self.sources_plot(freqs=fgi.spectra_freqs, patch_num=plt_patch_num, 
                                          show=show, save=save,
                                          min_snr=min_snr_to_match, 
                                          source_match_radius_dict=source_match_radius_dict,  
                                          max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
                    else:
                        self.true_flux_histogram_plots(freqs=fgi.spectra_freqs, patch_num=plt_patch_num, 
                                                       show=show, save=save, min_snr=min_snr_to_match, 
                                                       source_match_radius_dict=source_match_radius_dict,  
                                                       max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
            if self.subtract_clusters and plot_match_to_true:
                self.cluster_mass_vs_redshift_plot(match_radius=cluster_match_radius, 
                                                   min_snr=min_snr_to_match, 
                                                   show=show, save=save)
                if use_fig_settings:
                    self.cluster_completeness_per_mass_redshift_plot(match_radius=cluster_match_radius, 
                                                                     min_snr=min_snr_to_match, 
                                                                     show=show, save=save)


    def plots_dir(self, make_dir=False):
        odir = os.path.join(self.results_dir(), 'plots')
        if make_dir:
            utils.mkdir(odir)
        return odir
    
    
    def plot_fgcleaned_maps_without_cmb(self, freqs=fgi.spectra_freqs, 
                                        plot_patch=True, patch_num=None, 
                                        noise=False, show=True, save=True, 
                                        fname=None, clims=50, dpi=500):
        if plot_patch and (patch_num is None):
            patch_num = self.get_patch_num(ra=self.ra_ctr, dec=self.dec_ctr)
        freqs = self._validate_freqs(freqs)
        plt_components = []
        plt_component_names = []
        if self.subtract_clusters:
            plt_components.append('tsz')
            plt_component_names.append('tSZ')
        if self.subtract_sources:
            if 'cib' in self.map_components:
                plt_components.append('cib')
                plt_component_names.append('CIB')
            if 'radio' in self.map_components:
                plt_components.append('radio')
                plt_component_names.append('Radio')
        plt_component_label = '+'.join(plt_component_names)
        
        if save:
            if fname is None:
                plt_map_info = self._map_component_list2str(components=plt_components)
                if patch_num is not None:
                    plt_map_info = f'{plt_map_info}_patch{patch_num:02d}'
                plt_freq_info = '_'.join([f'{round(freq):03d}' for freq in freqs])
                fname = os.path.join(self.plots_dir(), f'{plt_freq_info}_{plt_map_info}_maps.pdf')
        else:
            fname = None
        
        self.infomsg(f"plotting {plt_component_label} maps before and after FG cleaning for {freqs = }")
        map_kwargs = {'components': plt_components, 'beam': True, 'noise': noise}
        if patch_num is not None:
            patch_shape, patch_wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=False)
            map_kwargs['shape'] = patch_shape
            map_kwargs['wcs'] = patch_wcs
        maps_before = {}
        maps_after = {}
        for freq in freqs:
            maps_before[freq] = self.get_sim(freq=freq, **map_kwargs)
            maps_after[freq] = self.get_sim(freq=freq, subtract_sources=True, subtract_clusters=True, **map_kwargs)
        plt_output = fgplots.plot_fgcleaned_maps_without_cmb(maps_before, maps_after, fg_label=plt_component_label, 
                                                             clims=clims, dpi=dpi, show=show, fname=fname)
        if save:
            self.infomsg(f"saved {fname}")
        if not show:
            return plt_output


    def plot_fgcleaned_maps_with_cmb(self, freqs=fgi.spectra_freqs, 
                                    cmb_map_freq=fgi.spectra_freqs[0], 
                                    plot_patch=True, patch_num=None, 
                                    noise=False, show=True, save=True, 
                                    fname=None, clims=50, cmb_clim=300, dpi=500):
        if plot_patch and (patch_num is None):
            patch_num = self.get_patch_num(ra=self.ra_ctr, dec=self.dec_ctr)
        freqs = self._validate_freqs(freqs)
        # frequency-dependent components in the maps:
        plt_components = []
        plt_component_names = []
        if self.subtract_clusters:
            plt_components.append('tsz')
            plt_component_names.append('tSZ')
        if self.subtract_sources:
            if 'cib' in self.map_components:
                plt_components.append('cib')
            if 'radio' in self.map_components:
                plt_components.append('radio')
        # frequency-independent components in maps:
        other_component_names = []
        if simutils.has_cmb(self.map_components):
            other_component_names.append('CMB')
        if 'ksz' in self.map_components:
            other_component_names.append('kSZ')
        other_components_label = ' & '.join(other_component_names)
        
        if save:
            if fname is None:
                plt_freq_info = '_'.join([f'{round(freq):03d}' for freq in freqs])
                plt_map_info = 'maps' if (patch_num is None) else f'patch{patch_num:02d}_maps'
                fname = os.path.join(self.plots_dir(), f'{plt_freq_info}_{plt_map_info}.pdf')
        else:
            fname = None
        
        map_kwargs = {'beam': True, 'noise': noise}
        if patch_num is not None:
            patch_shape, patch_wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=False)
            map_kwargs['shape'] = patch_shape
            map_kwargs['wcs'] = patch_wcs
        
        self.infomsg(f"plotting maps before and after FG cleaning for {freqs = }")
        # maps before & after FG cleaning w/ all components:
        map_before_with_cmb = self.get_sim(freq=cmb_map_freq, **map_kwargs)
        map_after_with_cmb = self.get_sim(freq=cmb_map_freq, subtract_sources=True, 
                                          subtract_clusters=True, **map_kwargs)
        # maps before & after FG cleaning w/ only freq-dependent components:
        maps_before = {}
        maps_after = {}
        for freq in freqs:
            maps_before[freq] = self.get_sim(freq=freq, components=plt_components, **map_kwargs)
            maps_after[freq] = self.get_sim(freq=freq, components=plt_components, 
                                            subtract_sources=True, subtract_clusters=True, **map_kwargs)
        plt_output = fgplots.plot_fgcleaned_maps_with_cmb(map_before_with_cmb, map_after_with_cmb, 
                                                          maps_before, maps_after, cmb_map_freq, 
                                                          maps_without_cmb_label=other_components_label, 
                                                          clims=clims, cmb_clim=cmb_clim, dpi=dpi, 
                                                          show=show, fname=fname)
        if save:
            self.infomsg(f"saved {fname}")
        if not show:
            return plt_output
    
    
    def plot_fgcleaned_spectra(self, freqs=fgi.spectra_freqs, patch_num=None, 
                               subtract_sources=True, subtract_clusters=True, 
                               mask=True, spec_type='dl', plt_lmax=20000, 
                               show=True, save=True, fname=None, **kwargs):
        '''kwargs are for the mask'''
        freqs = self._validate_freqs(freqs)
        _, fg_components, _ = simutils.separate_component_list(self.map_components)
        subtract_sources = subtract_sources and self.subtract_sources
        subtract_clusters = subtract_clusters and self.subtract_clusters
        if spec_type.lower() == 'dl':
            bin_dl = True
            bin_cl = False
        else:
            bin_dl = False
            bin_cl = True
        common_kwargs = {'beam': True, 'bin_cl': bin_cl, 'bin_dl': bin_dl, 
                         'subtract_sources': subtract_sources, 'subtract_clusters': subtract_clusters, 
                         'mask': mask, **self._get_mask_kwargs(**kwargs), 
                         **self._patch_kwargs_for_spectra(patch_num=patch_num)}
        spectra_kwargs = {'total_fg_noise': {**common_kwargs, 'components': fg_components, 'noise': True}}
        if subtract_sources:
            spectra_kwargs['cib_radio'] = {**common_kwargs, 'components': ['cib', 'radio']}
        if subtract_clusters:
            spectra_kwargs['tsz'] = {**common_kwargs, 'components': ['tsz']}
        
        sim_spectra = {}
        for freq in freqs:
            sim_spectra[freq] = {}
            for key in spectra_kwargs:
                sim_spectra[freq][key] = self.get_sim_power(freq=freq, save=True, **spectra_kwargs[key])
        prev_spectra = self.get_binned_previous_predicted_fgcleaned_spectra()
        
        if save:
            if fname is None:
                plt_map_info = self._map_component_list2str(components=fg_components)
                if patch_num is not None:
                    plt_map_info = f'{plt_map_info}_patch{patch_num:02d}'
                plt_freq_info = '_'.join([f'{round(freq):03d}' for freq in freqs])
                fname = os.path.join(self.plots_dir(), f'{plt_freq_info}_{plt_map_info}_fgcleaned_spectra.pdf')
        else:
            fname = None
        sim_fg_spectra_keys = [key for key in spectra_kwargs if ('total' not in key)]
        plt_lmax = min([self.lmax, plt_lmax])
        plt_output = fgplots.plot_fgcleaned_spectra(freqs, sim_spectra, prev_spectra, spec_type=spec_type.lower(), 
                                            plt_lmax=plt_lmax, sim_fg_components=fg_components, 
                                            sim_spectra_keys=sim_fg_spectra_keys, show=show, fname=fname)
        if save:
            self.infomsg(f"saved {fname}")
        if not show:
            return plt_output
        
        
    def plot_coadded_fgcleaned_spectra(self, patch_num=None, subtract_sources=True, subtract_clusters=True, 
                                       mask=True, spec_type='dl', plt_lmax=20000, label='Sim-based forecast', 
                                       dpi=500, show=True, save=True, fname=None, **kwargs):
        '''kwargs are for the mask'''
        # NOTE : can only compare to prev. HD prediction for 90+148 GHz - so `freqs` isn't an option
        freqs = fgi.spectra_freqs
        skey = f'{spec_type.lower()}tt'
        coadd_lmin = 0
        coadd_lmax = hd_data.HDMockData().lmax
        
        # previous HD prediction
        prev_spectra = self.get_binned_previous_predicted_fgcleaned_spectra()
        prev_lbin = prev_spectra[90]['total_fg_noise']['ells'].copy()
        if not all([c in self.map_components for c in ['ksz', 'tsz', 'cib', 'radio']]):
            _, fg_components, _ = simutils.separate_component_list(self.map_components)
            prev_fg_noise_spectra = {}
            for freq in freqs:
                prev_fg_noise_spectra[freq] = prev_spectra[freq]['noise'][skey].copy()
                for fg in fg_components:
                    prev_fg_noise_spectra[freq] += prev_spectra[freq][fg][skey].copy()
        else:
            prev_fg_noise_spectra = {freq: prev_spectra[freq]['total_fg_noise'][skey] for freq in freqs}
        prev_coadded_lbin, prev_coadded_spectra = fgresults.calc_coadded_noise(prev_lbin, prev_fg_noise_spectra, 
                                                                               lmin=coadd_lmin, lmax=coadd_lmax,
                                                                               interp=False)
        prev_loc = np.where(prev_coadded_lbin <= plt_lmax)
        
        # sim-based
        coadded_sim_spectra = self.get_coadded_fgcleaned_sim_power(freqs=freqs, patch_num=patch_num, 
                                                                   subtract_sources=subtract_sources, 
                                                                   subtract_clusters=subtract_clusters, 
                                                                   mask=mask, dl=(spec_type=='dl'), 
                                                                   cl=(spec_type=='cl'), **kwargs)
        lbin = coadded_sim_spectra['ells']
        loc = np.where(lbin <= plt_lmax)
        
        fig, axs, _, _ = fgplots._setup_multifreq_spectra_plot(spec_type=spec_type, lmax=plt_lmax, 
                                                               freq_label=False, plot_cmb=True, 
                                                               cmb_spectra=prev_spectra[90]['cmb'], cmb_label=None)
        legend_kwargs = {'fontsize': 10.5, 'bbox_to_anchor': (1, 0.8), 'loc': 'upper right', 
                         'markerfirst': False, 'alignment': 'right', 'frameon': False, 'handlelength': 1.5}
        ylims = [4e-2, 8e3]
        ax = axs[0]
        ax.plot(lbin[loc], coadded_sim_spectra[skey][loc], color='#b50202', label=label, zorder=3)
        ax.plot(prev_lbin[prev_loc], prev_coadded_spectra[prev_loc], color='#ff7070', label='Previous forecast')
        ax.text(0.95, 0.925, f'Coadded 90 & 148 GHz\nResidual FGs + Inst. Noise',
                va='top', ha='right', transform=ax.transAxes, color='k', fontsize=12, 
                bbox={'facecolor': 'w', 'edgecolor': 'k', 'boxstyle': 'round', 'pad': 0.25})
        ax.legend(**legend_kwargs)
        ax.grid(alpha=0.1)
        ax.set_ylim(ylims)
        
        if save:
            if fname is None:
                freq_info = '_'.join([f'{round(freq):03d}' for freq in freqs])
                coadd_info = f'coadded_{freq_info}'
                if patch_num is not None:
                    coadd_info = f'{coadd_info}_patch{patch_num:02d}'
                fname = os.path.join(self.plots_dir(), f'{coadd_info}_fgcleaned_noise_spectra.pdf')
            plt.savefig(fname, bbox_inches='tight', dpi=dpi)
        if show:
            plt.show()
        else:
            return fig, ax, legend_kwargs, ylims

    
    def spectral_index_plot(self, freq, component, patch_num=None, 
                            xticks=None, yticks=None, labelsize=11, ticksize=10, legendsize=9,
                            cmap='rainbow', fig=None, ax=None, cax=None, dpi=500,
                            show=True, save=True, fname=None,):
        component_name = 'CIB' if (component == 'cib') else 'radio'
        if patch_num is None:
            patch_num = self.get_patch_num(ra=self.ra_ctr, dec=self.dec_ctr)
        freq1 = 277 if (component == 'cib') else 90
        freq2 = freq
        index_cat = self.get_spectral_index_catalog(freq2, freq1, component=component, patch_num=patch_num)
        fluxs1 = index_cat[f'fluxmJy_{freq1}GHz'].values
        fluxs2 = index_cat[f'fluxmJy_{freq2}GHz'].values
        true_index, true_index_std = self.get_true_spectral_index(freq1, freq2, component, patch_num=patch_num)
        measured_index, measured_index_std = self.get_spectral_index(freq2, component, patch_num=patch_num)
        self.infomsg(f'true     average {freq1}-to-{freq2} {component_name} index = {true_index:.2f}')
        self.infomsg(f'measured average {freq1}-to-{freq2} {component_name} index = {measured_index:.2f}')
        if save:
            if fname is None:
                index_info = f'{round(freq1)}to{round(freq2)}GHz_{component}'
                fname = os.path.join(self.plots_dir(), f'{index_info}_spectral_index_patch{patch_num:02d}.pdf')
        else:
            fname = None
        plt_output = fgplots.spectral_index_plot(fluxs1, fluxs2, freq1, freq2, fig=fig, ax=ax, cax=cax,
                                         component_name=component_name, true_index=true_index, cmap=cmap,
                                         labelsize=labelsize, ticksize=ticksize, legendsize=legendsize,
                                         xticks=xticks, yticks=yticks, dpi=dpi, show=show, fname=fname)
        if not show:
            return plt_output
    
    
    def measured_vs_true_flux_plot(self, freq, plot_patch=True, patch_num=None, match_radius=None, 
                                   max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, min_match_snr=None, 
                                   labelsize=11, ticksize=10, legendsize=9, textsize=12,
                                   cmap='rainbow', plot_width=4, dpi=500,
                                   fig=None, ax=None, cax=None, show=True, save=True, fname=None):
        if plot_patch and (patch_num is None):
            patch_num = self.get_patch_num(ra=self.ra_ctr, dec=self.dec_ctr)

        matched_catalog, _, _ = self.match_to_true_sources_catalog(freq, patch_num=patch_num, 
                                                                   match_radius=match_radius, min_snr=min_match_snr,
                                                                   max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
        if 'method' in matched_catalog.columns.values:
            matched_catalog = matched_catalog[~matched_catalog['method'].eq('extrap')]

        min_measured_flux = np.min(matched_catalog['fluxmJy'].values)
        min_true_flux = np.min(matched_catalog['true_fluxmJy'].values)
        if min_true_flux < min_measured_flux/2: 
            # set the min. flux for the axis limits to be half the min. measured flux:
            min_flux = round(min_measured_flux/2, 2)
        else:
            min_flux = None

        if save:
            if fname is None:
                fname_info = f'{int(freq):03d}GHz_measured_vs_true_flux'
                if min_match_snr is None:
                    min_plt_snr = np.min(self.sources_snr_threshold_list)
                else:
                    min_plt_snr = min_match_snr
                fname_info = f'{fname_info}_SN{simutils.round_str(min_plt_snr)}'
                if patch_num is not None:
                    fname_info = f'{fname_info}_patch{patch_num:02d}'
                fname = os.path.join(self.plots_dir(), f'{fname_info}.pdf')
        else:
            fname = None
        plot_output = fgplots.measured_vs_true_flux_plot(matched_catalog, freq=freq,
                                                         true_flux_col='true_fluxmJy', measured_flux_col='fluxmJy', 
                                                         cbar_col='SNR', labelsize=labelsize, ticksize=ticksize, 
                                                         legendsize=legendsize, textsize=textsize,
                                                         min_flux=min_flux, cmap=cmap, plot_width=plot_width, dpi=dpi,
                                                         fig=fig, ax=ax, cax=cax, show=show, fname=fname)
        if not show:
            return plot_output
     
        
    def flux_measurement_plot(self, patch_num=None, 
                              freq_for_index_plot=148, component_for_index_plot='cib',
                              freq_for_flux_plot=90, match_radius=None,  min_match_snr=None,
                              max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err,
                              xticks=None, yticks=None, # only for spectral index
                              labelsize=11, ticksize=10, legendsize=9, textsize=12, cmap='rainbow',
                              plot_width=4, dpi=500, show=True, save=True, fname=None):
        '''
        combines plots for spectral index and for measured vs true flux
        '''
        if patch_num is None:
            patch_num = self.get_patch_num(ra=self.ra_ctr, dec=self.dec_ctr)
        plt_kwargs = {'patch_num': patch_num, 'labelsize': labelsize, 'ticksize': ticksize, 
                      'legendsize': legendsize, 'cmap': cmap, 'save': False, 'show': False}
        subfig_wspace = 0.1
        cax_frac = 0.05
        fig = plt.figure(figsize=(2 * plot_width * (1+cax_frac) * (1+subfig_wspace/2), plot_width), dpi=dpi)
        subfigs = fig.subfigures(ncols=2, wspace=subfig_wspace)
        _, ax1, cax1, cbar1, pts1 = self.measured_vs_true_flux_plot(freq_for_flux_plot, fig=subfigs[0],
                                                                    match_radius=match_radius, 
                                                                    min_match_snr=min_match_snr,
                                                                    max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err,
                                                                    textsize=textsize, **plt_kwargs)
        _, ax2, cax2, cbar2, pts2 = self.spectral_index_plot(freq_for_index_plot, component_for_index_plot, 
                                                             fig=subfigs[1], xticks=xticks, yticks=yticks, 
                                                             **plt_kwargs)
        if save:
            if fname is None:
                freq1 = 277 if (component_for_index_plot == 'cib') else 90
                freq2 = freq_for_index_plot
                index_info = f'{round(freq1)}to{round(freq2)}{component_for_index_plot}_index'
                flux_info = f'{round(freq_for_flux_plot)}GHz_flux'
                patch_info = f'patch{patch_num:02d}'
                fname_root = '_'.join([flux_info, index_info, patch_info])
                fname = os.path.join(self.plots_dir(), f'{fname_root}.pdf')
            plt.savefig(fname, bbox_inches='tight', dpi=dpi)
        if show:
            plt.show()

    
    def estimate_min_flux_for_completeness(self, freq, completeness=0.95, patch_num=None, 
                                           num_flux_bins=100, min_flux=None, match_radius=None, 
                                           max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, min_snr=None):
        '''
        if min_flux is not None: only consider fluxes above this
        '''
        # catalog of all true sources in map
        true_srcs = self.get_true_sources_catalog(patch_num=patch_num)
        # remove any true sources that were masked before FG cleaning
        imap_mask = self._get_imap_masks(patch_num=patch_num, freqs=[freq], apodize=False)[freq]
        true_srcs = fgmasks.add_mask_info_to_catalog(true_srcs, imap_mask)
        true_srcs = true_srcs[~true_srcs['masked']].copy()
        # catalog of detected sources that were matched to a true source
        matched_srcs, _, _ = self.match_to_true_sources_catalog(freq, patch_num=patch_num, 
                                                                match_radius=match_radius, min_snr=min_snr, 
                                                                max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
        # calculate completeness
        all_true_fluxes = true_srcs[f'fluxmJy_{freq}GHz'].values
        matched_true_fluxes = matched_srcs['true_fluxmJy'].values
        (src_completeness, flux_bin_ctrs, 
         flux_bin_edges, _, _) = fgresults.get_source_completeness(matched_true_fluxes, all_true_fluxes, 
                                                                   num_flux_bins=num_flux_bins, cumulative=True)
        if min_flux is not None:
            lower_bin_edges = flux_bin_edges[:-1]
            src_completeness = src_completeness[lower_bin_edges >= min_flux]
            flux_bin_ctrs = flux_bin_ctrs[lower_bin_edges >= min_flux]

        if not np.any(src_completeness >= completeness):
            self.infomsg((f"{freq} GHz : unable to estimate approx. flux for "
                          f"{simutils.round_str(100*completeness)}% completion; "
                          "this level of completeness was not reached"))
        else:
            flux = np.min(flux_bin_ctrs[src_completeness >= completeness])
            # round the estimate
            if flux < 0.1:
                flux = float(f'{flux:.1g}')
            elif flux < 1:
                flux = round(flux, 2)
            elif flux < 100:
                flux = round(flux, 1)
            else:
                flux = round(flux)
            self.infomsg((f"{freq} GHz : approx. flux for {simutils.round_str(100*completeness)}% "
                          f"completion = {flux} mJy"))
            return flux


    def true_flux_histogram_plots(self, freqs=fgi.spectra_freqs, patch_num=None, plot_patch=True,
                                  source_match_radius_dict=None,  min_snr=None, 
                                  max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err,
                                  plot_flux_at_completeness=True, completeness=0.95,
                                  num_flux_bins=250, fig=None, axs=None, 
                                  plot_width=4.5, plot_height=4.25, dpi=500, equal_ylims=True,
                                  labelsize=11, ticksize=10, legendsize=8.75, textsize=12,
                                  show=True, save=True, fname=None):
        if plot_patch and (patch_num is None):
            patch_num = self.get_patch_num(ra=self.ra_ctr, dec=self.dec_ctr)
        if source_match_radius_dict is None:
            source_match_radius_dict = {freq: None for freq in freqs}
        matched_bright_true_fluxes = {}
        matched_extrap_true_fluxes = {}
        unmatched_true_fluxes = {}
        unmatched_measured_fluxes = {}
        for freq in freqs:
            self.infomsg(f"getting {freq} GHz catalogs")
            kwargs = {'patch_num': patch_num, 'match_radius': source_match_radius_dict[freq],
                      'min_snr': min_snr, 'max_abs_flux_diff_vs_err': max_abs_flux_diff_vs_err}
            matched, unmatched, unmatched_true = self.match_to_true_sources_catalog(freq, **kwargs)
            matched_bright_true_fluxes[freq] = matched[~matched['method'].eq('extrap')]['true_fluxmJy'].values
            matched_extrap_true_fluxes[freq] =  matched[matched['method'].eq('extrap')]['true_fluxmJy'].values
            unmatched_true_fluxes[freq] = unmatched_true['fluxmJy'].values
            unmatched_measured_fluxes[freq] = unmatched['fluxmJy'].values
        if plot_flux_at_completeness:
            detected_flux_label = f"Flux for %s completion" % f'{simutils.round_str(100*completeness)}%'
            detected_flux_lims = {}
            for freq in freqs:
                kwargs = {'completeness': completeness, 'match_radius': source_match_radius_dict[freq],
                          'min_snr': min_snr, 'max_abs_flux_diff_vs_err': max_abs_flux_diff_vs_err}
                detected_flux_lims[freq] = self.estimate_min_flux_for_completeness(freq, **kwargs)
        else:
            detected_flux_lims = None
            detected_flux_label = None
        min_detected_snr = np.min(self.sources_snr_threshold_list) if (min_snr is None) else min_snr
        
        if save:
            if fname is None:
                freqs_info = '_'.join([f'{round(freq):03d}' for freq in freqs])
                fname = os.path.join(self.plots_dir(), f'{freqs_info}_sources_histograms.pdf')
        else:
            fname = None

        plt_output = fgplots.true_flux_histogram_plots(freqs, matched_bright_true_fluxes, 
                                                       matched_extrap_true_fluxes=matched_extrap_true_fluxes, 
                                                       unmatched_true_fluxes=unmatched_true_fluxes, 
                                                       unmatched_measured_fluxes=unmatched_measured_fluxes,
                                                       min_detected_snr=min_detected_snr, 
                                                       detected_flux_lims=detected_flux_lims, 
                                                       detected_flux_label=detected_flux_label,
                                                       num_flux_bins=num_flux_bins, equal_ylims=equal_ylims,
                                                       fig=fig, axs=axs, plot_width=plot_width, 
                                                       plot_height=plot_height, dpi=dpi,
                                                       labelsize=labelsize, ticksize=ticksize, 
                                                       legendsize=legendsize, textsize=textsize,
                                                       show=show, fname=fname)

        if not show:
            return plt_output


    def sources_plot(self, freqs=fgi.spectra_freqs, patch_num=None, plot_patch=True,
                     plt_lmax=20000, spec_type='dl', spectra_ylims=[0.05, 6600],
                     source_match_radius_dict=None,  min_snr=None, 
                     max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err,
                     plot_flux_at_completeness=True, completeness=0.95, num_flux_bins=250, 
                     equal_ylims=True, plot_width=4.5, plot_height=4.25, dpi=500,
                     labelsize=11, ticksize=10, legendsize=8.75, textsize=12,
                     show=True, save=True, fname=None):
        '''histograms + power spectra of sources'''
        if plot_patch and (patch_num is None):
            patch_num = self.get_patch_num(ra=self.ra_ctr, dec=self.dec_ctr)
        # needed for spectra:
        spec_key = f'{spec_type}tt'
        theo_labels = {'cmb': 'Lensed CMB', 'noise': 'Instrumental noise'}
        theo_colors = {'cmb': 'k', 'noise': 'tab:gray'}
        theo_lines = {'cmb': '-', 'noise': '--'}
        theo_spectra = self.get_binned_previous_predicted_fgcleaned_spectra()

        bin_dl = ('dl' in spec_type)
        bin_cl = not bin_dl
        patch_kwargs = {'beam': True, 'components': ['cib', 'radio'], 'bin_dl': bin_dl, 'bin_cl': bin_cl, 
                        **self._patch_kwargs_for_spectra(patch_num=patch_num)}

        # make lists of data to plot and kwargs for spectra at each freq:
        plt_spectra = {}
        spectra_index_for_ylims = None
        for i, freq in enumerate(freqs):
            plt_spectra[freq] = []
            # theory curves
            for key in ['cmb', 'noise']:
                if key == 'cmb':
                    spectra_index_for_ylims = len(plt_spectra[freq])
                plt_spectra[freq].append((theo_spectra[freq][key]['ells'], theo_spectra[freq][key][spec_key], 
                                          {'lw': 1.25, 'ls': theo_lines[key], 'color': theo_colors[key], 
                                           'label': theo_labels[key], 'zorder': 2}))
            # power of CIB+radio after only subtracting bright sources
            sub_bright_srcs_spectra = self._get_bright_src_sub_spectra(freq, patch_num=patch_num, 
                                                                       bin_dl=bin_dl, bin_cl=bin_cl)
            plt_spectra[freq].append((sub_bright_srcs_spectra['ells'], sub_bright_srcs_spectra[spec_key],
                                      {'label': "Bright CIB + Radio removed", 'color': 'tab:olive', 'lw': 1.5}))
            # power of CIB+radio after subtracting all sources
            sub_srcs_spectra = self.get_sim_power(freq=freq, subtract_sources=True, mask=True, **patch_kwargs)
            plt_spectra[freq].append((sub_srcs_spectra['ells'], sub_srcs_spectra[spec_key],
                                      {'label': "Bright & dim CIB + Radio removed", 'color': 'tab:pink', 'lw': 2}))
            # power of CIB+radio before any FG cleaning
            srcs_spectra = self.get_sim_power(freq=freq, **patch_kwargs)
            plt_spectra[freq].append((srcs_spectra['ells'], srcs_spectra[spec_key],
                                      {'color': 'tab:blue', 'lw': 1.5, 'ls': ':', 'alpha': 0.5})) # add label later

        # make the plots:
        ncol = len(freqs) 
        nrow = 2    
        fig, axs = plt.subplots(figsize=(plot_width * ncol, plot_height * nrow), ncols=ncol, nrows=nrow, dpi=dpi)
        # histograms
        hist_axs = [axs[0,i] for i in range(ncol)]
        hist_kwargs = {'freqs': freqs, 'patch_num': patch_num, 'plot_patch': True,
                       'source_match_radius_dict': source_match_radius_dict,
                       'max_abs_flux_diff_vs_err': max_abs_flux_diff_vs_err, 'min_snr': min_snr,
                       'num_flux_bins': num_flux_bins, 
                       'plot_flux_at_completeness': plot_flux_at_completeness, 'completeness': completeness,
                       'labelsize': labelsize, 'ticksize': ticksize, 'legendsize': legendsize, 'textsize': textsize,
                       'equal_ylims': equal_ylims, 'fig': fig, 'axs': hist_axs, 'show': False, 'save': False}
        fig, hist_axs, hist_legend_kwargs = self.true_flux_histogram_plots(**hist_kwargs) 
        # spectra plot
        spec_axs = [axs[1,i] for i in range(ncol)]
        spec_kwargs = {'fig': fig, 'axs': spec_axs, 'show': False, 'spec_type': spec_type, 'lmax': plt_lmax,
                       'spectra_index_for_ylims': spectra_index_for_ylims, 'logy': True, 
                       'freq_label': True, 'freq_label_pos': (0.87, 0.9), 
                       'freq_label_va': 'bottom', 'freq_label_ha': 'right', 
                       'labelsize': labelsize, 'ticksize': ticksize, 'legendsize': legendsize, 'textsize': textsize}
        fig, spec_axs, spec_legend_kwargs =  fgplots._plot_multifreq_spectra(freqs, plt_spectra, **spec_kwargs)
        spec_legend_kwargs['bbox_to_anchor'] = (1, 0.9)
        for ax in spec_axs:
            # make line darker for legend label:
            ax.plot([], [], label="CIB + Radio before removal", color='tab:blue', alpha=0.75, ls=':', lw=1.5)
            ax.legend(**spec_legend_kwargs)
            ax.set_ylim(spectra_ylims)
        plt.subplots_adjust(wspace=0.25)
        if save:
            if fname is None:
                freqs_info = '_'.join([f'{round(freq):03d}' for freq in freqs])
                fname = os.path.join(self.plots_dir(), f'{freqs_info}_sources_histograms_and_spectra.pdf')
            plt.savefig(fname, bbox_inches='tight', dpi=dpi)
        if show:
            plt.show()
     
    
    def cluster_mass_vs_redshift_plot(self, match_radius=fgi.cluster_match_radius, patch_num=None, min_snr=None,
                                      mass_axis_lims=[1e12, 2e15], redshift_axis_lims=None,
                                      labelsize=13, ticksize=12, dpi=500, figsize=(5,5),
                                      show=True, save=True, fname=None):
        (_, _, matched_true_clusters, 
               unmatched_true_clusters) = self.match_to_true_clusters_catalog(match_radius=match_radius, 
                                                                              patch_num=patch_num, 
                                                                              min_snr=min_snr)
        if save:
            if fname is None:
                fname_root = 'clusters_mass_vs_redshift'
                if (min_snr is not None) and (min_snr > np.min(self.clusters_snr_threshold_list)):
                    fname_root = f'{fname_root}_SN{simutils.round_str(min_snr)}'
                if patch_num is not None:
                    fname_root = f'{fname_root}_patch{patch_num:02d}'
                fname = os.path.join(self.plots_dir(), f'{fname_root}.pdf')
        plt_output = fgplots.cluster_mass_vs_redshift_plot(matched_true_clusters, unmatched_true_clusters, 
                                                   plot_mass_at_completeness=True, completeness=0.99,
                                                   zkey='z', mass_key='M500', 
                                                   mass_label=r'True $M_{500\mathrm{c}}~[M_{\odot}]$', 
                                                   redshift_label=r'True $z$',
                                                   mass_axis_lims=mass_axis_lims, 
                                                   redshift_axis_lims=redshift_axis_lims,
                                                   labelsize=labelsize, ticksize=ticksize, dpi=dpi, 
                                                   figsize=figsize, show=show, fname=fname)
        if not show:
            return plt_output
        
        
    def cluster_completeness_per_mass_redshift_plot(self, match_radius=fgi.cluster_match_radius, min_snr=None,
                                                    num_mass_bins=15, min_M500=1e13, max_M500=1.5e14, 
                                                    figsize=(4,4), dpi=500, labelsize=12, legendsize=11, 
                                                    colors=None, show=True, save=True, fname=None):
        (_, _, matched_true_clusters, 
               unmatched_true_clusters) = self.match_to_true_clusters_catalog(match_radius=match_radius, 
                                                                              min_snr=min_snr)
        true_clusters = fgcatalogs.combine_catalogs([matched_true_clusters.copy(), unmatched_true_clusters.copy()])
        # remove true clusters located in a 1 pixel border at edge of map
        map_rmin, map_rmax, map_dmin, map_dmax = maps.get_map_corner_coords(self.shape, self.wcs)
        pix_size = utils.arcmin2deg(self.res)
        corners = {'ra_min': map_rmin + pix_size, 'ra_max': map_rmax - pix_size, 
                   'dec_min': map_dmin + pix_size, 'dec_max': map_dmax - pix_size}
        true_clusters = fgcatalogs.trim_catalog_positions(true_clusters, **corners)
        matched_true_clusters = fgcatalogs.trim_catalog_positions(matched_true_clusters, **corners)

        if save and (fname is None):
            fname_root = 'cluster_completeness_per_mass_redshift'
            if min_snr is not None:
                fname_root = f'{fname_root}_SN{simutils.round_str(min_snr)}'
            fname = os.path.join(self.plots_dir(), f'{fname_root}.pdf')
        elif not save:
            fname = None

        if min_snr is None:
            min_snr = np.min(self.clusters_snr_threshold_list)
        ylabel = r'Completeness for SNR $\geq$ %s' % simutils.round_str(min_snr)
        plt_output = fgplots.cluster_completeness_per_mass_redshift_plot(true_clusters, matched_true_clusters, 
                                                                         num_mass_bins=num_mass_bins, 
                                                                         min_M500=min_M500, max_M500=max_M500, 
                                                                         ylabel=ylabel, show=show, fname=fname)
        if not show:
            return plt_output

