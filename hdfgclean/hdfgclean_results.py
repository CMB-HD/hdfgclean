import os
import numpy as np
from hdsims import siminfo as si, utils, simutils, fgcatalogs
from . import fgclean_info as fgi, fgutils, fgmasks, fgresults, hdfgclean_spectra


class HDFGCleanResults(hdfgclean_spectra.HDFGCleanSpectra):        
        
    def get_true_spectral_index(self, freq1, freq2, component, patch_num=None):
        all_true_sources_catalog = self.get_true_sources_catalog(save=False, patch_num=patch_num)
        true_sources_catalog = all_true_sources_catalog[all_true_sources_catalog['component'].eq(component)]
        index_mean, index_std = fgutils.get_avg_spectral_index(true_sources_catalog[f'fluxmJy_{freq1}GHz'].values, 
                                                        true_sources_catalog[f'fluxmJy_{freq2}GHz'].values, 
                                                        freq1, freq2)
        return index_mean, index_std
        
    
    def matched_sources_catalogs_dir(self, patch_num=None, make_dir=True):
        '''
        patch_num=None is for full map
        '''
        base_dir = self.results_dir() if (patch_num is None) else self.patch_sources_dir(patch_num=patch_num)#self.patch_fgclean_dir(patch_num=patch_num)
        odir = os.path.join(base_dir, 'match_to_true_catalogs')
        if make_dir:
            utils.mkdir(odir)
        return odir
    
    
    def matched_clusters_catalogs_dir(self, patch_num=None, make_dir=True):
        '''
        patch_num=None is for full map
        '''
        base_dir = self.results_dir() if (patch_num is None) else self.patch_clusters_dir(patch_num=patch_num)#self.patch_fgclean_dir(patch_num=patch_num)
        odir = os.path.join(base_dir, 'match_to_true_catalogs')
        if make_dir:
            utils.mkdir(odir)
        return odir

    
    
    def _get_map_area_info(self, patch_num=None, padded=False):
        if patch_num is None:
            width = self.padded_width if padded else self.width
            height = self.padded_height if padded else self.height
        else:
            width, height = self.patches.get_patch_dimensions(patch_num=patch_num, padded=padded)
        area_info = f'{simutils.round_str(width)}x{simutils.round_str(height)}deg'
        return area_info
        
    
    
    def true_sources_catalog_fname(self, patch_num=None, padded=False):
        map_area_info = self._get_map_area_info(patch_num=patch_num, padded=padded)
        ptsrc_components = [c for c in ['cib', 'radio'] if (c in self.map_components)]
        src_info = self._map_component_list2str(components=ptsrc_components)
        fname_root = '_'.join(['true_sources', src_info, map_area_info])
        fname = f'{fname_root}.csv'
        fname = os.path.join(self.matched_sources_catalogs_dir(patch_num=patch_num), fname)
        return fname


    def _sources_match_info(self, freq, match_radius=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, min_snr=None, patch_num=None):
        freq = simutils.validate_sim_freq(freq, valid_freqs=self.freqs)
        map_area_info = self._get_map_area_info(patch_num=patch_num)
        match_radius = utils.fwhm2sigma(si.beam_fwhm[freq]) if (match_radius is None) else match_radius
        match_dist_info = f'radius{simutils.round_str(match_radius, n=3)}arcmin'
        match_flux_info = f'maxfluxdiff{max_abs_flux_diff_vs_err}sigma'
        match_info = '_'.join([match_dist_info, match_flux_info, map_area_info])
        if min_snr is not None:
            match_info = f'minSNR{simutils.round_str(min_snr)}'
        return match_info


    def matched_subtracted_sources_catalog_fname(self, freq, match_radius=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, min_snr=None, patch_num=None):
        match_info = self._sources_match_info(freq, patch_num=patch_num, match_radius=match_radius, min_snr=min_snr,
                                              max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
        fname = os.path.join(self.matched_sources_catalogs_dir(patch_num=patch_num), 
                             f'{round(freq):03d}GHz_matched_sources_{match_info}.csv')
        return fname


    def unmatched_subtracted_sources_catalog_fname(self, freq, match_radius=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, min_snr=None, patch_num=None):
        match_info = self._sources_match_info(freq, patch_num=patch_num, match_radius=match_radius,  min_snr=min_snr,
                                              max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
        fname = os.path.join(self.matched_sources_catalogs_dir(patch_num=patch_num), 
                             f'{round(freq):03d}GHz_unmatched_subtracted_sources_{match_info}.csv')
        return fname


    def unmatched_true_sources_catalog_fname(self, freq, match_radius=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, min_snr=None, patch_num=None):
        match_info = self._sources_match_info(freq, patch_num=patch_num, match_radius=match_radius,  min_snr=min_snr,
                                              max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
        fname = os.path.join(self.matched_sources_catalogs_dir(patch_num=patch_num), 
                             f'{round(freq):03d}GHz_unmatched_true_sources_{match_info}.csv')
        return fname

    
    def get_true_sources_catalog(self, save=False, patch_num=None):
        fname = self.true_sources_catalog_fname(patch_num=patch_num)
        if os.path.exists(fname):
            catalog = fgcatalogs.load_catalog(fname)
        else:
            # get the catalog of all sources in the full maps that were FG cleaned (including apodized region):
            full_catalog_fname = self.true_sources_catalog_fname(patch_num=None, padded=True)
            if os.path.exists(full_catalog_fname):
                catalog = fgcatalogs.load_catalog(full_catalog_fname)
            else:
                ptsrc_components = [c for c in ['cib', 'radio'] if (c in self.map_components)]
                catalogs = {}
                for component in ptsrc_components:
                    catalogs[component] = self.get_catalog(component, include_padded_area=True)
                    catalogs[component]['component'] = component
                catalog = fgcatalogs.combine_catalogs([catalogs[c] for c in ptsrc_components])
                catalog['idx'] = list(range(len(catalog)))
                # only keep columns that are in both CIB and radio catalogs:
                flux_cols = [f'fluxmJy_{freq}GHz' for freq in si.freqs]
                cols = ['RADeg', 'decDeg', *flux_cols, 'component', 'idx']
                catalog = catalog[cols].copy()
                if save:
                    catalog.to_csv(full_catalog_fname)
                    self.infomsg(f"saved {full_catalog_fname}")
            # trim this catalog to only include sources in un-apodized region (for full map or patch)
            catalog = self.remove_objects_in_apodized_region(catalog, patch_num=patch_num)
            if save:
                catalog.to_csv(fname)
                self.infomsg(f"saved {fname}")
        return catalog



    def match_to_true_sources_catalog(self, freq, match_radius=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, min_snr=None, patch_num=None):
        matched_catalog_fname = self.matched_subtracted_sources_catalog_fname(freq, patch_num=patch_num, match_radius=match_radius, min_snr=min_snr, max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
        unmatched_catalog_fname = self.unmatched_subtracted_sources_catalog_fname(freq, patch_num=patch_num, match_radius=match_radius, min_snr=min_snr, max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
        unmatched_true_catalog_fname = self.unmatched_true_sources_catalog_fname(freq, patch_num=patch_num, match_radius=match_radius, min_snr=min_snr, max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
        catalogs_saved = all([os.path.exists(fname) for fname in [matched_catalog_fname, unmatched_catalog_fname, unmatched_true_catalog_fname]])
        if catalogs_saved:
            matched_catalog = fgcatalogs.load_catalog(matched_catalog_fname)
            unmatched_catalog = fgcatalogs.load_catalog(unmatched_catalog_fname)
            unmatched_true_catalog = fgcatalogs.load_catalog(unmatched_true_catalog_fname)
        else:
            patch_info = '' if (patch_num is None) else f'[{patch_num = }]'
            self.infomsg(f"{patch_info} matching catalog of subtracted {simutils.round_str(freq)} GHz sources to the true catalog")
            # get catalog of true sources:
            true_srcs = self.get_true_sources_catalog(patch_num=patch_num)
            true_srcs['fluxmJy'] = true_srcs[f'fluxmJy_{round(freq)}GHz'].values.copy()
            true_srcs = true_srcs[['RADeg', 'decDeg', 'fluxmJy', 'component', 'idx']].copy()
            # remove any true sources that were masked before FG cleaning
            imap_mask = self._get_imap_masks(patch_num=patch_num, freqs=[freq], apodize=False)[freq]
            true_srcs = fgmasks.add_mask_info_to_catalog(true_srcs, imap_mask)
            true_srcs = true_srcs[~true_srcs['masked']].copy()
            
            # get catalog of subtracted sources:
            sub_srcs = self.get_catalog_of_subtracted_sources(freq, patch_num=patch_num, include_apodized_region=False)
            sub_srcs = sub_srcs[~sub_srcs['method'].eq('external')].copy()
            if min_snr is not None:
                sub_srcs = sub_srcs[sub_srcs['SNR'].ge(min_snr)].copy()
            # do the matching:
            match_radius = utils.fwhm2sigma(si.beam_fwhm[freq]) if (match_radius is None) else match_radius
            matched_catalog, unmatched_catalog, unmatched_true_catalog = fgresults.match_subtracted_to_true_sources(sub_srcs, true_srcs, match_radius,
                                                                                                          max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err,
                                                                                                          verbose=self.verbose, log=self.log)
            # save the catalogs
            matched_catalog.to_csv(matched_catalog_fname)
            unmatched_catalog.to_csv(unmatched_catalog_fname)
            unmatched_true_catalog.to_csv(unmatched_true_catalog_fname)

        return matched_catalog, unmatched_catalog, unmatched_true_catalog
    
    
    
            
                
    def true_clusters_catalog_fname(self, patch_num=None, padded=False):
        map_area_info = self._get_map_area_info(patch_num=patch_num, padded=padded)
        fname = f'true_clusters_{map_area_info}.csv'
        fname = os.path.join(self.matched_clusters_catalogs_dir(patch_num=patch_num), fname)
        return fname
    
    
    def _clusters_match_info(self, match_radius=1, min_snr=None, patch_num=None):
        map_area_info = self._get_map_area_info(patch_num=patch_num)
        match_dist_info = f'radius{simutils.round_str(match_radius, n=3)}arcmin'
        match_info = f'{match_dist_info}_{map_area_info}'
        if min_snr is not None:
            match_info = f'minSNR{simutils.round_str(min_snr)}'
        return match_info
    
    def matched_subtracted_clusters_catalog_fname(self, match_radius=1,  min_snr=None, patch_num=None):
        match_info = self._clusters_match_info(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr)
        fname = os.path.join(self.matched_clusters_catalogs_dir(patch_num=patch_num), f'matched_subtracted_clusters_{match_info}.csv')
        return fname
    
    def unmatched_subtracted_clusters_catalog_fname(self, match_radius=1,  min_snr=None, patch_num=None):
        match_info = self._clusters_match_info(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr)
        fname = os.path.join(self.matched_clusters_catalogs_dir(patch_num=patch_num), f'unmatched_subtracted_clusters_{match_info}.csv')
        return fname
    
    def matched_true_clusters_catalog_fname(self, match_radius=1,  min_snr=None, patch_num=None):
        match_info = self._clusters_match_info(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr)
        fname = os.path.join(self.matched_clusters_catalogs_dir(patch_num=patch_num), f'matched_true_clusters_{match_info}.csv')
        return fname
    
    def unmatched_true_clusters_catalog_fname(self, match_radius=1, min_snr=None,  patch_num=None):
        match_info = self._clusters_match_info(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr)
        fname = os.path.join(self.matched_clusters_catalogs_dir(patch_num=patch_num), f'unmatched_true_clusters_{match_info}.csv')
        return fname
    
    
    def matched_cluster_idxs_fname(self, match_radius=1,  min_snr=None, patch_num=None):
        match_info = self._clusters_match_info(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr)
        fname = os.path.join(self.matched_clusters_catalogs_dir(patch_num=patch_num), f'matched_cluster_idxs_{match_info}.yaml')
        return fname


    def get_true_clusters_catalog(self, patch_num=None, save=False):
        fname = self.true_clusters_catalog_fname(patch_num=patch_num)
        if os.path.exists(fname):
            catalog = fgcatalogs.load_catalog(fname)
        else:
            # get the catalog of all clusters in the full maps that were FG cleaned (including apodized region):
            full_catalog_fname = self.true_clusters_catalog_fname(patch_num=None, padded=True)
            if os.path.exists(full_catalog_fname):
                catalog = fgcatalogs.load_catalog(full_catalog_fname)
            else:
                catalog = self.get_catalog('tsz', include_padded_area=True)
                catalog = catalog.sort_values('M500', ascending=False).reset_index(drop=True)
                catalog['idx'] = list(range(len(catalog)))
                if save:
                    catalog.to_csv(full_catalog_fname)
                    self.infomsg(f"saved {full_catalog_fname}")
            # trim this catalog to only include clusters in un-apodized region (for full map or patch)
            catalog = self.remove_objects_in_apodized_region(catalog, patch_num=patch_num)
            if save:
                catalog.to_csv(fname)
                self.infomsg(f"saved {fname}")
        return catalog
    
    
    
    def match_to_true_clusters_catalog(self, match_radius=1, patch_num=None, min_snr=None):
        matched_catalog_fname = self.matched_subtracted_clusters_catalog_fname(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr)
        unmatched_catalog_fname = self.unmatched_subtracted_clusters_catalog_fname(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr)
        matched_true_catalog_fname = self.matched_true_clusters_catalog_fname(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr)
        unmatched_true_catalog_fname = self.unmatched_true_clusters_catalog_fname(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr)
        catalog_fnames = [matched_catalog_fname, unmatched_catalog_fname, matched_true_catalog_fname, unmatched_true_catalog_fname]
        catalogs_saved = all([os.path.exists(fname) for fname in catalog_fnames])
        
        # if using a higher min. SNR, check if catalogs were saved for the lowest SNR used:
        if (min_snr is not None) and (min_snr > np.min(self.clusters_snr_threshold_list)):
            lowersnr_matched_catalog_fname = self.matched_subtracted_clusters_catalog_fname(patch_num=patch_num, match_radius=match_radius)
            lowersnr_unmatched_catalog_fname = self.unmatched_subtracted_clusters_catalog_fname(patch_num=patch_num, match_radius=match_radius)
            lowersnr_matched_true_catalog_fname = self.matched_true_clusters_catalog_fname(patch_num=patch_num, match_radius=match_radius)
            lowersnr_unmatched_true_catalog_fname = self.unmatched_true_clusters_catalog_fname(patch_num=patch_num, match_radius=match_radius)
            lowersnr_catalog_fnames = [lowersnr_matched_catalog_fname, lowersnr_unmatched_catalog_fname, 
                                       lowersnr_matched_true_catalog_fname, lowersnr_unmatched_true_catalog_fname]
            lowersnr_catalogs_saved = all([os.path.exists(fname) for fname in lowersnr_catalog_fnames])
        else:
            lowersnr_catalogs_saved = False

        if catalogs_saved:
            matched_catalog = fgcatalogs.load_catalog(matched_catalog_fname)
            unmatched_catalog = fgcatalogs.load_catalog(unmatched_catalog_fname)
            matched_true_catalog = fgcatalogs.load_catalog(matched_true_catalog_fname)
            unmatched_true_catalog = fgcatalogs.load_catalog(unmatched_true_catalog_fname)
        elif lowersnr_catalogs_saved:
            lowersnr_matched_catalog = fgcatalogs.load_catalog(lowersnr_matched_catalog_fname)
            lowersnr_unmatched_catalog = fgcatalogs.load_catalog(lowersnr_unmatched_catalog_fname)
            lowersnr_matched_true_catalog = fgcatalogs.load_catalog(lowersnr_matched_true_catalog_fname)
            lowersnr_unmatched_true_catalog = fgcatalogs.load_catalog(lowersnr_unmatched_true_catalog_fname)
            # all measured clusters above the `min_snr`:
            subtracted_clusters_catalog = self.get_catalog_of_subtracted_clusters(include_apodized_region=False)
            subtracted_clusters_above_snr = subtracted_clusters_catalog[subtracted_clusters_catalog['SNR'].ge(min_snr)].copy()
            subtracted_cluster_idxs = subtracted_clusters_above_snr['idx'].values
            # load in dict w/ key for each true cluster that was matched, containing list of measured clusters that were matched to it
            lowersnr_matched_idxs = utils.load_yaml(self.matched_cluster_idxs_fname(patch_num=patch_num, match_radius=match_radius))
            # only keep measured clusters above the given snr
            tmp_cluster_idxs = {true_idx : [idx for idx in sub_idxs if (idx in subtracted_cluster_idxs)] for (true_idx, sub_idxs) in lowersnr_matched_idxs.items()}
            # remove any true clusters that no longer have a match
            matched_idxs = {true_idx: sub_idxs for (true_idx, sub_idxs) in tmp_cluster_idxs.items() if (len(sub_idxs) > 0)}
            # get catalogs for this `min_snr`:
            matched_catalog = lowersnr_matched_catalog[lowersnr_matched_catalog['SNR'].ge(min_snr)].copy()
            unmatched_catalog = lowersnr_unmatched_catalog[lowersnr_unmatched_catalog['SNR'].ge(min_snr)].copy()
            matched_true_catalog = lowersnr_matched_true_catalog[lowersnr_matched_true_catalog['idx'].isin(matched_idxs)].copy()
            unmatched_true_catalog = lowersnr_unmatched_true_catalog[~lowersnr_unmatched_true_catalog['idx'].isin(matched_idxs)].copy()  
        else:
            true_clusters = self.get_true_clusters_catalog(patch_num=patch_num)
            subtracted_clusters = self.get_catalog_of_subtracted_clusters(patch_num=patch_num, include_apodized_region=False)
            if min_snr is not None:
                subtracted_clusters = subtracted_clusters[subtracted_clusters['SNR'].ge(min_snr)]
            subtracted_clusters = subtracted_clusters.sort_values('SNR', ascending=False)
            matched_catalog, unmatched_catalog, matched_true_catalog, unmatched_true_catalog, matched_idxs = fgresults.match_subtracted_to_true_clusters(subtracted_clusters, true_clusters, 
                                                                                                                                               match_radius=match_radius, 
                                                                                                                                               verbose=self.verbose, log=self.log)
        
        if not catalogs_saved:
            utils.save_yaml(self.matched_cluster_idxs_fname(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr), matched_idxs)
            matched_catalog.to_csv(matched_catalog_fname)
            unmatched_catalog.to_csv(unmatched_catalog_fname)
            matched_true_catalog.to_csv(matched_true_catalog_fname)
            unmatched_true_catalog.to_csv(unmatched_true_catalog_fname)
        
        return matched_catalog, unmatched_catalog, matched_true_catalog, unmatched_true_catalog
    
    
            
        
        
        
    def _matched_patch_source_catalogs_saved(self, patch_num=0, freqs_to_match=None, min_snr=None,
                                      source_match_radius_dict=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err):
        source_catalogs_saved = True
        if self.subtract_sources:
            match_freqs = self.freqs if (freqs_to_match is None) else [freq for freq in freqs_to_match if (freq in self.freqs)]
            if len(match_freqs) == 0:
                raise ValueError(f"`{freqs_to_match = }`, but the map frequencies used for FG cleaning are"
                                 f" {self.freqs}. Each frequency (in GHz) in the `freqs_to_match` list must"
                                 " correspond to a map frequency.")
            if source_match_radius_dict is None:
                source_match_radius_dict = {freq: None for freq in match_freqs}
            for freq in match_freqs:
                kwargs = {'patch_num': patch_num, 'match_radius': source_match_radius_dict[freq], 
                          'max_abs_flux_diff_vs_err': max_abs_flux_diff_vs_err, 'min_snr': min_snr}
                matched_catalog_fname = self.matched_subtracted_sources_catalog_fname(freq, **kwargs)
                unmatched_catalog_fname = self.unmatched_subtracted_sources_catalog_fname(freq, **kwargs)
                unmatched_true_catalog_fname = self.unmatched_true_sources_catalog_fname(freq, **kwargs)
                catalogs_saved_for_freq = all([os.path.exists(fname) for fname in [matched_catalog_fname, unmatched_catalog_fname, unmatched_true_catalog_fname]])
                source_catalogs_saved = source_catalogs_saved and catalogs_saved_for_freq
        return source_catalogs_saved 
    
    
    def _matched_patch_cluster_catalogs_saved(self, patch_num=0, min_snr=None, cluster_match_radius=fgi.cluster_match_radius):
        cluster_catalogs_saved = True
        if self.subtract_clusters:
            kwargs = {'patch_num': patch_num, 'match_radius': cluster_match_radius, 'min_snr': min_snr}
            matched_catalog_fname = self.matched_subtracted_clusters_catalog_fname(**kwargs)
            unmatched_catalog_fname = self.unmatched_subtracted_clusters_catalog_fname(**kwargs)
            matched_true_catalog_fname = self.matched_true_clusters_catalog_fname(**kwargs)
            unmatched_true_catalog_fname = self.unmatched_true_clusters_catalog_fname(**kwargs)
            catalog_fnames = [matched_catalog_fname, unmatched_catalog_fname, matched_true_catalog_fname, unmatched_true_catalog_fname]
            cluster_catalogs_saved = all([os.path.exists(fname) for fname in catalog_fnames])
        return cluster_catalogs_saved
    
    
    def _matched_patch_catalogs_saved(self, patch_num=0, freqs_to_match=None, min_snr=None,
                                      source_match_radius_dict=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, cluster_match_radius=fgi.cluster_match_radius):
        source_catalogs_saved = self._matched_patch_source_catalogs_saved(patch_num=patch_num, 
                                                                          freqs_to_match=freqs_to_match, 
                                                                          min_snr=min_snr,
                                                                          source_match_radius_dict=source_match_radius_dict, 
                                                                          max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err)
        cluster_catalogs_saved = self._matched_patch_cluster_catalogs_saved(patch_num=patch_num, min_snr=min_snr, 
                                                                            cluster_match_radius=cluster_match_radius)
        return source_catalogs_saved and cluster_catalogs_saved
            
        
        
        
    
    
    def all_matched_patch_catalogs_saved(self, freqs_to_match=None, min_snr=None, source_match_radius_dict=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, cluster_match_radius=fgi.cluster_match_radius):
        kwargs = {'freqs_to_match': freqs_to_match, 'min_snr': min_snr, 'source_match_radius_dict': source_match_radius_dict,
                  'max_abs_flux_diff_vs_err': max_abs_flux_diff_vs_err, 'cluster_match_radius': cluster_match_radius}
        return all([self._matched_patch_catalogs_saved(patch_num=n, **kwargs) for n in self.patches.patch_nums])
        
    
    
    def _combine_matched_patch_source_catalogs(self, freq, match_radius=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, min_snr=None):
        kwargs = {'match_radius': match_radius, 'max_abs_flux_diff_vs_err': max_abs_flux_diff_vs_err, 'min_snr': min_snr}
        catalog_fnames = {'matched_sub': self.matched_subtracted_sources_catalog_fname(freq, **kwargs),
                          'unmatched_sub': self.unmatched_subtracted_sources_catalog_fname(freq, **kwargs),
                          'unmatched_true': self.unmatched_true_sources_catalog_fname(freq, **kwargs)}
        catalogs_saved = all([os.path.exists(fname) for fname in catalog_fnames.values()])
        
        if not catalogs_saved:
            # will update the `'idx'` column in (matched or unmatched) subtracted sources catalog for each patch 
            # to correspond to `'idx'` column in full catalog of subtracted sources:
            all_subtracted_sources = self.get_catalog_of_subtracted_sources(freq, include_apodized_region=True)
            all_subtracted_sources = all_subtracted_sources[~all_subtracted_sources['method'].eq('external')].copy()

            # load catalogs for each patch:
            catalogs = {key: {} for key in catalog_fnames.keys()}
            for patch_num in self.patches.patch_nums:
                patch_matched_catalog, patch_unmatched_catalog, patch_unmatched_true_catalog = self.match_to_true_sources_catalog(freq, patch_num=patch_num, **kwargs)
                catalogs['unmatched_true'][patch_num] = self._prepare_patch_catalog(patch_unmatched_true_catalog, patch_num, add_patch_idx_col=False, include_apodized_region=False)
                # update the `idx` column:
                all_subtracted_sources_in_patch = all_subtracted_sources[all_subtracted_sources['patch_num'].eq(patch_num)].copy()
                all_subtracted_sources_in_patch = all_subtracted_sources_in_patch.sort_values('patch_idx')
                for key, catalog in zip(['matched_sub', 'unmatched_sub'], [patch_matched_catalog, patch_unmatched_catalog]):
                    catalog = self._prepare_patch_catalog(catalog, patch_num, include_apodized_region=False).sort_values('patch_idx')
                    patch_idxs = catalog['patch_idx'].values
                    idxs = all_subtracted_sources_in_patch[all_subtracted_sources_in_patch['patch_idx'].isin(patch_idxs)]['idx'].values
                    catalog['idx'] = idxs.copy()
                    catalogs[key][patch_num] = catalog.copy()
                    
            # combine catalogs from patches into single catalogs, remove sources in apodized region, and save:
            for key, fname in catalog_fnames.items():
                catalog = fgcatalogs.combine_catalogs([catalogs[key][n] for n in self.patches.patch_nums])
                catalog = self.remove_objects_in_apodized_region(catalog).sort_values('idx')
                catalog.to_csv(fname)
                self.infomsg(f"saved {fname}")

    
    
    
    def _combine_matched_patch_cluster_catalogs(self, match_radius=1, min_snr=None):
        catalog_fnames = {'matched_sub': self.matched_subtracted_clusters_catalog_fname(match_radius=match_radius, min_snr=min_snr),
                          'unmatched_sub': self.unmatched_subtracted_clusters_catalog_fname(match_radius=match_radius, min_snr=min_snr),
                          'matched_true': self.matched_true_clusters_catalog_fname(match_radius=match_radius, min_snr=min_snr),
                          'unmatched_true': self.unmatched_true_clusters_catalog_fname(match_radius=match_radius, min_snr=min_snr)}
        catalogs_saved = all([os.path.exists(fname) for fname in catalog_fnames.values()])
        
        if not catalogs_saved:
            # will update the `'idx'` column in (matched or unmatched) subtracted clusters catalog for each patch 
            # to correspond to `'idx'` column in full catalog of subtracted clusters:
            all_subtracted_clusters = self.get_catalog_of_subtracted_clusters(include_apodized_region=True)

            # load catalogs for each patch:
            catalogs = {key: {} for key in catalog_fnames.keys()}
            matched_idxs = {} # key = true cluster idx ; value = list of subtracted cluster idxs matched to this true cluster
            for patch_num in self.patches.patch_nums:
                patch_matched_catalog, patch_unmatched_catalog, patch_matched_true_catalog, patch_unmatched_true_catalog = self.match_to_true_clusters_catalog(match_radius=match_radius, min_snr=min_snr, patch_num=patch_num)
                catalogs['matched_true'][patch_num] = self._prepare_patch_catalog(patch_matched_true_catalog, patch_num, add_patch_idx_col=False, include_apodized_region=False)
                catalogs['unmatched_true'][patch_num] = self._prepare_patch_catalog(patch_unmatched_true_catalog, patch_num, add_patch_idx_col=False, include_apodized_region=False)
                
                # update the `idx` column:
                all_subtracted_clusters_in_patch = all_subtracted_clusters[all_subtracted_clusters['patch_num'].eq(patch_num)].copy()
                all_subtracted_clusters_in_patch = all_subtracted_clusters_in_patch.sort_values('patch_idx')
                for key, catalog in zip(['matched_sub', 'unmatched_sub'], [patch_matched_catalog, patch_unmatched_catalog]):
                    catalog = self._prepare_patch_catalog(catalog, patch_num, include_apodized_region=False).sort_values('patch_idx')
                    patch_idxs = catalog['patch_idx'].values
                    idxs = all_subtracted_clusters_in_patch[all_subtracted_clusters_in_patch['patch_idx'].isin(patch_idxs)]['idx'].values
                    catalog['idx'] = idxs.copy()
                    catalogs[key][patch_num] = catalog.copy()
                
                # keep track of which subtracted cluster `idx`s are matched to each true cluster
                patch_matched_idxs = utils.load_yaml(self.matched_cluster_idxs_fname(patch_num=patch_num, match_radius=match_radius, min_snr=min_snr))
                for true_idx, idxs in patch_matched_idxs.items():
                    if true_idx in matched_idxs:
                        other_patch_idxs = matched_idxs[true_idx]
                    else:
                        other_patch_idxs = []
                    matched_idxs[true_idx] = [*other_patch_idxs, *idxs]
            
            # combine catalogs from patches into single catalogs, remove sources in apodized region, and save:
            for key, fname in catalog_fnames.items():
                catalog = fgcatalogs.combine_catalogs([catalogs[key][n] for n in self.patches.patch_nums])
                catalog = self.remove_objects_in_apodized_region(catalog).sort_values('idx')
                catalog.to_csv(fname)
                self.infomsg(f"saved {fname}")
            
            utils.save_yaml(self.matched_cluster_idxs_fname(match_radius=match_radius, min_snr=min_snr), matched_idxs)
                

    
    
    def save_combined_matched_patch_catalogs(self, freqs_to_match=None, source_match_radius_dict=None, max_abs_flux_diff_vs_err=fgi.max_abs_flux_diff_vs_err, cluster_match_radius=fgi.cluster_match_radius, min_snr=None):
        if self.subtract_sources:
            match_freqs = self.freqs if (freqs_to_match is None) else [freq for freq in freqs_to_match if (freq in self.freqs)]
            if len(match_freqs) == 0:
                raise ValueError(f"`{freqs_to_match = }`, but the map frequencies used for FG cleaning are"
                                 f" {self.freqs}. Each frequency (in GHz) in the `freqs_to_match` list must"
                                 " correspond to a map frequency.")
            if source_match_radius_dict is None:
                source_match_radius_dict = {freq: None for freq in match_freqs}
            for freq in match_freqs:
                self._combine_matched_patch_source_catalogs(freq, match_radius=source_match_radius_dict[freq], 
                                                            max_abs_flux_diff_vs_err=max_abs_flux_diff_vs_err, min_snr=min_snr)
            # make sure catalog of all true sources is also saved:
            self.get_true_sources_catalog(save=True)
        if self.subtract_clusters:
            self._combine_matched_patch_cluster_catalogs(match_radius=cluster_match_radius)
            # make sure catalog of all true clusters is also saved:
            self.get_true_clusters_catalog(save=True)


