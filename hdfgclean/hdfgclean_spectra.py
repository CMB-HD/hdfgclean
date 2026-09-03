import os
import time
import numpy as np
from pixell import enmap
from hdsims import siminfo as si, utils, simutils, maps, simpower
from hd_mock_data import hd_data
from . import fgclean_info as fgi, mpi, fgutils, fgresults, hdfgclean_maps


def load_previous_predicted_fgcleaned_spectra(lmax=si.lmax4spectra, prev_hd_data_version='v1.1'):
    lmax = lmax+1 # need to add 1 b/c of shape of pspy binning matrix
    ells = np.arange(lmax+1)
    datalib = hd_data.HDMockData(version=prev_hd_data_version)
    freqs = fgi.spectra_freqs
    spectra = {}
    for freq in freqs:
        hd_freq = 150 if (freq == 148) else freq # freq used for HD prediction
        noise_cls = datalib.white_noise_cls(hd_freq, output_lmax=lmax)['tt']
        fg_cls = datalib.fg_spectra(hd_freq, output_lmax=lmax)
        spectra[freq] = {'ells': ells.copy(),
                         'noise': {'cltt': noise_cls, 'dltt': utils.cl2dl(ells, noise_cls)},
                         'total_fg': {'cltt': np.zeros(lmax+1), 'dltt': np.zeros(lmax+1)},}
        for fg_name in ['tsz', 'ksz', 'cib', 'radio']:
            fg_cltt = fg_cls[fg_name].copy()
            spectra[freq][fg_name] = {'cltt': fg_cltt.copy(), 'dltt': utils.cl2dl(ells, fg_cltt.copy())}
            for key in ['cltt', 'dltt']:
                spectra[freq]['total_fg'][key] += spectra[freq][fg_name][key].copy()
        spectra[freq]['total_fg_noise'] = {}
        spectra[freq]['cib_radio'] = {}
        for key in ['cltt', 'dltt']:
            spectra[freq]['total_fg_noise'][key] = spectra[freq]['total_fg'][key] + spectra[freq]['noise'][key]
            spectra[freq]['cib_radio'][key] = spectra[freq]['cib'][key] + spectra[freq]['radio'][key]
    return spectra



class HDFGCleanSpectra(hdfgclean_maps.HDFGCleanMaps):


    def spectra_after_subtraction_dir(self, mask=False, make_dir=False, **kwargs):
        '''kwargs are for mask (passed to _get_mask_info)'''
        sub_spectra_dir = os.path.join(self.results_dir(), 'spectra')
        mask_kwargs = self._get_mask_kwargs(**kwargs)
        mask_info = self._get_mask_info(**mask_kwargs)
        if mask and (mask_info is not None):
            spectra_dir = os.path.join(sub_spectra_dir, f'mask_{mask_info}')
        else:
            spectra_dir = os.path.join(sub_spectra_dir, 'without_mask')
        if make_dir:
            utils.mkdir(sub_spectra_dir)
            utils.mkdir(spectra_dir)
        return spectra_dir


    # --- inv. mode-coupling matrix that includes the mask: ---

    def binning_dir_with_mask(self, make_dir=False, **kwargs):
        '''for inv. mcm with mask
        kwargs are for the mask (passed to _get_mask_info)
        '''
        spectra_dir = self.spectra_after_subtraction_dir(mask=True, make_dir=make_dir, **kwargs)
        binning_dir = os.path.join(spectra_dir, 'binning_and_mode_decoupling')
        if make_dir:
            binning_dir = utils.mkdir(binning_dir)
        return binning_dir




    def get_mode_coupling_fnames(self, bin_dl=False, beam=False, freq=None, mask=False, **kwargs):
        '''
        kwargs:
            lmax
            pol
            kwargs for mask
            shape, wcs, apod_width
            patch_num : if `patch_num` in kwargs, the shape, wcs, and apod width for the patch are used
        '''
        kwargs = self._add_patch_spectra_kwargs(**kwargs)
        mcm_fname, bbl_fname = super().get_mode_coupling_fnames(bin_dl=bin_dl, beam=beam, freq=freq, **kwargs)
        if mask:
            mask_kwargs = self._get_mask_kwargs(**kwargs)
            mcm_dir_for_mask = self.binning_dir_with_mask(**mask_kwargs)
            mcm_dir_without_mask = self.binning_dir()
            mcm_fname = mcm_fname.replace(mcm_dir_without_mask, mcm_dir_for_mask)
            bbl_fname = bbl_fname.replace(mcm_dir_without_mask, mcm_dir_for_mask)
            # if calculating mode-coupling with the mask for a smaller region within the map,
            # need to include info about both the ra and dec of the patch center
            # (without a mask, only the dec matters)
            patch_info = self._patch_info_for_spectra(include_ra_ctr=False, **kwargs)
            if patch_info is not None:
                new_patch_info = self._patch_info_for_spectra(include_ra_ctr=True, **kwargs)
                # make sure only the file name is updated, not the output directory:
                _, mcm_file = os.path.split(mcm_fname)
                _, bbl_file = os.path.split(bbl_fname)
                mcm_fname = os.path.join(mcm_dir_for_mask, mcm_file.replace(patch_info, new_patch_info))
                bbl_fname = os.path.join(mcm_dir_for_mask, bbl_file.replace(patch_info, new_patch_info))
        return mcm_fname, bbl_fname



    def _calc_mode_coupling_with_mask(self, bin_dl=False, beam=False, freq=None, save_maps=False, **kwargs):
        '''
        save_maps is just whether to save the apod window and the mask
        - default is False b/c don't want multiple mpi processes trying to save simultaneously

        kwargs:
            lmax
            pol
            kwargs for mask
            shape, wcs, apod_width
        '''
        window = self.get_apod_window(save=save_maps, **kwargs)
        mask_kwargs = self._get_mask_kwargs(**kwargs)
        mask = enmap.project(self.get_mask(freq, save=save_maps, **mask_kwargs), window.shape, window.wcs)
        binning_file = self.binning_file()
        pol = self.get_kwarg('pol', **kwargs)
        lmax = self.get_kwarg('lmax', **kwargs)
        if beam:
            if freq is None:
                raise ValueError(f"`{beam = }` and `{freq = }`. You must pass a frequency (in GHz) to correct for the beam.")
            freq = simutils.validate_sim_freq(freq)
            beam_fwhm = si.beam_fwhm[freq]
        else:
            beam_fwhm = None
        self.infomsg(f"calculating inv. mode-coupling matrix for the mask with {lmax = }, {beam_fwhm = }, {pol = }, {bin_dl = }")
        t = time.time()
        mbb_inv, bbl = simpower.calc_mode_coupling(window * mask, lmax, binning_file, bin_dl=bin_dl, beam_fwhm=beam_fwhm, pol=pol, pol_window=window)
        self.infomsg(f"{utils.tmsg(time.time() - t)} for inv. mode-coupling matrix")
        return mbb_inv, bbl


    def calc_mode_coupling(self, bin_dl=False, beam=False, freq=None, mask=False, **kwargs):
        '''
        if `patch_num` in kwargs, the shape, wcs, and apod width for the patch are used
        '''
        kwargs = self._add_patch_spectra_kwargs(**kwargs)
        if mask:
            mbb_inv, bbl = self._calc_mode_coupling_with_mask(bin_dl=bin_dl, beam=beam, freq=freq, **kwargs)
        else:
            mbb_inv, bbl = super().calc_mode_coupling(bin_dl=bin_dl, beam=beam, freq=freq, **kwargs)
        return mbb_inv, bbl


    # --- taking power spectra w/ mask: ---

    def _take_power_with_mask(self, imap, pixwin=True, beam=False, freq=None, bin_cl=True, bin_dl=False, **kwargs):
        '''
        kwargs:
            lmax
            pol
            kwargs for mask
            shape, wcs, apod_width
        '''
        window = self.get_apod_window(save=False, **kwargs)
        has_pol = len(imap.shape) > 2
        lmax = int(round(self.get_kwarg('lmax', **kwargs)))

        imap = enmap.project(imap.copy(), window.shape, window.wcs)
        if pixwin:
            # deconvolve pixwin before applying the mask
            imap = enmap.unapply_window(imap * window)
            # make sure we don't deconvolve it again
            pixwin = False
            # since the map was already apodized when deconvolving the pixel window,
            # pass a `window` that is one everywhere, so the map doesn't get apodized twice
            window = enmap.ones(window.shape, window.wcs)

        # apply the mask
        mask_kwargs = self._get_mask_kwargs(**kwargs)
        mask = enmap.project(self.get_mask(freq, save=False, **mask_kwargs), window.shape, window.wcs)
        if has_pol:
            # only apply mask to temperature map
            imap[0] *= mask
        else:
            imap *= mask

        # get the inverse mode-coupling matrix
        if not (bin_cl or bin_dl):
            raise ValueError(f"`{bin_cl = }` and `{bin_dl = }`. At least one of `bin_cl` or `bin_dl` must be `True`.")
        mbb_inv_dict = {}
        spec_types = []
        if bin_cl:
            spec_types.append('cl')
            mbb_inv_dict['cl'] = self.get_mode_coupling(bin_dl=False, beam=beam, freq=freq, binning_matrix=False, mask=True, **kwargs)
            if (not has_pol) and ('spin0xspin0' in mbb_inv_dict['cl']):
                mbb_inv_dict['cl'] = mbb_inv_dict['cl']['spin0xspin0']
        if bin_dl:
            spec_types.append('dl')
            mbb_inv_dict['dl'] = self.get_mode_coupling(bin_dl=True, beam=beam, freq=freq, binning_matrix=False, mask=True, **kwargs)
            if (not has_pol) and ('spin0xspin0' in mbb_inv_dict['dl']):
                mbb_inv_dict['dl'] = mbb_inv_dict['dl']['spin0xspin0']

        # take the power of the sim:
        sim_power = simpower.calc_sim_power(imap, window, lmax, self.binning_file(), mbb_inv_dict,
                                            bin_cl=bin_cl, bin_dl=bin_dl, deconvolve_pixwin=pixwin)
        return sim_power



    def take_power(self, imap, pixwin=True, beam=False, freq=None, bin_cl=True, bin_dl=False, mask=False, **kwargs):
        '''
        kwargs:
            lmax
            pol
            kwargs for mask
            shape, wcs, apod_width
            patch_num : if `patch_num` in kwargs, the shape, wcs, and apod width for the patch are used

        '''
        kwargs = self._add_patch_spectra_kwargs(**kwargs)
        if mask:
            sim_power = self._take_power_with_mask(imap, pixwin=pixwin, beam=beam, freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
        else:
            sim_power = super().take_power(imap, pixwin=pixwin, beam=beam, freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
        return sim_power



    # --- power of sims after subtraction ---


    def get_fgcleaned_sim_power_fname(self, freq, beam=False, noise=False, bin_dl=False,
                                              subtract_sources=True, subtract_clusters=True,
                                              sources_sub_info=None, clusters_sub_info=None,
                                              mask=False, **kwargs):
        '''
        kwargs :
        - passed to `get_sim_power_fname` method of `HDSims`
        - patch_num : if `patch_num` in kwargs, the shape, wcs, and apod width for the patch are used
        - for mask
        '''
        kwargs = self._add_patch_spectra_kwargs(**kwargs)
        fname_before_subtraction = super().get_sim_power_fname(freq=freq, beam=beam, noise=noise, bin_dl=bin_dl, **kwargs)
        sub_info = self._get_subtracted_components_info(subtract_sources=subtract_sources, subtract_clusters=subtract_clusters,
                                                        sources_sub_info=sources_sub_info, clusters_sub_info=clusters_sub_info, **kwargs)
        if sub_info is None: # nothing is being subtracted from map
            fname = fname_before_subtraction
            if mask: # we are still applying the mask, so save the spectra to the directory for that mask:
                masked_spectra_dir = self.spectra_after_subtraction_dir(mask=mask, **kwargs)
                fname = fname.replace(self.spectra_dir(), masked_spectra_dir)
        else:
            # remove path and file extension
            fname_before_subtraction = os.path.split(fname_before_subtraction)[1]
            fname_root_before_subtraction = os.path.splitext(fname_before_subtraction)[0]
            fname_info_before_subtraction = fname_root_before_subtraction.split('_')
            # add info about what was subtracted
            fname_info = [*fname_info_before_subtraction[:-1], f'sub_{sub_info}', fname_info_before_subtraction[-1]]
            fname_root = '_'.join(fname_info)
            # add (new) path and file ext.
            spectra_dir = self.spectra_after_subtraction_dir(mask=mask, **kwargs)
            fname = os.path.join(spectra_dir, f'{fname_root}.txt')
        return fname


    # NOTE : over-writing method in `hdsims` so we can call `load_sim_power` without needing to re-write it
    # NOTE : still defining a separate `get_fgcleaned_sim_power_fname` method (and just calling it here) to allow for different defaults
    def get_sim_power_fname(self, freq=None, beam=False, noise=False, bin_dl=False,
                            subtract_sources=False, subtract_clusters=False, mask=False, **kwargs):
        return self.get_fgcleaned_sim_power_fname(freq, beam=beam, noise=noise, bin_dl=bin_dl,
                                                          subtract_sources=subtract_sources, subtract_clusters=subtract_clusters,
                                                          mask=mask, **kwargs)


    # NOTE : need to overwrite `get_noise_sim_power_fname` so we can take power of noise-only sim w/ the mask
    # but everything else (related to noise sim power) should work when passing `mask=True`?
    #    because `mask=True` will be included in `kwargs` that get passed to `get_noise_sim_power_fname`, and to `take_power` if file doesn't exist?
    def get_noise_sim_power_fname(self, freq, pixwin=True, beam=False, bin_dl=False, mask=False, **kwargs):
        '''
        kwargs:
            passed to `HDSims.get_noise_sim_power_fname`
            patch_num : if `patch_num` in kwargs, the shape, wcs, and apod width for the patch are used
            kwargs for mask
        '''
        kwargs = self._add_patch_spectra_kwargs(**kwargs)
        fname = super().get_noise_sim_power_fname(freq, pixwin=pixwin, beam=beam, bin_dl=bin_dl, **kwargs)
        if mask:
            masked_spectra_dir = self.spectra_after_subtraction_dir(mask=mask, **kwargs)
            fname = fname.replace(self.spectra_dir(), masked_spectra_dir)
        return fname





    def take_fgcleaned_sim_power(self, freq=None, beam=False, noise=False, bin_cl=True, bin_dl=False, save=True, save_sim=False,
                                         subtract_sources=True, subtract_clusters=True, mask=False, **kwargs):
        if not (bin_cl or bin_dl):
            raise ValueError(f"`{bin_cl = }` and `{bin_dl = }`. At least one of `bin_cl` or `bin_dl` must be `True`.")
        sub_sources = self._can_subtract_sources(**kwargs) and subtract_sources
        sub_clusters = self._can_subtract_clusters(**kwargs) and subtract_clusters
        if not (sub_sources or sub_clusters or mask):
            sim_power = self.take_sim_power(freq=freq, beam=beam, noise=noise, bin_cl=bin_cl, bin_dl=bin_dl, save=save, save_sim=save_sim, **kwargs)
        else:
            sim = self.get_fgcleaned_sim(freq, beam=beam, noise=noise,
                                                 subtract_sources=subtract_sources, subtract_clusters=subtract_clusters,
                                                 save=save_sim, **kwargs)
            components = self._get_map_components(**kwargs)
            self.infomsg(f'taking power of sim after FG cleaning for {freq = }, {components = }, {beam = }, {noise = }')
            t = time.time()
            sim_power = self.take_power(sim, pixwin=True, beam=beam, freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, mask=mask, **kwargs)
            self.infomsg(f'{utils.tmsg(time.time() - t)} to take power')
            if save:
                pol = self.get_kwarg('pol', default=False, **kwargs)
                col_names = simutils.get_spectra_keys(components, pol=pol)
                spec_types = [] # cl or dl
                if bin_cl:
                    spec_types.append('cl')
                if bin_dl:
                    spec_types.append('dl')
                for spec_type in spec_types:
                    fname = self.get_fgcleaned_sim_power_fname(freq=freq, beam=beam, noise=noise, bin_dl=('dl' in spec_type),
                                                                       subtract_sources=subtract_sources, subtract_clusters=subtract_clusters,
                                                                       mask=mask, **kwargs)
                    keys = simutils.get_spectra_keys(components, cl=('cl' in spec_type), dl=('dl' in spec_type), pol=pol)
                    utils.save_dict_to_file(fname, sim_power, keys=keys, col_names=col_names)
                    self.infomsg(f"saved {fname}")
        return sim_power



    def get_sim_power(self, freq=None, beam=False, noise=False, bin_cl=True, bin_dl=False, save=True, save_sim=False,
                      subtract_sources=False, subtract_clusters=False, mask=False, **kwargs):
        kwargs = {**kwargs, 'subtract_sources': subtract_sources, 'subtract_clusters': subtract_clusters, 'mask': mask}
        try:
            sim_power = self.load_sim_power(freq=freq, beam=beam, noise=noise, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
        except FileNotFoundError:
            sim_power = self.take_fgcleaned_sim_power(freq=freq, beam=beam, noise=noise, bin_cl=bin_cl, bin_dl=bin_dl,
                                                      save=save, save_sim=save_sim, **kwargs)
        return sim_power


    def get_fgcleaned_sim_power(self, freq=None, beam=False, noise=False, bin_cl=True, bin_dl=False, save=True, save_sim=False,
                                subtract_sources=True, subtract_clusters=True, mask=False, **kwargs):
        sim_power = self.get_sim_power(freq=freq, beam=beam, noise=noise, bin_cl=bin_cl, bin_dl=bin_dl,
                                                 save=save, save_sim=save_sim, subtract_sources=subtract_sources,
                                                 subtract_clusters=subtract_clusters, mask=mask, **kwargs)
        return sim_power


    def _patch_kwargs_for_spectra(self, patch_num=None):
        if patch_num is not None:
            patch_shape, patch_wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=False)
            kwargs = {'patch_num': patch_num, 'shape': patch_shape, 'wcs': patch_wcs, 'apod_width': self.patches.patch_apod_width}
        else:
            kwargs = {}
        return kwargs


    def _add_patch_spectra_kwargs(self, patch_num=None, **kwargs):
        return {**kwargs, **self._patch_kwargs_for_spectra(patch_num=patch_num)}


    def _make_spectra_dirs_and_save_files(self, apply_mask=True, **kwargs):
        '''
        used when running in parallel to make sure files and directories exist
        '''
        if mpi.is_rank0:
            binning_file = self.binning_file()
            apod_window = self.get_apod_window(save=True, **kwargs)
            spectra_after_fgclean_dir = self.spectra_after_subtraction_dir(make_dir=True)
            if apply_mask:
                mask_kwargs = self._get_mask_kwargs(**kwargs)
                masked_spectra_dir = self.spectra_after_subtraction_dir(mask=True, make_dir=True, **mask_kwargs)
                masked_binning_dir = self.binning_dir_with_mask(make_dir=True, **mask_kwargs)
        mpi.comm.barrier()


    def _calculate_all_mode_coupling(self, spectra_freqs=fgi.spectra_freqs, bin_cl=False, bin_dl=True, apply_mask=True, mask_freqs=fgi.spectra_freqs, **kwargs):
        '''
        distribute calculation of all inverse mode-coupling matrices
        (that are required by the `get_sim_power_before_and_after_fgclean` method)
        over MPI processes

        reduces (total) time needed for calculations, and ensures that MPI processes don't try to simultaneously save to same file
        '''
        # figure out which inv. mode-coupling matrices need to be calculated:
        pol = self.get_kwarg('pol', **kwargs)
        bin_dls = []
        if bin_cl:
            bin_dls.append(False)
        if bin_dl:
            bin_dls.append(True)
        unsaved_mcm_info = []
        for freq in spectra_freqs:
            use_mask = [True, False] if (apply_mask and (freq in mask_freqs)) else [False]
            for mask in use_mask:
                for dl in bin_dls:
                    mcm_fname, _ = self.get_mode_coupling_fnames(bin_dl=dl, beam=True, freq=freq, mask=mask, **kwargs)
                    if pol:
                        mcm_saved = all([os.path.exists(mcm_fname[key]) for key in mcm_fname])
                    else:
                        mcm_saved = os.path.exists(mcm_fname)
                    if not mcm_saved:
                        unsaved_mcm_info.append({'freq': freq, 'bin_dl': dl, 'mask': mask})
        mpi.comm.barrier() # make sure all mpi processes agree on what is not saved yet
        if len(unsaved_mcm_info) > 0:
            if mpi.is_rank0:
                self.infomsg(f"calculating inverse mode-coupling matrices")
            # distribute the calculations over the available MPI processes:
            unsaved_mcm_indices = mpi.distribute(len(unsaved_mcm_info), mpi.size, mpi.rank)
            for i in unsaved_mcm_indices:
                mcm_kwargs = {**kwargs, **unsaved_mcm_info[i], 'beam': True, 'binning_matrix': True}
                self.get_mode_coupling(**mcm_kwargs)
            mpi.comm.barrier() # don't proceed until everything is calculated



    def _calculate_mode_coupling_for_patches(self, patch_nums=None, spectra_freqs=fgi.spectra_freqs, bin_cl=False, bin_dl=True, **kwargs):
        '''note: only without mask'''
        patch_nums = self.patches.patch_nums if (patch_nums is None) else patch_nums
        patch_row_nums = list(set([self.patches.get_patch_row_and_col(n)[0] for n in patch_nums]))
        # save an apod window and inverse mode-coupling matrices for patches in each row
        freqs_for_spectra = [freq for freq in spectra_freqs if (freq in self.freqs)]
        pol = self.get_kwarg('pol', **kwargs)
        bin_dls = []
        if bin_cl:
            bin_dls.append(False)
        if bin_dl:
            bin_dls.append(True)
        patch_mcm_kwargs = []
        for row_num in patch_row_nums:
            patch_num = self.patches.get_patch_num(patch_col_num=0, patch_row_num=row_num)
            patch_shape, patch_wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=False)
            if mpi.is_rank0: # save the apod. window
                patch_window = self.get_apod_window(shape=patch_shape, wcs=patch_wcs, apod_width=self.patches.patch_apod_width, save=True)
            for freq in freqs_for_spectra:
                for dl in bin_dls:
                    mcm_kwargs = {**kwargs, 'freq': freq, 'beam': True, 'bin_dl': dl,
                                  'shape': patch_shape, 'wcs': patch_wcs, 'apod_width': self.patches.patch_apod_width}
                    mcm_fname, _ = self.get_mode_coupling_fnames(**mcm_kwargs)
                    if not os.path.exists(mcm_fname):
                        patch_mcm_kwargs.append(mcm_kwargs.copy())
        mpi.comm.barrier() # make sure the windows were saved before proceeding
        if len(patch_mcm_kwargs) > 0:
            if mpi.is_rank0:
                self.infomsg(f"calculating inverse mode-coupling matrices for {patch_nums = }")
            patch_mcm_indices = mpi.distribute(len(patch_mcm_kwargs), mpi.size, mpi.rank)
            for i in patch_mcm_indices:
                self.get_mode_coupling(**patch_mcm_kwargs[i])
            mpi.comm.barrier() # don't proceed until everything is calculated


    def _save_spectra_before_and_after_fgclean(self, freq, bin_cl=False, bin_dl=True, **kwargs):
        kwargs = {**kwargs, 'beam': True, 'save': True}
        # map with all components:
        self.get_sim_power(freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
        # without cmb:
        if simutils.has_cmb(self.map_components):
            fg_components = [c for c in self.map_components if ('cmb' not in c)]
            sim_kwargs = {**kwargs, 'components': fg_components}
            self.get_sim_power(freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **sim_kwargs)
        # tSZ:
        if 'tsz' in self.map_components:
            sim_kwargs = {**kwargs, 'components': ['tsz']}
            self.get_sim_power(freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **sim_kwargs)
        # CIB + radio:
        sim_ptsrc_components = [c for c in ['cib', 'radio'] if (c in self.map_components)]
        if len(sim_ptsrc_components) > 0:
            sim_kwargs = {**kwargs, 'components': sim_ptsrc_components}
            self.get_sim_power(freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **sim_kwargs)
        # cmb-only or noise-only (if `kwargs['noise']` is `False` or `True`, respectively ; default is `False`)
        subtract_sources = kwargs.get('subtract_sources', False)
        subtract_clusters = kwargs.get('subtract_clusters', False)
        subtract_fgs = subtract_sources or subtract_clusters
        mask = kwargs.get('mask', False)
        noise = kwargs.get('noise', False)
        if (not subtract_fgs) or mask:
            if noise: # beam- and pixel-window-deconvolved
                self.get_noise_sim_power(freq=freq, pixwin=True, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
            elif simutils.has_cmb(self.map_components):
                sim_kwargs = {**kwargs, 'components': [simutils.cmb_component_name(self.map_components)]}
                self.get_sim_power(freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **sim_kwargs)




    def get_sim_power_before_and_after_fgclean(self, spectra_freqs=fgi.spectra_freqs, bin_cl=False, bin_dl=True, apply_mask=True, mask_freqs=fgi.spectra_freqs, **kwargs):
        '''
        kwargs are for mask and for sims/spectra
        '''
        if mpi.is_rank0:
            self.infomsg(f"taking power of maps before and after subtraction")
        kwargs = fgutils.dict_without_keys(kwargs, ['beam', 'noise', 'freq', 'save', 'shape', 'wcs'])
        freqs_for_mask = [freq for freq in mask_freqs if (freq in self.freqs)] if apply_mask else []
        freqs_for_spectra = [freq for freq in spectra_freqs if (freq in self.freqs)]
        if len(freqs_for_spectra) == 0:
            raise ValueError(f"`{spectra_freqs = }`, but the map frequencies used for FG cleaning are {self.freqs}. "
                             "Each frequency (in GHz) in the `spectra_freqs` list must correspond to a map frequency.")
        if apply_mask and mpi.is_rank0: # make sure masks are saved
            self.get_masks(freqs=freqs_for_mask, save=True, **self._get_mask_kwargs(**kwargs))
        self._make_spectra_dirs_and_save_files(apply_mask=apply_mask, **kwargs)
        t0 = time.time()

        # if running w/ multiple MPI processes, need to make sure inv mode-coupling matrices are already saved
        # (so two MPI processes don't calculate and try to save them to save file simultaneously) ;
        # otherwise, when running with a single process, just calculate each as it's needed
        if mpi.size > 1:
            self._calculate_all_mode_coupling(spectra_freqs=freqs_for_spectra, bin_cl=bin_cl, bin_dl=bin_dl,
                                              apply_mask=apply_mask, mask_freqs=freqs_for_mask, **kwargs)

        # distribute sim power spectra calculations over mpi processes:
        sim_spectra_kwargs = {'before': {**kwargs, 'subtract_sources': False, 'subtract_clusters': False},
                              'after': {**kwargs, 'subtract_sources': True, 'subtract_clusters': True},
                              'mask': {**kwargs, 'subtract_sources': True, 'subtract_clusters': True, 'mask': True}}
        spectra_description = {'before': 'before FG cleaning', 'after': 'after FG cleaning', 'mask': 'with mask after FG cleaning'}

        sim_spectra_info = []
        for freq in freqs_for_spectra:
            keys = ['before', 'after', 'mask'] if (freq in freqs_for_mask) else ['before', 'after']
            for key in keys:
                for noise in [False, True]:
                    sim_spectra_info.append({'freq': freq, 'key': key, 'noise': noise})

        spectra_indices = mpi.distribute(len(sim_spectra_info), mpi.size, mpi.rank)
        for i in spectra_indices:
            freq = sim_spectra_info[i]['freq']
            key = sim_spectra_info[i]['key']
            noise = sim_spectra_info[i]['noise']
            noise_info = 'with noise' if noise else 'without noise'
            self.infomsg(f"taking power {spectra_description[key]} of {freq} GHz maps {noise_info}")
            spectra_kwargs = {**sim_spectra_kwargs[key], 'noise': noise, }
            self._save_spectra_before_and_after_fgclean(freq, bin_cl=bin_cl, bin_dl=bin_dl, **spectra_kwargs)
        self.infomsg(f"{utils.tmsg(time.time() - t0)} for spectra")
        mpi.comm.barrier()




    def get_patch_power_before_and_after_fgclean(self, patch_num, spectra_freqs=fgi.spectra_freqs,
                                                 bin_cl=False, bin_dl=True, apply_mask=True, mask_freqs=fgi.spectra_freqs, **kwargs):
        '''
        kwargs are for mask and for sims/spectra
        '''
        self.infomsg(f"taking power of {patch_num = } maps before and after subtraction")
        t0 = time.time()
        kwargs = fgutils.dict_without_keys(kwargs, ['beam', 'noise', 'freq', 'save'])
        freqs_for_mask = [freq for freq in mask_freqs if (freq in self.freqs)] if apply_mask else []
        freqs_for_spectra = [freq for freq in spectra_freqs if (freq in self.freqs)]
        if len(freqs_for_spectra) == 0:
            raise ValueError(f"`{spectra_freqs = }`, but the map frequencies used for FG cleaning are {self.freqs}. "
                             "Each frequency (in GHz) in the `spectra_freqs` list must correspond to a map frequency.")
        patch_shape, patch_wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=False)
        kwargs = {**kwargs, 'patch_num': patch_num, 'shape': patch_shape, 'wcs': patch_wcs, 'apod_width': self.patches.patch_apod_width}
        sim_spectra_kwargs = {'before': {**kwargs, 'subtract_sources': False, 'subtract_clusters': False},
                              'after': {**kwargs, 'subtract_sources': True, 'subtract_clusters': True},
                              'mask': {**kwargs, 'subtract_sources': True, 'subtract_clusters': True, 'mask': True}}
        spectra_description = {'before': 'before FG cleaning', 'after': 'after FG cleaning', 'mask': 'with mask after FG cleaning'}
        for freq in freqs_for_spectra:
            keys = ['before', 'after', 'mask'] if (freq in freqs_for_mask) else ['before', 'after']
            for key in keys:
                for noise in [False, True]:
                    noise_info = 'with noise' if noise else 'without noise'
                    self.infomsg(f"taking power {spectra_description[key]} of {freq} GHz maps {noise_info} on patch {patch_num}")
                    spectra_kwargs = {**sim_spectra_kwargs[key], 'noise': noise, }
                    self._save_spectra_before_and_after_fgclean(freq, bin_cl=bin_cl, bin_dl=bin_dl, **spectra_kwargs)
        self.infomsg(f"{utils.tmsg(time.time() - t0)} for patch {patch_num} spectra")



    # --- coadded residual FG + noise curves ---

    def get_binned_previous_predicted_fgcleaned_spectra(self, prev_hd_data_version='v1.1', lmax=None):
        lmax = self.lmax if (lmax is None) else lmax
        ells = np.arange(lmax+2) # need to add 1 b/c of shape of pspy binning matrix
        datalib = hd_data.HDMockData(version=prev_hd_data_version)
        # load in CMB theory curve (and use finer binning for it):
        if os.path.exists(self.get_sim_theory_fname('cmb')):
            cmb_theo = self.load_sim_theory('cmb')
        else:
            cmb_theo = datalib.cmb_theory_spectra('lensed', output_lmax=lmax)
        bmat = datalib.binning_matrix(lmin=datalib.lmin, lmax=lmax)
        cmb_lbin = bmat @ ells[2:lmax+1]
        cmb_clbin = bmat @ cmb_theo['tt'][2:lmax+1]
        cmb_dlbin = bmat @ utils.cl2dl(ells[2:lmax+1], cmb_theo['tt'][2:lmax+1])
        # previous HD prediction for white noise, kSZ and FG-cleaned tSZ, CIB, radio:
        spectra = load_previous_predicted_fgcleaned_spectra(lmax=lmax)
        binned_spectra = {}
        for freq in spectra:
            binned_spectra[freq] = {'cmb': {'ells': cmb_lbin, 'cltt': cmb_clbin, 'dltt': cmb_dlbin},}
            spectra_component_names = [name for name in spectra[freq].keys() if (name != 'ells')]
            for fg_name in spectra_component_names:
                binned_spectra[freq][fg_name] = {}
                for spec_type in ['cl', 'dl']:
                    bin_dl = (spec_type == 'dl')
                    spectra_dict = {'ells': ells, 'tt': spectra[freq][fg_name]['cltt']}
                    binned_spectra_dict = self.bin_theory(spectra_dict, bin_dl=bin_dl)
                    binned_spectra[freq][fg_name]['ells'] = binned_spectra_dict['ells']
                    binned_spectra[freq][fg_name][f'{spec_type}tt'] = binned_spectra_dict['tt'].copy()
        return binned_spectra



    def get_coadded_fgcleaned_sim_power(self, freqs=fgi.spectra_freqs, patch_num=None, subtract_sources=True,
                                        subtract_clusters=True, mask=True, cl=False, dl=True, interp=False,
                                        fg_components=None, lmax=None, save=True, **kwargs):
        '''kwargs are for the mask'''
        kwargs = {'freqs': freqs, 'patch_num': patch_num,
                  'subtract_sources': subtract_sources, 'subtract_clusters': subtract_clusters,
                  'mask': mask, 'interp': interp, 'fg_components': fg_components, 'lmax': lmax,
                  **self._get_mask_kwargs(**kwargs)}
        coadded_spectra = {}

        # first, try to load saved spectra:
        fnames = {}
        if dl:
            fnames['dl'] = self.get_coadded_fgcleaned_sim_power_fname(dl=True, **kwargs)
        if cl:
            fnames['cl'] = self.get_coadded_fgcleaned_sim_power_fname(dl=False, **kwargs)
        spec_types = list(fnames.keys())
        spec_keys = {spec_type: f'{spec_type}tt' for spec_type in spec_types}
        for spec_type, fname in fnames.items():
            if os.path.exists(fname):
                coadded_spectra['ells'], coadded_spectra[spec_keys[spec_type]] = np.loadtxt(fname, unpack=True)

        # then, calculate spectra if necessary:
        missing_spec_types = [spec_type for spec_type in spec_types if (spec_keys[spec_type] not in coadded_spectra)]
        if len(missing_spec_types) > 0:
            calc_cl = ('cl' in missing_spec_types)
            calc_dl = ('dl' in missing_spec_types)
            coadded_spectra_dict = self.coadd_fgcleaned_sim_power(cl=calc_cl, dl=calc_dl, **kwargs)
            for key in coadded_spectra_dict:
                if key not in coadded_spectra:
                    coadded_spectra[key] = coadded_spectra_dict[key].copy()

        # save the coadded spectra, if necessary:
        if save:
            header = "multipoles, TT coadded residual FG + noise power spectrum [uK^2]"
            for spec_type, fname in fnames.items():
                key = spec_keys[spec_type]
                if not os.path.exists(fname):
                    np.savetxt(fname, np.column_stack([coadded_spectra['ells'], coadded_spectra[key]]), header=header)

        return coadded_spectra


    def coadd_fgcleaned_sim_power(self, freqs=fgi.spectra_freqs, patch_num=None, subtract_sources=True, subtract_clusters=True,
                                  mask=True, cl=False, dl=True, interp=False, fg_components=None, lmax=None, **kwargs):
        '''
        kwargs are for the mask

        lmax is value used for calculating sim power spectra (passed to `get_sim_power` method)

        NOTE : sim spectra are binned; can bin as C_ell's or D_ell's ;
              if interpolating coadded power to all ell's, we ALWAYS coadd the binned sim D_ell's
              - i.e. may calculate a new binning matrix even if `dl=False`!!

        NOTE : we can only interp up to the last bin center of the binned sim spectra
                - so if `lmax` is passed, output power spectra only output to ell_max ~ lmax - (bin_width / 2)
               we can also only interp from the first bin center
                - IF using default list of freqs, will use prev. predicted hd spectra to "fill in" a value
                  at ell=2 used to interp down this far
        '''
        freqs = self._validate_freqs(freqs)
        _, imap_fg_components, _ = simutils.separate_component_list(self.map_components)
        if fg_components is None:
            fg_components = imap_fg_components
        else:
            fg_components = simutils.validate_map_components(fg_components, valid_components=imap_fg_components)
        subtract_sources = subtract_sources and self._can_subtract_sources(components=fg_components)
        subtract_clusters = subtract_clusters and self._can_subtract_clusters(components=fg_components)
        bin_dl = True if interp else dl
        bin_cl = False if interp else cl
        sim_spectra_kwargs = {'beam': True, 'noise': True, 'components': fg_components,
                              'bin_cl': bin_cl, 'bin_dl': bin_dl,
                              'subtract_sources': subtract_sources, 'subtract_clusters': subtract_clusters,
                              'mask': mask, **self._patch_kwargs_for_spectra(patch_num=patch_num),
                              **self._get_mask_kwargs(**kwargs)}
        if lmax is not None:
            sim_spectra_kwargs['lmax'] = int(lmax)

        sim_spectra = {freq: self.get_sim_power(freq=freq, **sim_spectra_kwargs) for freq in freqs}
        coadded_spectra = {}
        if interp:
            spectra_to_coadd = {freq: sim_spectra[freq]['dltt'] for freq in freqs}
            lbin = sim_spectra[freqs[0]]['ells']
            coadd_kwargs = {}
            if set(freqs) == set(fgi.spectra_freqs):
                # use previous prediction of hd residual fg+noise spectra at ell = 2
                sim_spectra_lmax = int(np.max(lbin)) - 1 # last bin center
                prev_spectra = load_previous_predicted_fgcleaned_spectra(lmax=sim_spectra_lmax)
                coadd_kwargs['fill_val_ells'] = prev_spectra[90]['ells']
                coadd_kwargs['fill_vals'] = {freq: prev_spectra[freq]['total_fg_noise']['dltt'] for freq in freqs}
                coadd_kwargs['ells_to_fill'] = [2]
                coadd_kwargs['lmin'] = 0
            coadded_spectra['ells'], coadd_dls = fgresults.calc_coadded_noise(lbin, spectra_to_coadd, interp=True, **coadd_kwargs)
            if dl:
                coadded_spectra['dltt'] = coadd_dls.copy()
            if cl:
                coadd_lfact = coadded_spectra['ells'] * (coadded_spectra['ells'] + 1) / (2 * np.pi)
                loc = np.where(coadded_spectra['ells'] >= 2)
                coadd_spectra['cltt'] = np.zeros(coadd_dls.shape)
                coadd_spectra['cltt'][loc] = coadd_dls[loc] / coadd_lfact[loc]
        else:
            lbin = sim_spectra[freqs[0]]['ells']
            spec_keys = [key for key in sim_spectra[freqs[0]] if (key != 'ells')]
            for key in spec_keys:
                spectra_to_coadd = {freq: sim_spectra[freq][key] for freq in freqs}
                coadded_spectra['ells'], coadded_spectra[key] = fgresults.calc_coadded_noise(lbin, spectra_to_coadd, interp=False)

        return coadded_spectra


    def get_coadded_fgcleaned_sim_power_fname(self, freqs=fgi.spectra_freqs, patch_num=None, subtract_sources=True,
                                              subtract_clusters=True, mask=True, dl=True, interp=False,
                                              fg_components=None, lmax=None, **kwargs):
        '''kwargs are for the mask'''
        freqs = self._validate_freqs(freqs)
        _, imap_fg_components, _ = simutils.separate_component_list(self.map_components)
        if fg_components is None:
            fg_components = imap_fg_components
        else:
            fg_components = simutils.validate_map_components(fg_components, valid_components=imap_fg_components)
        subtract_sources = subtract_sources and self._can_subtract_sources(components=fg_components)
        subtract_clusters = subtract_clusters and self._can_subtract_clusters(components=fg_components)
        sim_spectra_kwargs = {'beam': True, 'noise': True, 'components': fg_components, 'bin_dl': dl,
                              'subtract_sources': subtract_sources, 'subtract_clusters': subtract_clusters,
                              'mask': mask, **self._patch_kwargs_for_spectra(patch_num=patch_num),
                              **self._get_mask_kwargs(**kwargs)}
        if lmax is not None:
            sim_spectra_kwargs['lmax'] = int(lmax)
        # get filename for a single freq
        spectra_dir, fname_template = os.path.split(self.get_sim_power_fname(freq=freqs[0], **sim_spectra_kwargs))
        fname_root, fname_ext = os.path.splitext(fname_template)
        fname_parts = fname_root.split('_')
        # update filename
        freqs_info = [f'{int(freq):03d}' for freq in freqs]
        spec_type_info = 'dls' if dl else 'cls'
        for i, fname_info in enumerate(fname_parts):
            if fname_info == freqs_info[0]:
                fname_parts[i] = '_'.join(['coadd', *freqs_info])
            elif ('beam' in fname_info) and ('arcmin' in fname_info):
                fname_parts[i] = 'beam'
            elif ('noise' in fname_info) and ('uKarcmin' in fname_info):
                fname_parts[i] = 'noise'
            elif interp and (fname_info == spec_type_info):
                fname_parts[i] = f'interp_{spec_type_info}'
        fname_root = '_'.join(fname_parts)
        fname = os.path.join(spectra_dir, f'{fname_root}{fname_ext}')
        return fname




    # --- misc ---

    def _save_sources_spectra_for_plot(self, patch_num=None):
        if patch_num is None:
            patch_num = self.get_patch_num(ra=self.ra_ctr, dec=self.dec_ctr)
        patch_kwargs = {'beam': True, 'noise': False,
                        'components': ['cib', 'radio'], 'bin_dl': True, 'bin_cl': False,
                        **self._patch_kwargs_for_spectra(patch_num=patch_num)}
        # distribute calculation over mpi processes
        tasks = []
        for freq in fgi.freqs:
            for mask in [False, True]:
                tasks.append((freq, mask))
        task_indices = mpi.distribute(len(tasks), mpi.size, mpi.rank)
        for i in task_indices:
            freq, mask = tasks[i]
            if not mask:
                # power of CIB+radio before any FG cleaning
                self.infomsg(f"taking power of {freq} GHz CIB+radio on patch {patch_num} before FG cleaning")
                srcs_spectra = self.get_sim_power(freq=freq, **patch_kwargs)
                # power of CIB+radio after only subtracting bright sources
                self.infomsg(f"taking power of {freq} GHz CIB+radio on patch {patch_num} after removing sources "
                             f"with SNR >= {simutils.round_str(np.min(self.sources_snr_threshold_list))}")
                sub_bright_srcs_spectra = self._get_bright_src_sub_spectra(freq, patch_num=patch_num,
                                                                           bin_dl=bin_dl, bin_cl=bin_cl)
            else:
                # power of CIB+radio after subtracting all sources
                self.infomsg(f"taking power of {freq} GHz CIB+radio on patch {patch_num} after FG cleaning")
                sub_srcs_spectra = self.get_sim_power(freq=freq, subtract_sources=True, mask=True, **patch_kwargs)
        mpi.comm.barrier()


    def _get_bright_src_sub_spectra(self, freq, patch_num=0, bin_cl=False, bin_dl=True, **kwargs):
        kwargs = self._add_patch_spectra_kwargs(patch_num=patch_num, **kwargs)
        ptsrc_components = [c for c in ['cib', 'radio'] if (c in self.map_components)]
        spectra_fnames = {}
        if bin_cl:
            spectra_fnames['cl'] = self.get_fgcleaned_sim_power_fname(freq, components=ptsrc_components, beam=True, noise=False, bin_dl=False, sources_sub_info='bright', **kwargs)
        if bin_dl:
            spectra_fnames['dl'] = self.get_fgcleaned_sim_power_fname(freq, components=ptsrc_components, beam=True, noise=False, bin_dl=True, sources_sub_info='bright', **kwargs)
        spectra_saved = all([os.path.exists(spectra_fnames[spec_type]) for spec_type in spectra_fnames.keys()])

        if not spectra_saved:
            self._init_patch_ptsrclib(patch_num=patch_num)
            self.ptsrclibs[patch_num]._init_ptsrclib(freq)
            bright_srcs_catalog = self.ptsrclibs[patch_num].ptsrclibs[freq].load_measured_bright_srcs_catalog()
            patch_shape, patch_wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=False)
            padded_patch_shape, padded_patch_wcs = self.patches.get_patch_geometry(patch_num=patch_num, padded=True)
            beam_fwhm = self._patch_beam_fwhms(patch_num=patch_num)[freq]

            bright_srcs_map = maps.make_src_map(padded_patch_shape, padded_patch_wcs, [bright_srcs_catalog], freq)
            bright_srcs_map = maps.convolve_sim_with_beam(enmap.apply_window(bright_srcs_map), beam_fwhm)
            bright_srcs_map = enmap.project(bright_srcs_map, patch_shape, patch_wcs)

            imap = self.get_sim(freq=freq, components=ptsrc_components, beam=True, noise=False, shape=patch_shape, wcs=patch_wcs)
            sub_map = imap - bright_srcs_map
            sim_spectra = self.take_power(sub_map, pixwin=True, beam=True, freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
            # save it
            for spec_type, fname in spectra_fnames.items():
                cols = [sim_spectra['ells'], sim_spectra[f'{spec_type}tt']]
                np.savetxt(fname, np.column_stack(cols))

        else: #  load it
            sim_spectra = {}
            for spec_type, fname in spectra_fnames.items():
                sim_spectra['ells'], sim_spectra[f'{spec_type}tt'] = np.loadtxt(fname, unpack=True)

        return sim_spectra

