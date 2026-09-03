import os
import time
import numpy as np
from scipy import ndimage
from pixell import enmap
from hdsims import hdsims, siminfo as si, utils, simutils, fgcatalogs, maps
from . import mpi, fgclean_info as fgi, fgutils, fgmaps


def get_ft(imap, apod_width=0, normalize=True):
    '''apod_width in degrees'''
    real_map = imap.copy()
    if apod_width > 0:
        apod_npix = fgmaps.dist2npix(apod_width, imap.shape, imap.wcs, units='degrees')
        real_map = enmap.apod(real_map, apod_npix)
    return enmap.fft(real_map, normalize=normalize)


def get_inv_ft(ftmap, normalize=True):
    return enmap.ifft(ftmap, normalize=normalize).real


def calc_power2d(imap, imap2=None, apod_width=0, smooth_npix=0):
    fmap = get_ft(imap, apod_width=apod_width)
    if imap2 is not None:
        fmap2 = get_ft(imap2, apod_width=apod_width)
    else:
        fmap2 = fmap
    p2d = fmap * np.conj(fmap2)
    if smooth_npix > 0:
        smooth_kernel_size = (smooth_npix, smooth_npix)
        shifted_p2d = enmap.fftshift(p2d.copy())
        shifted_p2d[:] = ndimage.gaussian_filter(shifted_p2d[:], smooth_kernel_size)
        p2d = enmap.ifftshift(shifted_p2d)
    return p2d.real


def calc_sn_map(filt_map, rms_map):
    sn_map = enmap.zeros(filt_map.shape, filt_map.wcs)
    mask = np.greater(rms_map, 0)
    sn_map[mask] = filt_map[mask] / rms_map[mask]
    sn_map[np.isnan(sn_map)] = 0
    return sn_map  


def get_even_rms_gw_pix(grid_size_pix, num_pix):
    num_cells = int(round(num_pix / grid_size_pix))
    num_pix_in_cells = num_cells * grid_size_pix
    if num_pix_in_cells == num_pix:
        gw_pix = grid_size_pix
    else:
        if num_pix_in_cells < num_pix:
            gw_pix_smaller = int(round(num_pix / (num_cells + 1)))
            gw_pix_larger = int(round(num_pix / num_cells))
        else:
            gw_pix_smaller = int(round(num_pix / num_cells))
            gw_pix_larger = int(round(num_pix / (num_cells - 1)))
        gw_diff_smaller = abs(grid_size_pix - gw_pix_smaller)
        gw_diff_larger = abs(grid_size_pix - gw_pix_larger)
        gw_pix = gw_pix_smaller if (gw_diff_smaller < gw_diff_larger) else gw_pix_larger
    return gw_pix


def get_even_rms_gw(rms_gw, shape, wcs):
    pixel_size = maps.get_map_resolution(shape, wcs)
    grid_size_pix = int(round(rms_gw / pixel_size))
    
    ny, nx = shape
    num_x_cells = int(round(nx / grid_size_pix))
    num_y_cells = int(round(ny / grid_size_pix))
    
    # check if the grid size and map size results in cells of different sizes
    # (b/c nx or ny not divisible by rms_gw)
    nx_in_cells = num_x_cells * grid_size_pix
    if nx_in_cells != nx:
        xgw = pixel_size * get_even_rms_gw_pix(grid_size_pix, nx)
    else:
        xgw = rms_gw
        
    ny_in_cells = num_y_cells * grid_size_pix
    if ny_in_cells != ny:
        ygw = pixel_size * get_even_rms_gw_pix(grid_size_pix, ny)
    else:
        ygw = rms_gw
        
    return xgw, ygw 




def make_rms_grid(imap, rms_gw, apod_width=0,
                 fix_rms_gw=False, use_overlap_pix=True):
    # get map resolution & geometry of region to use for RMS calculation:
    pixel_size = maps.get_map_resolution(imap.shape, imap.wcs)
    apod_npix = fgmaps.dist2npix(apod_width, imap.shape, imap.wcs, units='degrees')
    shape_for_rms, wcs_for_rms = fgmaps.trim_map_geometry(imap.shape, imap.wcs, width_to_remove=apod_width)
    
    # set up the grid on the region used to calculate rms:
    ny, nx = shape_for_rms
    if fix_rms_gw:
        rms_gw_x = rms_gw
        rms_gw_y = rms_gw
    else:
        # try to automatically find grid size in each direction,
        # such that each cell is approximately the same size:
        rms_gw_x, rms_gw_y = get_even_rms_gw(rms_gw, shape_for_rms, wcs_for_rms)
    # number of pixels in each direction per cell:
    grid_size_xpix = int(round(rms_gw_x / pixel_size))
    grid_size_ypix = int(round(rms_gw_y / pixel_size))
    # number of cells in each direction:
    num_x_cells = int(round(nx / grid_size_xpix))
    num_y_cells = int(round(ny / grid_size_ypix))
    
    # get the x and y pixels (in the map of the un-apodized region used to calculate RMS) at the bin/cell edges: 
    x_bin_edges = np.linspace(0, nx, num_x_cells + 1, dtype=int)
    y_bin_edges = np.linspace(0, ny, num_y_cells + 1, dtype=int)
    x_lower_edges = x_bin_edges[:-1].copy()
    x_upper_edges = x_bin_edges[1:].copy()
    y_lower_edges = y_bin_edges[:-1].copy()
    y_upper_edges = y_bin_edges[1:].copy()
    if use_overlap_pix: # use region outside of cell to calculate RMS value to use within the cell:
        overlap_xpix = int(round(grid_size_xpix / 2))
        overlap_ypix = int(round(grid_size_ypix / 2))
        x_lower_edges[1:] -= overlap_xpix
        x_upper_edges[:-1] += overlap_xpix
        y_lower_edges[1:] -= overlap_ypix
        y_upper_edges[:-1] += overlap_ypix

    # get pixels in the full map corresponding to the bin/cell edges:
    imap_nx, imap_ny = imap.shape
    # edges of the cells
    imap_x_bin_edges = x_bin_edges.copy() + apod_npix
    imap_x_bin_edges[0] = 0
    imap_x_bin_edges[-1] = imap_nx
    imap_y_bin_edges = y_bin_edges.copy() + apod_npix
    imap_y_bin_edges[0] = 0
    imap_y_bin_edges[-1] = imap_ny
    # upper/lower edges of each cell
    imap_x_lower_edges = imap_x_bin_edges[:-1].copy()
    imap_x_upper_edges = imap_x_bin_edges[1:].copy()
    imap_y_lower_edges = imap_y_bin_edges[:-1].copy()
    imap_y_upper_edges = imap_y_bin_edges[1:].copy()

    rms_edges = (x_lower_edges, x_upper_edges, y_lower_edges, y_upper_edges)
    imap_edges = (imap_x_lower_edges, imap_x_upper_edges, imap_y_lower_edges, imap_y_upper_edges)
    
    return rms_edges, imap_edges



def calc_rms_map(imap, rms_gw, apod_width=0,
                 fix_rms_gw=False, use_overlap_pix=True, 
                 niter=10, nsigma=3,
                 smooth=True, smooth_pix=None):
    # get map resolution & geometry of region to use for RMS calculation:
    pixel_size = maps.get_map_resolution(imap.shape, imap.wcs)
    apod_npix = fgmaps.dist2npix(apod_width, imap.shape, imap.wcs, units='degrees')
    shape_for_rms, wcs_for_rms = fgmaps.trim_map_geometry(imap.shape, imap.wcs, width_to_remove=apod_width)
    
    # get the pixel locations of the bin/grid/cell edges for both the 
    # un-apodized region used to calculate RMS, and for the full map:
    rms_edges, imap_edges = make_rms_grid(imap, rms_gw, apod_width=apod_width, fix_rms_gw=fix_rms_gw, use_overlap_pix=use_overlap_pix)
    x_lower_edges, x_upper_edges, y_lower_edges, y_upper_edges = rms_edges
    imap_x_lower_edges, imap_x_upper_edges, imap_y_lower_edges, imap_y_upper_edges = imap_edges
    
    # cut out the un-apodized region to use
    cutout = enmap.project(imap.copy(), shape_for_rms, wcs_for_rms)
    
    # calculate RMS within each cell of the un-apodized region, 
    # and fill the corresponding cell (which is larger at the map edges) within the full map region with that RMS value:
    rms_map = np.zeros(imap.shape)
    for i, (y0, y1, ymin, ymax) in enumerate(zip(y_lower_edges, y_upper_edges, imap_y_lower_edges, imap_y_upper_edges)):
        for j, (x0, x1, xmin, xmax) in enumerate(zip(x_lower_edges, x_upper_edges, imap_x_lower_edges, imap_x_upper_edges)):
            values_in_cell = np.array(cutout[y0:y1, x0:x1])
            good_pixels = np.not_equal(values_in_cell, 0) # don't include zeros
            cell_values = values_in_cell[good_pixels].copy()
            if np.not_equal(cell_values, 0).sum() != 0:
                cell_mean = np.mean(cell_values)
                cell_std = np.std(cell_values)
                for i in range(niter):
                    # find values that  are within mean +/- nsigma * sigma
                    mask = np.less(abs(cell_values), abs(cell_mean + nsigma * cell_std))
                    if mask.sum() > 0:
                        # update mean and standard deviation using only values
                        #  within +/- nsigma * sigma of current mean
                        # (i.e., remove extreme values from estimate)
                        cell_mean = np.mean(cell_values[mask])
                        cell_std = np.std(cell_values[mask])
                rms_map[ymin:ymax, xmin:xmax] = cell_std
    rms_map = enmap.enmap(rms_map, imap.wcs)
    
    if smooth:
        # smooth edges of grid cells by convolving with a Gaussian:
        smooth_pix = round((rms_gw/10) / pixel_size) if (smooth_pix is None) else smooth_pix
        smooth_sigma = utils.arcmin2rad(smooth_pix * pixel_size)
        smooth_apod_npix = int(round(apod_npix/2)) # need to apodize before smoothing
        rms_map = enmap.smooth_gauss(enmap.apod(rms_map, smooth_apod_npix), smooth_sigma)
        
        # re-fill the region in map that was just apodized (to avoid dividing by zero):
        rms_map[:apod_npix, :] = rms_map[apod_npix+1, :]   # upper edge
        rms_map[-apod_npix:, :] = rms_map[-apod_npix-1, :] # lower edge
        # left edge
        fill_vals = np.transpose(rms_map[:, :apod_npix])
        fill_vals[:apod_npix,:] = rms_map[:, apod_npix+1].copy()
        rms_map[:, :apod_npix] = np.transpose(fill_vals)
        # right edge
        fill_vals = np.transpose(rms_map[:, -apod_npix:])
        fill_vals[-apod_npix:, :] = rms_map[:, -apod_npix-1].copy()
        rms_map[:, -apod_npix:] = np.transpose(fill_vals)
        # corners:
        rms_map[:apod_npix, :apod_npix] = rms_map[apod_npix+1, apod_npix+1]
        rms_map[:apod_npix, -apod_npix:] = rms_map[apod_npix+1, -apod_npix-1]
        rms_map[-apod_npix:, :apod_npix] = rms_map[-apod_npix-1, apod_npix+1]
        rms_map[-apod_npix:, -apod_npix:] = rms_map[-apod_npix-1, -apod_npix-1]

    return rms_map



# ----- sources: -----

def apply_filter(imap, filt, apod_width=0):
    ftmap = get_ft(imap, apod_width=apod_width)
    return get_inv_ft(ftmap * filt.copy(), normalize=False)


def calc_filter(shape, wcs, noise_p2d, beam_template, deconvolve_pixwin=True):
    """`normalize` is passed to `enmap.fft` (doesn't refer to filter normalization)"""
    # calculate un-normalized filter:
    filt = enmap.zeros(shape, wcs)
    if not deconvolve_pixwin: # then account for effect of pixwin in the filter
        ft_beam = get_ft(enmap.apply_window(beam_template.copy()))
    else:
        ft_beam = get_ft(beam_template)
    good_pixels = np.not_equal(noise_p2d, 0)
    filt[good_pixels] = abs(ft_beam)[good_pixels] / noise_p2d[good_pixels]
    # calculate the normalization
    filt_beam = apply_filter(enmap.apply_window(beam_template), filt)
    if deconvolve_pixwin:
        filt_beam = enmap.unapply_window(filt_beam)
    norm = np.max(filt_beam.data)
    filt /= norm
    return filt


def calc_filtered_maps(imap, filt, grid_size_arcmin, 
                       apod_width=0, # in degrees
                       apodize=False, # whether to apodize map (False if already apodized by the `apod_width`)
                       deconvolve_pixwin=True,
                       rms_map=None, smooth_rms_map=True, smooth_rms_pix=None,
                       rms_niter=10, rms_nsigma=3, fix_rms_gw=False, use_overlap_pix=True,):
    filt_map = apply_filter(imap, filt, apod_width=apod_width)
    if deconvolve_pixwin:
        mask = np.equal(filt_map, 0)
        filt_map = enmap.unapply_window(filt_map)
        filt_map[mask] = 0
    if rms_map is None:
        rms_map = calc_rms_map(filt_map, grid_size_arcmin, apod_width=apod_width,
                               fix_rms_gw=fix_rms_gw, use_overlap_pix=use_overlap_pix,
                               niter=rms_niter, nsigma=rms_nsigma,
                               smooth=smooth_rms_map, smooth_pix=smooth_rms_pix)
    sn_map = calc_sn_map(filt_map, rms_map)
    return filt_map, sn_map, rms_map



class PointSourceFilter:
    def __init__(self, beam_fwhm, noise_map,
                 apod_width=0, apply_apod=False,
                 deconvolve_pixwin=True,
                 smooth_p2d_npix=fgi.p2d_smooth_npix,
                 rms_gw=fgi.rms_gw_sources, use_fixed_rms_gw=fgi.rms_fixed_gw, 
                 use_overlap_pix_for_rms=fgi.rms_use_overlap_pix,
                 rms_niter=fgi.rms_niter, rms_nsigma=fgi.rms_nsigma, 
                 smooth_rms=fgi.rms_smooth, smooth_rms_npix=fgi.rms_smooth_pix,
                 mask=None,
                ):
        '''
        beam_fwhm in arcmin
        apod_width in degrees
        rms_gw in arcmin
        '''
        self.shape = noise_map.shape
        self.wcs = noise_map.wcs
        self.mask = mask

        self.apodize = apply_apod
        self.apod_width = apod_width

        self.deconvolve_pixwin = deconvolve_pixwin

        self.rms_gw = rms_gw
        self.rms_niter = rms_niter
        self.rms_nsigma = rms_nsigma
        self.use_fixed_rms_gw = use_fixed_rms_gw
        self.use_overlap_pix = use_overlap_pix_for_rms
        self.smooth_rms = smooth_rms
        self.smooth_rms_npix = smooth_rms_npix

        # calculate the filter:
        ra_ctr, dec_ctr, _, _ = maps.get_map_ctr_extent(self.shape, self.wcs)
        beam_template = fgmaps.get_beam_template(self.shape, self.wcs, beam_fwhm, ra_ctr, dec_ctr)
        if self.apodize:
            apod_npix = fgmaps.dist2npix(self.apod_width, self.shape, self.wcs, units='degrees')
            noise_map = enmap.apod(noise_map.copy(), apod_npix)
        noise_p2d = calc_power2d(noise_map, smooth_npix=smooth_p2d_npix)
        self.filt = calc_filter(self.shape, self.wcs, noise_p2d, beam_template, deconvolve_pixwin=self.deconvolve_pixwin)


    def get_filter(self):
        return self.filt


    def apply_filter(self, imap):
        apod_width = self.apod_width if self.apodize else 0
        return apply_filter(imap, self.filt, apod_width=apod_width)


    def calc_filtered_maps(self, imap, apply_mask_to_sn_map=True):
        filtered_map = self.apply_filter(imap)
        if self.deconvolve_pixwin:
            mask = np.equal(filtered_map, 0)
            filtered_map = enmap.unapply_window(filtered_map)
            filtered_map[mask] = 0
        rms_map = calc_rms_map(filtered_map, self.rms_gw, apod_width=self.apod_width,
                               fix_rms_gw=self.use_fixed_rms_gw, use_overlap_pix=self.use_overlap_pix,
                               niter=self.rms_niter, nsigma=self.rms_nsigma,
                               smooth=self.smooth_rms, smooth_pix=self.smooth_rms_npix)
        sn_map = calc_sn_map(filtered_map, rms_map)
        if apply_mask_to_sn_map and (self.mask is not None):
            sn_map *= self.mask
        return filtered_map, sn_map, rms_map


# ----- clusters -----

def gauss_1d(r, sigma, amplitude=1):
    rprof = amplitude * np.exp(-0.5 * (r / sigma)**2)
    return rprof


def gauss_profile_name(sigma):
    return f'gauss{simutils.round_str(sigma)}arcmin'


def gauss_cluster_profiles_dict(sigmas, nsigma_for_rmax=5):
    profiles = {}
    for sigma in sigmas:
        profile_name = gauss_profile_name(sigma)
        profiles[profile_name] = {'rmax': nsigma_for_rmax * sigma, 'radial_prof_func': gauss_1d, 'args': [sigma], 'kwargs': {}}
    return profiles


def get_default_gauss_cluster_profiles_dict():
    default_cluster_profiles = gauss_cluster_profiles_dict(fgi.cluster_profile_sigmas)
    return default_cluster_profiles


def calc_power2d_multifreq(imaps, apod_width=0, smooth_npix=0, deconvolve_pixwin=False, verbose=False, log=None):
    """
    `imaps` should have shape `(nfreq, Ny, Nx)`
    """
    t0 = time.time()
    fgutils.print_msg('calculating multi-frequency 2d power', verbose=verbose, log=log)

    maps = imaps.copy()
    nfreq = maps.shape[0]
    map_shape = maps.shape[-2:]
    if deconvolve_pixwin:
        apod_npix = fgmaps.dist2npix(apod_width, map_shape, imaps.wcs, units='degrees')
        apod_window = enmap.apod(enmap.ones(map_shape, maps.wcs), apod_npix)
        for i in range(nfreq):
            maps[i] = enmap.unapply_window(maps[i] * apod_window)
        apod_width = 0 # don't apodize again

    p2d = enmap.ones((nfreq, nfreq, *map_shape), maps.wcs)
    for i in range(nfreq):
        for j in range(nfreq):
            p2d[i, j, :, :] = calc_power2d(maps[i], imap2=maps[j], apod_width=apod_width, smooth_npix=smooth_npix)
    fgutils.print_msg(f'{utils.tmsg(time.time() - t0)} to calculate 2d power', verbose=verbose, log=log)

    return p2d


def calc_inv_power2d_multifreq(p2d, verbose=False, log=None):
    t0 = time.time()
    fgutils.print_msg('calculating inverse of multi-frequency 2d power', verbose=verbose, log=log)
    # re-arrange shape from (nfreq, nfreq, Ny, Nx) to (Ny, Nx, nfreq, nfreq)
    tmp_p2d = np.moveaxis(p2d.copy(), [0, 1, 2, 3], [2, 3, 0, 1])
    # take inverse, then put back into correct shape
    inv_p2d = np.moveaxis(np.linalg.inv(tmp_p2d), [0, 1, 2, 3], [2, 3, 0, 1])
    fgutils.print_msg(f'{utils.tmsg(time.time() - t0)} to calculate inverse', verbose=verbose, log=log)
    return inv_p2d


def get_inv_power2d_multifreq(imaps, apod_width=0, smooth_npix=0, deconvolve_pixwin=False, verbose=False, log=None):
    p2d = calc_power2d_multifreq(imaps, apod_width=apod_width, smooth_npix=smooth_npix,
                                 deconvolve_pixwin=deconvolve_pixwin, verbose=verbose, log=log)
    inv_p2d = calc_inv_power2d_multifreq(p2d, verbose=verbose, log=log)
    return inv_p2d


def inv_noise_power2d_multifreq_block_fname(freq1, freq2, deconvolve_pixwin=False, output_dir=None):
    ''' NOTE : always returns the same filename for same freq pair (freq1 x freq2 same as freq2 x freq1) '''
    nu1 = min([freq1, freq2])
    nu2 = max([freq1, freq2])
    fname_root = f'inv_noise_power2d_multifreq_block_{round(nu1):03d}x{round(nu2):03d}'
    if deconvolve_pixwin:
        fname_root = f'{fname_root}_pixwin_deconvolved'
    fname = f'{fname_root}.fits'
    if output_dir is not None:
        fname = os.path.join(output_dir, fname)
    return fname


def save_inv_noise_power2d_multifreq(inv_noise_power2d, freqs, deconvolve_pixwin=False, output_dir=None, 
                                     overwrite=False, verbose=False, log=None):
    '''save blocks of the (symmetric) inv noise 2d power
    
    freqs should be in correct order (same as order in `inv_noise_power2d`)
    '''
    for i, freq1 in enumerate(freqs):
        for j, freq2 in enumerate(freqs):
            if j >= i:
                fname = inv_noise_power2d_multifreq_block_fname(freq1, freq2, deconvolve_pixwin=deconvolve_pixwin, output_dir=output_dir)
                if overwrite or (not os.path.exists(fname)):
                    enmap.write_map(fname, inv_noise_power2d[i, j])
                    fgutils.print_msg(f"saved {fname}", verbose=verbose, log=log)


def inv_noise_power2d_multifreq_is_saved(freqs, deconvolve_pixwin=False, output_dir=None):
    files_saved = []
    for i, freq1 in enumerate(freqs):
        for j, freq2 in enumerate(freqs):
            if j >= i:
                fname = inv_noise_power2d_multifreq_block_fname(freq1, freq2, deconvolve_pixwin=deconvolve_pixwin, output_dir=output_dir)
                files_saved.append(os.path.exists(fname))
    return all(files_saved)


def load_inv_noise_power2d_multifreq(shape, wcs, freqs, deconvolve_pixwin=False, output_dir=None):
    nfreq = len(freqs)
    inv_noise_power2d_shape = (nfreq, nfreq, *shape[-2:])
    inv_noise_power2d = enmap.zeros(inv_noise_power2d_shape, wcs)
    for i, freq1 in enumerate(freqs):
        for j, freq2 in enumerate(freqs):
            if j >= i:
                fname = inv_noise_power2d_multifreq_block_fname(freq1, freq2, deconvolve_pixwin=deconvolve_pixwin, output_dir=output_dir)
                inv_noise_power2d[i, j] = enmap.read_map(fname)
                if j != i:
                    inv_noise_power2d[j, i] = inv_noise_power2d[i, j].copy()
    return inv_noise_power2d


def get_inv_noise_power2d_multifreq(noise_maps, freqs, apod_width=0, smooth_npix=0, deconvolve_pixwin=False,
                                    output_dir=None, save=False, verbose=False, log=None):
    if inv_noise_power2d_multifreq_is_saved(freqs, deconvolve_pixwin=deconvolve_pixwin, output_dir=output_dir):
        inv_p2d = load_inv_noise_power2d_multifreq(noise_maps.shape[-2:], noise_maps.wcs, freqs,
                                                   deconvolve_pixwin=deconvolve_pixwin, output_dir=output_dir)
    else:
        inv_p2d = get_inv_power2d_multifreq(noise_maps, apod_width=apod_width, smooth_npix=smooth_npix,
                                            deconvolve_pixwin=deconvolve_pixwin, verbose=verbose, log=log)
        if save:
            save_inv_noise_power2d_multifreq(inv_p2d, freqs, deconvolve_pixwin=deconvolve_pixwin,
                                             output_dir=output_dir, verbose=verbose, log=log)
    return inv_p2d


def apply_filters_multifreq(imaps, filters, apod_width=0):
    '''
    filters should be a dict of filters
    '''
    fmaps = get_ft(imaps, apod_width=apod_width)
    filtered_maps = {}
    for key, filt in filters.items():
        filtered_maps[key] = get_inv_ft(fmaps * filt, normalize=False).sum(axis=0)
    return filtered_maps


def apply_filter_multifreq(imaps, filters, apod_width=0):
    fmaps = get_ft(imaps, apod_width=apod_width)
    filtered_map = get_inv_ft(fmaps * filters, normalize=False).sum(axis=0)
    return filtered_map


def calc_multifreq_filter_norm(freqs, unnormalized_filt, template_maps,
                               convolve_pixwin=False, convolve_beam=False, beam_fwhms=None,
                               deconvolve_pixwin=False, # whether we will deconvolve pixwin from filtered maps
                               verbose=False, log=None):
    '''
    deconvolve_pixwin (bool) : whether pixel window will be deconvolved from filtered maps
    '''
    maps_to_filter = template_maps.copy()
    y0 = 2e-4 
    for i, freq in enumerate(freqs):
        maps_to_filter[i] *= fgutils.y_to_uK(y0, freq)
        if convolve_pixwin:
            maps_to_filter[i] = enmap.apply_window(maps_to_filter[i])
        if convolve_beam:
            maps_to_filter[i] = maps.convolve_sim_with_beam(maps_to_filter[i], beam_fwhms[freq])
    filtered_map = apply_filter_multifreq(maps_to_filter, unnormalized_filt)
    if deconvolve_pixwin:
        filtered_map = enmap.unapply_window(filtered_map)
    filt_norm = y0 / np.max(filtered_map) 
    return filt_norm


def calc_multifreq_filter(freqs, shape, wcs, radial_prof_func, rmax, inv_noise_power2d,
                          convolve_pixwin=False, convolve_beam=False, beam_fwhms=None,
                          deconvolve_pixwin=False, verbose=False, log=None):
    """
    deconvolve_pixwin (bool) : whether pixel window will be deconvolved from filtered map

    note : we ask for `inv_noise_power2d` instead of the noise maps themselves, since `inv_noise_power2d` only needs to be calculated once & used for different filter
    """
    t0 = time.time()
    template_maps = fgmaps.get_profile_template_maps(freqs, shape, wcs, radial_prof_func, rmax,
                                              convolve_pixwin=convolve_pixwin,
                                              convolve_beam=convolve_beam, beam_fwhms=beam_fwhms)
    ftmaps = get_ft(template_maps)
    filt = enmap.zeros(template_maps.shape, template_maps.wcs)
    nfreq = len(freqs)
    for i in range(nfreq):
        for j, freq in enumerate(freqs):
            filt[i] += inv_noise_power2d[i,j] * fgutils.f_tSZ(freq) * abs(ftmaps[j]) 
    filt_norm = calc_multifreq_filter_norm(freqs, filt, template_maps, convolve_pixwin=(not convolve_pixwin),
                                           convolve_beam=(not convolve_beam), beam_fwhms=beam_fwhms,
                                           deconvolve_pixwin=deconvolve_pixwin, verbose=verbose, log=log)
    filt *= filt_norm
    fgutils.print_msg(f'{utils.tmsg(time.time() - t0)} to calculate filter for {freqs = }', verbose=verbose, log=log)
    return filt, filt_norm



class ClusterFilter:
    def __init__(self, freqs, beam_fwhms, noise_maps, radial_prof_func, rmax,
                 inv_noise_power2d=None, save_inv_noise_power2d=False, save_dir=None,
                 apod_width=0, apply_apod=False, deconvolve_pixwin=True, 
                 smooth_p2d_npix=fgi.p2d_smooth_npix,
                 rms_gw=fgi.rms_gw_clusters, use_fixed_rms_gw=fgi.rms_fixed_gw, 
                 use_overlap_pix_for_rms=fgi.rms_use_overlap_pix,
                 rms_niter=fgi.rms_niter, rms_nsigma=fgi.rms_nsigma, 
                 smooth_rms=fgi.rms_smooth, smooth_rms_npix=fgi.rms_smooth_pix,
                 mask=None, verbose=False, log=None):
        '''
        beam_fwhm in arcmin
        apod_width in degrees
        rms_gw in arcmin
        '''
        self.freqs = freqs.copy()
        self.nfreq = len(self.freqs)
        self.beam_fwhms = beam_fwhms
        self.noise_maps = noise_maps.copy()
        self.shape = noise_maps.shape[-2:]
        self.wcs = noise_maps.wcs
        self.apodize = apply_apod
        self.apod_width = apod_width
        self.deconvolve_pixwin = deconvolve_pixwin
        self.mask = mask

        self.radial_prof_func = radial_prof_func
        self.rmax = rmax

        self.rms_gw = rms_gw
        self.rms_niter = rms_niter
        self.rms_nsigma = rms_nsigma
        self.use_fixed_rms_gw = use_fixed_rms_gw
        self.use_overlap_pix = use_overlap_pix_for_rms
        self.smooth_rms = smooth_rms
        self.smooth_rms_npix = smooth_rms_npix

        self.inv_noise_power2d = inv_noise_power2d
        self.smooth_p2d_npix = smooth_p2d_npix
        self.save_inv_noise_power2d = save_inv_noise_power2d
        self.save_dir = save_dir

        self.filt = None
        self.filt_norm = None
        self.verbose = verbose
        self.log = log


    def calc_filter(self):
        if self.filt is None:
            if self.inv_noise_power2d is None:
                apod_width = self.apod_width if self.apodize else 0
                self.inv_noise_power2d = get_inv_noise_power2d_multifreq(self.noise_maps, self.freqs,
                                                                         apod_width=apod_width, 
                                                                         smooth_npix=self.smooth_p2d_npix,
                                                                         deconvolve_pixwin=self.deconvolve_pixwin,
                                                                         output_dir=self.save_dir, 
                                                                         save=self.save_inv_noise_power2d,
                                                                         verbose=self.verbose, log=self.log)
            self.filt, self.filt_norm = calc_multifreq_filter(self.freqs, self.shape, self.wcs,
                                                              self.radial_prof_func, self.rmax, self.inv_noise_power2d,
                                                              convolve_pixwin=(not self.deconvolve_pixwin),
                                                              convolve_beam=True, beam_fwhms=self.beam_fwhms,
                                                              deconvolve_pixwin=self.deconvolve_pixwin,
                                                              verbose=self.verbose, log=self.log)
        return self.filt, self.filt_norm


    def get_filter(self):
        self.calc_filter()
        return self.filt


    def get_filter_norm(self):
        self.calc_filter()
        return self.filt_norm


    def apply_filter(self, imaps):
        '''imaps should have shape (nfreq, ny, nx)'''
        apod_width = self.apod_width if self.apodize else 0
        return apply_filter_multifreq(imaps, self.get_filter(), apod_width=apod_width)


    def calc_rms_map(self, filtered_map):
        rms_map = calc_rms_map(filtered_map, self.rms_gw, apod_width=self.apod_width,
                               niter=self.rms_niter, nsigma=self.rms_nsigma,
                               fix_rms_gw=self.use_fixed_rms_gw, use_overlap_pix=self.use_overlap_pix,
                               smooth=self.smooth_rms, smooth_pix=self.smooth_rms_npix)
        return rms_map


    def calc_filtered_maps(self, imaps, apply_mask_to_sn_map=True):
        filtered_map = self.apply_filter(imaps)
        if self.deconvolve_pixwin:
            mask = np.equal(filtered_map, 0)
            filtered_map = enmap.unapply_window(filtered_map)
            filtered_map[mask] = 0
        rms_map = self.calc_rms_map(filtered_map)
        sn_map = calc_sn_map(filtered_map, rms_map)
        if apply_mask_to_sn_map and (self.mask is not None):
            sn_map *= self.mask
        return filtered_map, sn_map, rms_map




class ClusterFilters:
    def __init__(self, freqs, beam_fwhms, noise_maps, profiles,
                 inv_noise_power2d=None, save_inv_noise_power2d=False, save_dir=None,
                 apod_width=0, apply_apod=False,
                 deconvolve_pixwin=True,
                 smooth_p2d_npix=fgi.p2d_smooth_npix,
                 rms_gw=fgi.rms_gw_clusters, use_fixed_rms_gw=fgi.rms_fixed_gw, 
                 use_overlap_pix_for_rms=fgi.rms_use_overlap_pix,
                 rms_niter=fgi.rms_niter, rms_nsigma=fgi.rms_nsigma, 
                 smooth_rms=fgi.rms_smooth, smooth_rms_npix=fgi.rms_smooth_pix,
                 mask=None,
                 verbose=False, log=None,
                ):
        '''
        beam_fwhm in arcmin
        apod_width in degrees
        rms_gw in arcmin

        profiles is a dict of dicts:
            keys are profile names
            values are dicts w/ keys for radial_prof_func, rmax + optional keys for args and kwargs passed to profile func.
        '''
        self.profiles = profiles
        self.profile_names = list(profiles.keys())
        self.filter_kwargs = {'apod_width': apod_width, 'apply_apod': apply_apod,
                              'deconvolve_pixwin': deconvolve_pixwin, 'smooth_p2d_npix': smooth_p2d_npix,
                              'rms_gw': rms_gw, 'rms_niter': rms_niter, 'rms_nsigma': rms_nsigma,
                              'use_fixed_rms_gw': use_fixed_rms_gw, 'use_overlap_pix_for_rms': use_overlap_pix_for_rms,
                              'smooth_rms': smooth_rms, 'smooth_rms_npix': smooth_rms_npix,
                              'mask': mask, 'verbose': verbose, 'log': log}
        self.cluster_filters = {} # will hold instances of `ClusterFilter`

        self.freqs = freqs.copy()
        self.nfreq = len(self.freqs)
        self.beam_fwhms = beam_fwhms
        self.noise_maps = noise_maps.copy()
        self.shape = noise_maps.shape[-2:]
        self.wcs = noise_maps.wcs
        self.apodize = apply_apod
        self.apod_width = apod_width
        self.deconvolve_pixwin = deconvolve_pixwin

        self.inv_noise_power2d = inv_noise_power2d
        self.smooth_p2d_npix = smooth_p2d_npix
        self.save_inv_noise_power2d = save_inv_noise_power2d
        self.save_dir = save_dir

        self.verbose = verbose
        self.log = log


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


    def _validate_profile_name(self, profile_name):
        if profile_name not in self.profile_names:
            raise ValueError(f"`{profile_name = }` is not a key in the `profiles` dictionary passed during"
                             " initialization. You may use the `add_profile` method to add a new profile.")
        return profile_name


    def _validate_profile_names(self, profile_names):
        # make sure `profile_names` is a list instead of a single value:
        profile_names = [profile_name for profile_name in np.atleast_1d(profile_names)]
        invalid_names = [profile_name for profile_name in profile_names if (profile_name not in self.profile_names)]
        if len(invalid_names) > 0:
            raise ValueError(f"The profile names {invalid_names} in the list of `profile_names` are not keys in the"
                             " `profiles` dictionary passed during initialization. You may use the `add_profile`"
                             " method to add a new profile.")
        return profile_names



    def add_profile(self, profile_name, profile_func, rmax, profile_func_args=[], profile_func_kwargs={}):
        if profile_name not in self.profiles:
            self.profiles[profile_name] = {'radial_prof_func': profile_func, 'rmax': rmax, 'args': profile_func_args, 'kwargs': profile_func_kwargs}
            if profile_name not in self.profile_names:
                self.profile_names.append(profile_name)


    def get_inv_noise_power2d(self):
        if self.inv_noise_power2d is None:
            apod_width = self.apod_width if self.apodize else 0
            self.inv_noise_power2d = get_inv_noise_power2d_multifreq(self.noise_maps, self.freqs,
                                                                     apod_width=apod_width, smooth_npix=self.smooth_p2d_npix,
                                                                     deconvolve_pixwin=self.deconvolve_pixwin,
                                                                     output_dir=self.save_dir, save=self.save_inv_noise_power2d,
                                                                     verbose=self.verbose, log=self.log)
        return self.inv_noise_power2d


    def _init_filt(self, profile_name):
        profile_name = self._validate_profile_name(profile_name)
        if profile_name not in self.cluster_filters:
            profile_info = self.profiles[profile_name]
            if 'args' in profile_info:
                args = profile_info['args']
            else:
                args = []
            if 'kwargs' in profile_info:
                kwargs = profile_info['kwargs']
            else:
                kwargs = {}
            radial_prof_func = lambda r: profile_info['radial_prof_func'](r, *args, **kwargs)
            self.cluster_filters[profile_name] = ClusterFilter(self.freqs, self.beam_fwhms, self.noise_maps,
                                                               radial_prof_func, self.profiles[profile_name]['rmax'],
                                                               inv_noise_power2d=self.get_inv_noise_power2d(), **self.filter_kwargs)


    def calc_filter(self, profile_name):
        '''calculate the filter for a single profile'''
        self.infomsg(f"getting the filter for {profile_name = }")
        self._init_filt(profile_name)
        filt, filt_norm = self.cluster_filters[profile_name].calc_filter()
        return filt, filt_norm


    def calc_filters(self, profile_names=None):
        '''calculate the filter for multiple profiles'''
        profile_names = self.profile_names if (profile_names is None) else self._validate_profile_names(profile_names)
        filters = {}
        norms = {}
        for profile_name in profile_names:
            filters[profile_name], norms[profile_name] = self.calc_filter(profile_name)
        return filters, norms


    def get_filter(self, profile_name):
        self._init_filt(profile_name)
        if self.cluster_filters[profile_name].filt is None:
            self.infomsg(f"getting the filter for {profile_name = }")
        return self.cluster_filters[profile_name].get_filter()


    def get_filters(self, profile_names=None):
        profile_names = self.profile_names if (profile_names is None) else self._validate_profile_names(profile_names)
        filters = {}
        for profile_name in profile_names:
            filters[profile_name] = self.get_filter(profile_name)
        return filters


    def get_filter_norm(self, profile_name):
        self._init_filt(profile_name)
        return self.cluster_filters[profile_name].get_filter_norm()


    def get_filter_norms(self, profile_names=None):
        profile_names = self.profile_names if (profile_names is None) else self._validate_profile_names(profile_names)
        norms = {}
        for profile_name in profile_names:
            norms[profile_name] = self.get_filter_norm(profile_name)
        return norms


    def apply_filter(self, imaps, profile_name):
        '''imaps should have shape (nfreq, ny, nx)

        NOTE: this won't deconvolve pixwin from filtered map - use `calc_filtered_maps` for that
        '''
        self._init_filt(profile_name)
        return self.cluster_filters[profile_name].apply_filter(imaps)


    def calc_filtered_maps(self, imaps, profile_name, apply_mask_to_sn_map=True):
        '''filtered/SN/RMS maps for single profile'''
        self._init_filt(profile_name)
        filtered_map, sn_map, rms_map = self.cluster_filters[profile_name].calc_filtered_maps(imaps, apply_mask_to_sn_map=apply_mask_to_sn_map)
        return filtered_map, sn_map, rms_map


    def get_filtered_maps(self, imaps, profile_names=None, apply_mask_to_sn_map=True):
        '''filtered/SN/RMS maps for multiple profiles'''
        profile_names = self.profile_names if (profile_names is None) else self._validate_profile_names(profile_names)
        filters = {profile_name: self.get_filter(profile_name) for profile_name in profile_names}
        apod_width = self.apod_width if self.apodize else 0
        filtered_maps = apply_filters_multifreq(imaps, filters, apod_width=apod_width)
        sn_maps = {}
        rms_maps = {}
        for profile_name in profile_names:
            rms_maps[profile_name] = self.cluster_filters[profile_name].calc_rms_map(filtered_maps[profile_name])
            sn_maps[profile_name] = calc_sn_map(filtered_maps[profile_name], rms_maps[profile_name])
            if apply_mask_to_sn_map and (self.cluster_filters[profile_name].mask is not None):
                sn_maps[profile_name] *= self.cluster_filters[profile_name].mask
        return filtered_maps, sn_maps, rms_maps


# ----- noise maps for matched filters: -----

class NoiseSimsForFilters(hdsims.HDSims):
    def __init__(self, hd_sims_dir, freqs=fgi.freqs,
                 ra_ctr=fgi.noise_map_ra_ctr, dec_ctr=fgi.noise_map_dec_ctr,
                 width=fgi.noise_map_width, height=fgi.noise_map_height,
                 apod_width=fgi.noise_map_apod_width,
                 cmb_seed=fgi.noise_map_cmb_seed,
                 noise_seeds=fgi.noise_map_noise_seeds,
                 use_default_catalogs=True,
                 sources_to_subtract_catalog_files=None,
                 clusters_to_subtract_catalog_file=None,
                 cluster_profiles=None,
                 **kwargs):
        '''note: ra/dec/width/height/etc. should be for patch used to make noise map (not patch that is being fg cleaned)'''
        hdsims_kwargs = {'freqs': freqs, 'ra_ctr': ra_ctr, 'dec_ctr': dec_ctr, 'width': width, 'height': height,
                         'apod_width': apod_width, 'noise_seeds': noise_seeds, 'cmb_seed': cmb_seed, 'pol': False,
                         **kwargs, 'make_output_dirs': mpi.is_rank0}
        super().__init__(hd_sims_dir, **hdsims_kwargs)

        if use_default_catalogs and (sources_to_subtract_catalog_files is None):
            # catalog of point sources to subtract from maps at each frequency
            # (in order to only include residual sources in noise maps for tsz cluster filters)
            sources_to_subtract_catalog_files = self._default_source_catalog_files(**kwargs)
        if use_default_catalogs and (clusters_to_subtract_catalog_file is None):
            # catalog of clusters to subtract from maps
            # (in order to only include residual clusters in noise maps
            # for point source filters when making point source masks)
            clusters_to_subtract_catalog_file = self._default_cluster_catalog_file()
            if clusters_to_subtract_catalog_file is not None:
                # also need to use default set of cluster profiles
                cluster_profiles = get_default_gauss_cluster_profiles_dict()

        if sources_to_subtract_catalog_files is not None:
            self.source_catalogs = {freq: fgcatalogs.load_catalog(sources_to_subtract_catalog_files[freq]) for freq in self.freqs}
        else:
            self.source_catalogs = None
        if clusters_to_subtract_catalog_file is not None:
            self.cluster_catalog = fgcatalogs.load_catalog(clusters_to_subtract_catalog_file)
        else:
            self.cluster_catalog = None
        self.cluster_profiles = cluster_profiles
        if self.cluster_profiles is None:
            self.cluster_profiles = get_default_gauss_cluster_profiles_dict()

        # make sure all map components have been generated and saved:
        if mpi.is_rank0:
            for component in self.map_components:
                map_freqs = self.freqs if (component in ['tsz', 'cib', 'radio']) else [None]
                for freq in map_freqs:
                    sim_is_saved = os.path.exists(self.get_signal_sim_fname(component, freq=freq))
                    if component == 'cmb': # also check if combined TQU is saved
                        tqu_is_saved = os.path.exists(self.get_signal_sim_fname(component, pol=True))
                        sim_is_saved = sim_is_saved or tqu_is_saved
                    if not sim_is_saved:
                        self.get_signal_sim(component, freq=freq)
        mpi.comm.barrier()


    def _default_source_catalog_files(self, **kwargs):
        # check if we can actually use the catalogs:
        same_ra_ctr = np.isclose(self.ra_ctr, fgi.noise_map_ra_ctr)
        same_dec_ctr = np.isclose(self.dec_ctr, fgi.noise_map_dec_ctr)
        same_res = np.isclose(self.res, si.hd_res)
        same_ctr_res = same_ra_ctr and same_dec_ctr and same_res
        # can only use pre-computed catalogs for certain frequencies:
        same_freqs = not any([(freq not in fgi.freqs) for freq in self.freqs])
        # due to the way the CIB maps/catalogs are generated, the size of the noise maps
        # needs to exactly match the size of the maps used to calculate the catalogs of
        # sources to subtract from the noise maps:
        width_to_use = None
        for width in sorted(fgi.noise_map_widths)[::-1]: # check largest first
            same_width = np.isclose(self.width, width)
            same_height = np.isclose(self.height, width)
            same_apod = np.isclose(self.apod_width, fgi.noise_map_apod_widths[width])
            if same_width and same_height and same_apod:
                width_to_use = width
        # we also need to be using the same (default/baseline) CIB model:
        same_cib = ('cib_model' not in kwargs)
        if same_ctr_res and same_freqs and same_cib and (width_to_use is not None):
            return fgi.noise_map_source_catalog_files[width_to_use]


    def _default_cluster_catalog_file(self):
        # check if we can actually use the catalogs:
        same_ra_ctr = np.isclose(self.ra_ctr, fgi.noise_map_ra_ctr)
        same_dec_ctr = np.isclose(self.dec_ctr, fgi.noise_map_dec_ctr)
        same_res = np.isclose(self.res, si.hd_res)
        same_ctr_res = same_ra_ctr and same_dec_ctr and same_res
        # check if size of noise maps is same or smaller than region in saved cluster catalog:
        width_to_use = None
        # compare rounded values
        w = round(self.width, 4)
        h = round(self.height, 4)
        for width in sorted(fgi.noise_map_widths)[::-1]: # check largest first
            if (w <= round(width, 4)) and (h <= round(width, 4)):
                width_to_use = width
        if same_ctr_res and (width_to_use is not None):
            return fgi.noise_map_cluster_catalog_files[width_to_use]
        

    def _check_geometry_is_compatible(self, shape, wcs):
        if not maps.map_resolution_is_equal(shape, wcs, self.shape, self.wcs):
            res = maps.get_map_resolution(shape, wcs)
            raise ValueError("The resolution of the requested map geometry (`shape` and `wcs`) is "
                             f"{round(res,6)} arcminutes, which differs from the noise map "
                             f"resolution of {round(self.res,6)} arcminutes.")
        if (shape[-2] > self.padded_shape[-2]) or (shape[-1] > self.padded_shape[-1]):
            raise ValueError(f"`{shape = }`: The requested map shape is larger than the maximum available shape"
                             f" of `{self.padded_shape}` for a {simutils.round_str(self.padded_width)} degree x"
                             f" {simutils.round_str(self.padded_height)} map.")


    def _noise_map_geometry(self, shape, wcs):
        '''
        shape, wcs is geometry of map that is being filtered/fg cleaned
        returns shape, wcs for map of same area as input geometry, but centered at ctr. of noise map patch
          (i.e., geometry used to get cut out from noise map)
        '''
        self._check_geometry_is_compatible(shape, wcs)
        _, _, width, height = maps.get_map_ctr_extent(shape, wcs)
        noise_map_shape, noise_map_wcs = maps.get_shape_wcs(self.res, self.ra_ctr, self.dec_ctr, width, height=height)
        return noise_map_shape, noise_map_wcs
    
    
    def _noise_map_apod_window(self, shape, wcs):
        _, noise_map_wcs = self._noise_map_geometry(shape, wcs)
        window = self.get_apod_window(shape=shape, wcs=noise_map_wcs, apod_width=self.map_apod_width)
        return window
    
    
    def noise_maps_for_source_filters(self, shape, wcs, freqs=None, apod=True):
        freqs = self.freqs if (freqs is None) else simutils.validate_sim_freqs(freqs)
        _, noise_map_wcs = self._noise_map_geometry(shape, wcs)
        window = 1 if (not apod) else self._noise_map_apod_window(shape, wcs)
        noise_map_components = [c for c in self.map_components if (c not in ['cib', 'radio'])]
        noise_maps = {}
        for freq in freqs:
            if len(noise_map_components) > 0:
                noise_map = self.get_sim(freq=freq, components=noise_map_components, beam=True, noise=True, shape=shape, wcs=noise_map_wcs)
            else:
                noise_map = self.get_noise_sim(freq, shape=shape, wcs=noise_map_wcs) 
            noise_maps[freq] = enmap.enmap(noise_map[:], wcs) * window
        return noise_maps
    
    
    def noise_maps_for_cluster_filters(self, shape, wcs, subtract_sources=True, freqs=None, apod=True):
        freqs = self.freqs if (freqs is None) else simutils.validate_sim_freqs(freqs)
        _, noise_map_wcs = self._noise_map_geometry(shape, wcs)
        window = 1 if (not apod) else self._noise_map_apod_window(shape, wcs)
        noise_map_components = [c for c in self.map_components if (c != 'tsz')]
        can_subtract_sources = ('cib' in self.map_components) and ('radio' in self.map_components) and (self.source_catalogs is not None)
        if subtract_sources and can_subtract_sources:
            sources_to_subtract = {}
            for freq in freqs:
                sources_to_subtract[freq] = maps.make_src_map(self.padded_shape, self.padded_wcs, [self.source_catalogs[freq]], freq)
                sources_to_subtract[freq] = maps.convolve_sim_with_beam(enmap.apply_window(sources_to_subtract[freq]), si.beam_fwhm[freq])
                sources_to_subtract[freq] = enmap.project(sources_to_subtract[freq], shape, noise_map_wcs)
        noise_maps = {}
        for freq in freqs:
            if len(noise_map_components) > 0:
                noise_map = self.get_sim(freq=freq, components=noise_map_components, beam=True, noise=True, shape=shape, wcs=noise_map_wcs)
                if subtract_sources and can_subtract_sources:
                    noise_map -= sources_to_subtract[freq]
            else:
                noise_map = self.get_noise_sim(freq, shape=shape, wcs=noise_map_wcs) 
            noise_maps[freq] = enmap.enmap(noise_map[:], wcs) * window
        return noise_maps
    
    
    def noise_maps_for_source_mask_filters(self, shape, wcs, subtract_clusters=True, freqs=None, apod=True):
        freqs = self.freqs if (freqs is None) else simutils.validate_sim_freqs(freqs)
        _, noise_map_wcs = self._noise_map_geometry(shape, wcs)
        window = 1 if (not apod) else self._noise_map_apod_window(shape, wcs)
        noise_map_components = [c for c in self.map_components if (c not in ['cib', 'radio'])]
        can_subtract_clusters = (self.cluster_catalog is not None) and ('tsz' in self.map_components)
        if subtract_clusters and can_subtract_clusters:
            beam_fwhms = {freq: si.beam_fwhm[freq] for freq in freqs}
            clusters_to_subtract = fgmaps.make_cluster_sims_from_catalog(self.padded_shape, self.padded_wcs, 
                                                                  self.cluster_catalog, self.cluster_profiles, freqs,
                                                                  convolve_pixwin=True, convolve_beam=True, beam_fwhms=beam_fwhms)
            for freq in freqs:
                clusters_to_subtract[freq] = enmap.project(clusters_to_subtract[freq], shape, noise_map_wcs)
        noise_maps = {}
        for freq in freqs:
            if len(noise_map_components) > 0:
                noise_map = self.get_sim(freq=freq, components=noise_map_components, beam=True, noise=True, shape=shape, wcs=noise_map_wcs)
                if subtract_clusters and can_subtract_clusters:
                    noise_map -= clusters_to_subtract[freq]
            else:
                noise_map = self.get_noise_sim(freq, shape=shape, wcs=noise_map_wcs) 
            noise_maps[freq] = enmap.enmap(noise_map[:], wcs) * window
        return noise_maps

