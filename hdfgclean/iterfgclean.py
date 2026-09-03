import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import interp1d
from pixell import enmap
from hdsims import siminfo as si, utils, simutils, fgcatalogs, maps
from . import fgutils, fgmaps


def locate_objs(imap, threshold):
    threshold_mask = np.greater(imap, threshold)
    obj_map, num_objs = ndimage.label(threshold_mask)
    obj_ids = np.unique(obj_map)
    obj_positions = ndimage.maximum_position(imap, labels=obj_map, index=obj_ids)
    obj_npix = ndimage.sum(threshold_mask, labels=obj_map, index=obj_ids)
    return obj_ids, obj_positions, obj_npix, obj_map


def find_objs(sn_map, snr_threshold=5, snr_threshold_below=None, use_abs_sn=False, min_npix=1):
    # first check if there are any pixels that exceed the snr threshold:
    if use_abs_sn:
        map_has_objs = np.any(np.abs(sn_map) >= snr_threshold)
    else:
        map_has_objs = np.any(sn_map >= snr_threshold)
        if snr_threshold_below is not None:
            map_has_objs = map_has_objs or np.any(sn_map <= snr_threshold_below)

    if map_has_objs:
        # find locations in the S/N map above the given minimum SNR:
        if use_abs_sn:
            obj_ids, obj_positions, obj_num_pix, obj_map = locate_objs(np.abs(sn_map), snr_threshold)
        else:
            obj_ids, obj_positions, obj_num_pix, obj_map = locate_objs(sn_map, snr_threshold)
            if snr_threshold_below is not None:
                obj_ids2, obj_positions2, obj_num_pix2, obj_map2 = locate_objs(-sn_map, -snr_threshold_below)
                if len(obj_ids2[1:]) > 0:
                    max_obj_id = np.max(obj_ids)
                    obj_ids2[obj_ids2 > 0] += max_obj_id
                    obj_map2[obj_map2 > 0] += max_obj_id
                    obj_ids = np.concatenate([obj_ids, obj_ids2[1:]])
                    obj_positions = np.concatenate([obj_positions, obj_positions2[1:]])
                    obj_num_pix = np.concatenate([obj_num_pix, obj_num_pix2[1:]])
                    obj_map += obj_map2
        obj_ids = obj_ids[1:]
        obj_positions = obj_positions[1:] #if (len(obj_ids) > 0) else np.array([])
        obj_num_pix = obj_num_pix[1:].astype(int)
        if (min_npix > 1) and (len(obj_ids) > 0):
            loc = np.where(obj_num_pix >= min_npix)[0].astype(int)
            obj_ids = obj_ids[loc]
            obj_positions = np.array(obj_positions)[loc]
            obj_num_pix = obj_num_pix[loc]

    else:
        obj_ids = np.array([], dtype=int)
        obj_positions = []
        obj_num_pix = np.array([], dtype=int)
        obj_map = np.zeros(sn_map.shape, dtype=int)

    return obj_ids, obj_positions, obj_num_pix, obj_map


# ---------- sources ----------

def calc_beam_solid_angle_per_dec(beam_fwhm, delta_dec, shape, wcs):
    delta_dec_deg = utils.arcmin2deg(delta_dec)
    _, _, dec_min, dec_max = maps.get_map_corner_coords(shape, wcs)
    ra, _, _, _ = maps.get_map_ctr_extent(shape, wcs)
    decs = np.arange(dec_min + delta_dec_deg / 2, dec_max + delta_dec_deg / 2, delta_dec_deg)
    x_pixels, y_pixels = maps.get_pixel_positions([ra]*len(decs), decs, shape, wcs)
    # make sure all pixels are actually in the map:
    if y_pixels[0] < 0:
        decs = decs[1:]
        x_pixels = x_pixels[1:]
        y_pixels = y_pixels[1:]
    if y_pixels[-1] >= shape[0]:
        decs = decs[:-1]
        x_pixels = x_pixels[:-1]
        y_pixels = y_pixels[:-1]
    # calculate beam solid angle at each dec:
    pixsizemap = enmap.pixsizemap(shape, wcs)
    beam_solid_angles = np.zeros(len(decs))
    for i, (dec, x, y) in enumerate(zip(decs, x_pixels, y_pixels)):
        beam_map = fgmaps.get_beam_template(shape, wcs, beam_fwhm, ra, dec, normalize=False)
        beam_solid_angles[i] = pixsizemap[y,x] / beam_map[y,x]
    # make sure we can interpolate over entire range from `dec_min` to `dec_max`:
    decs = [dec_min, *decs, dec_max]
    beam_solid_angles = [beam_solid_angles[0], *beam_solid_angles, beam_solid_angles[-1]]
    return decs, beam_solid_angles


def get_beam_solid_angle_per_dec_fname(beam_fwhm, delta_dec, shape, wcs, save_dir=None):
    beam_info = f'beam{simutils.round_str(beam_fwhm)}arcmin'
    _, _, dec_min, dec_max = maps.get_map_corner_coords(shape, wcs)
    dec_info = f'{simutils.round_str(dec_min)}to{simutils.round_str(dec_max)}deg_every{simutils.round_str(delta_dec)}arcmin'
    fname = f'{beam_info}_solid_angle_vs_dec_{dec_info}.txt'
    if save_dir is not None:
        fname = os.path.join(save_dir, fname)
    return fname
    
    
def get_beam_solid_angle_per_dec(beam_fwhm, delta_dec, shape, wcs, save=True, save_dir=None, verbose=False, log=None):
    fname = get_beam_solid_angle_per_dec_fname(beam_fwhm, delta_dec, shape, wcs, save_dir=save_dir)
    if os.path.exists(fname):
        decs, beam_solid_angles = np.loadtxt(fname, unpack=True)
    else:
        fgutils.print_msg(("Calculating beam solid angle as a function of declination "
                           f"for a {simutils.round_str(beam_fwhm)} arcminute beam"), verbose=verbose, log=log)
        t = time.time()
        decs, beam_solid_angles = calc_beam_solid_angle_per_dec(beam_fwhm, delta_dec, shape, wcs)
        if save:
            np.savetxt(fname, np.column_stack([decs, beam_solid_angles]), 
                       header='dec [degrees], beam solid angle [steradians]')
        fgutils.print_msg(f"{utils.tmsg(time.time() - t)} to calculate beam solid angle function", 
                          verbose=verbose, log=log)
    return decs, beam_solid_angles


def get_beam_solid_angle_per_dec_func(beam_fwhm, delta_dec, shape, wcs, save=True, save_dir=None, verbose=False, log=None):
    decs, beam_solid_angles = get_beam_solid_angle_per_dec(beam_fwhm, delta_dec, shape, wcs, save=save, 
                                                           save_dir=save_dir, verbose=verbose, log=log)
    return interp1d(decs, beam_solid_angles)


def get_const_beam_solid_angle_func(beam_fwhm):
    beam_solid_angle = fgutils.beam_fwhm_to_solid_angle(beam_fwhm)
    beam_solid_angle_func = lambda dec : beam_solid_angle
    return beam_solid_angle_func


def iter_sources_catalog_fname(iter_num, snr_threshold, iter_catalog_dir=None, extrap=False):
    if extrap:
        fname = f'iter{iter_num:03d}_extrap.csv'
    else:
        fname = f'iter{iter_num:03d}_SNRabove{simutils.round_str(snr_threshold)}.csv'
    if iter_catalog_dir is not None:
        fname = os.path.join(iter_catalog_dir, fname)
    return fname


def get_max_iter_num(iter_catalog_dir):
    """NOTE: returns `max_inum = 0` if nothing saved; then `next_inum = max_inum+1` is one"""
    max_inum = 0
    for fname in os.listdir(iter_catalog_dir):
        if ('iter' in fname) and fname.endswith('csv'):
            iter_num_info = fname.split('_')[0]
            iter_num = int(iter_num_info.strip('iter'))
            max_inum = max([iter_num, max_inum])
    return max_inum


def get_max_source_idx(iter_catalog_dir):
    """NOTE: returns `max_idx = -1` if nothing saved; then `next_inum = max_inum+1` is zero"""
    max_inum = get_max_iter_num(iter_catalog_dir)
    if max_inum > 0:
        max_iter_fname_root = f'iter{max_inum:03d}'
        for fname in os.listdir(iter_catalog_dir):
            if fname.startswith(max_iter_fname_root) and fname.endswith('csv'):
                iter_catalog = fgcatalogs.load_catalog(os.path.join(iter_catalog_dir, fname))
                max_idx = np.max(iter_catalog['idx'].values)
    else:
        max_idx = -1
    return max_idx


def measure_sources_above_snr(filtered_map, sn_map, snr_threshold, freq, 
                              beam_fwhm=None, beam_solid_angle_func=None, 
                              measured_positions_catalog=None, nearby_source_radius=None,
                              min_npix=1):
    # get the beam solid angle (either constant or a function of declination):
    if beam_solid_angle_func is None:
        if beam_fwhm is None: 
            if int(freq) in si.freqs:
                beam_fwhm = si.beam_fwhm[int(freq)]
                msg = (f"Using the CMB-HD {int(freq)} GHz beam full-width at half-maximum of "
                       f"{simutils.round_str(beam_fwhm)} arcminutes to calculate flux")
                warnings.warn(msg)
            else:
                errmsg = ("You must pass `beam_fwhm`, the full-width at half-maximum (in arcminutes) "
                          f"of the {freq} GHz Gaussian beam, to calculate flux.")
                raise ValueError(errmsg)
        beam_solid_angle_func = lambda dec: 2 * np.pi * utils.fwhm2sigma(utils.arcmin2rad(beam_fwhm))**2

    obj_ids, obj_positions, num_pix, segmentation_map = find_objs(sn_map, snr_threshold=snr_threshold, 
                                                                  min_npix=min_npix)
    if len(obj_ids) > 0:
        y_pixels, x_pixels = np.array(obj_positions).T
        ras, decs = maps.get_coord_positions(x_pixels, y_pixels, sn_map.shape, sn_map.wcs)
        methods = ['map'] * len(obj_ids)
        
        if measured_positions_catalog is not None:
            # make a 'map' of the integer IDs (`idx`s) in the `measured_positions_catalog` at the measured positions:
            idx_map = np.zeros(sn_map.shape, dtype=int)
            idx_map[measured_positions_catalog['y_pixel'].values, measured_positions_catalog['x_pixel'].values] = measured_positions_catalog['idx'].values + 1
            idx_map = enmap.ndmap(idx_map, sn_map.wcs)
            # keep track of the `idxs` in the `measured_positions_catalog` where we measure a source at this freq:
            cat_idxs = np.zeros(len(obj_ids), dtype=int) - 1
            
            # make a mask with holes centered at measured positions in the sn/filtered map
            if nearby_source_radius is None:
                # determine mask radius from the beam size (use the standard deviation of the Gaussian beam)
                if beam_fwhm is None:
                    if int(freq) in si.freqs:
                        beam_fwhm = si.beam_fwhm[int(freq)]
                    else:
                        # measure the beam FWHM using the beam solid angle function
                        _, dec_ctr, _, _ = maps.get_map_ctr_extent(sn_map.shape, sn_map.wcs)
                        beam_solid_angle = beam_solid_angle_func(dec_ctr)
                        beam_sigma = utils.rad2arcmin(np.sqrt(beam_solid_angle / (2 * np.pi)))
                        beam_fwhm = utils.sigma2fwhm(beam_sigma)
                nearby_source_radius = utils.fwhm2sigma(beam_fwhm)
            nearby_src_mask = np.less(fgmaps.make_binary_mask(ras, decs, nearby_source_radius, sn_map.shape, sn_map.wcs), 1)
            # make a mask of only catalog positions near the measured positions and above SNR threshold
            catalog_mask = np.greater(idx_map, 0)  * nearby_src_mask
            for i, obj_id in enumerate(obj_ids):
                # if location of max SNR is close to a previously-measured 
                # (at another freq) position, use that position instead:
                cat_src_mask = np.equal(segmentation_map, obj_id) * catalog_mask
                if np.any(sn_map*cat_src_mask > 0): 
                    y_pixels[i], x_pixels[i] = enmap.argmax(sn_map*cat_src_mask, unit='pix')
                    ras[i], decs[i] = maps.pix2coord(x_pixels[i], y_pixels[i], sn_map.shape, sn_map.wcs)
                    methods[i] = 'catalog'
                    cat_idxs[i] = int(idx_map[y_pixels[i], x_pixels[i]] - 1)
        
        snrs = sn_map[y_pixels, x_pixels]
        Tmaxs = filtered_map[y_pixels, x_pixels]
        fluxs = utils.uK_to_mJy_per_str(Tmaxs, freq) * beam_solid_angle_func(np.round(decs, 6))
    
    else:
        x_pixels = []
        y_pixels = []
        ras = []
        decs = []
        fluxs = []
        snrs = []
        Tmaxs = []
        methods = []
        cat_idxs = []

    ocat = pd.DataFrame({'RADeg': ras, 'decDeg': decs, 'SNR': snrs, 'numSigPix': num_pix, 
                         'deltaT_c': Tmaxs, 'fluxmJy': fluxs, 'method': methods})
    if measured_positions_catalog is not None:
        ocat['catalog_idx'] = cat_idxs
    ocat = ocat[ocat['SNR'].ge(snr_threshold)]
    ocat['snr_threshold'] = snr_threshold
    ocat = ocat.sort_values('SNR', ascending=False)
    ocat = ocat.reset_index(drop=True)
    return ocat


def find_sources_in_shared_pixels(catalog, shape, wcs, ra_key='RADeg', dec_key='decDeg'):
    catalog = maps.add_pixel_coords_to_catalog(catalog, shape, wcs, add_pixel_num=True, ra_key=ra_key, dec_key=dec_key)
    return catalog[catalog.duplicated(subset=['pixel_num'], keep=False)].copy()


def sum_sources_in_shared_pixels(catalog, shape, wcs, key_to_use='idx', sort_key_to_use_ascending=True, 
                                            keys_to_sum=['fluxmJy', 'deltaT_c'], sum_snr=True,
                                            ra_key='RADeg', dec_key='decDeg', idx_key='idx', snr_key='SNR',):
    """returns a catalog with one source per pixel, 
    only containing sources at same pixel location measured multiple times 
    (as opposed to all measured sources in the catalog)
    """
    srcs_in_shared_pixels = find_sources_in_shared_pixels(catalog, shape, wcs, ra_key=ra_key, dec_key=dec_key)
    idxs_to_keep = []
    for _, src in srcs_in_shared_pixels.iterrows():
        pixel_num = int(src['pixel_num'])
        srcs_in_pixel = srcs_in_shared_pixels[srcs_in_shared_pixels['pixel_num'].eq(pixel_num)]
        
        if ('method' in srcs_in_pixel.columns.values) and ('catalog' in srcs_in_pixel['method'].values):
            # keep row that was measured using catalog at another freq
            idx_to_keep = srcs_in_pixel[srcs_in_pixel['method'].eq('catalog')][idx_key].values[0]
        else:    
            # keep row in catalog according to value in the `key_to_use` col:
            srcs_in_pixel = srcs_in_pixel.sort_values(key_to_use, ascending=sort_key_to_use_ascending)
            idx_to_keep = srcs_in_pixel[idx_key].values[0]
        
        if idx_to_keep not in idxs_to_keep:
            src_index = srcs_in_pixel.index.values[0]
            # update the catalog, setting the values in the `keys_to_sum` columns to be the sum of the measurements in this pixel:
            for key in keys_to_sum:
                if key in srcs_in_shared_pixels.columns.values:
                    srcs_in_shared_pixels.loc[src_index, key] = np.sum(srcs_in_pixel[key].values)
            if sum_snr: # also sum the snr values, in quadrature
                srcs_in_shared_pixels.loc[src_index, snr_key] = np.sqrt(np.sum(srcs_in_pixel[snr_key].values**2))
            idxs_to_keep.append(idx_to_keep)
    combined_srcs_in_shared_pixels = srcs_in_shared_pixels[srcs_in_shared_pixels[idx_key].isin(idxs_to_keep)]
    return combined_srcs_in_shared_pixels


def combine_sources_in_shared_pixels(catalog, shape, wcs, 
                                       key_to_use='idx', sort_key_to_use_ascending=True, 
                                       keys_to_sum=['fluxmJy',  'deltaT_c'], sum_snr=True,
                                       ra_key='RADeg', dec_key='decDeg', idx_key='idx', snr_key='SNR', 
                                       sort_output_catalog_by_col=None, sort_output_col_ascending=True, 
                                       reset_index=True, drop_old_index=True, reset_idx=False):
    """returns a catalog with one source per pixel, containing all measured sources in the catalog"""
    cols = catalog.columns
    srcs_in_shared_pixels = find_sources_in_shared_pixels(catalog, shape, wcs, ra_key=ra_key, dec_key=dec_key)
    if len(srcs_in_shared_pixels) > 0:
        combined_srcs_in_shared_pixels = sum_sources_in_shared_pixels(catalog, shape, wcs, key_to_use=key_to_use, 
                                                                                 sort_key_to_use_ascending=sort_key_to_use_ascending,
                                                                                 keys_to_sum=keys_to_sum, sum_snr=sum_snr,
                                                                                 ra_key=ra_key, dec_key=dec_key, idx_key=idx_key, snr_key=snr_key)
        ocat = fgcatalogs.combine_catalogs([catalog[~catalog[idx_key].isin(srcs_in_shared_pixels[idx_key].values)].copy(), combined_srcs_in_shared_pixels])
        if sort_output_catalog_by_col is not None:
            ocat = ocat.sort_values(sort_output_catalog_by_col, ascending=sort_output_col_ascending)
        if reset_index:
            ocat = ocat.reset_index(drop=drop_old_index)
        if reset_idx:
            ocat[idx_key] = list(range(len(ocat)))
    else:
        ocat = catalog
    return ocat[cols]


def iteratively_measure_sources(imap, freq, beam_fwhm, filt_obj, snr_threshold_list, iter_catalog_dir,
                                beam_solid_angle_func=None, min_num_iter_sources_per_snr=10,
                                initial_iter_num=1, initial_source_id=0, remeasured=False,
                                measured_positions_catalog=None, nearby_source_radius=None,
                                log=None, verbose=True):
    t = time.time()
    iter_catalogs = {}
    num_measured_srcs = 0
    inum = initial_iter_num
    inums = []
    sub_sim = imap.copy() # map after subtracting measured sources from prev. iters; will be the input sim on next iter
    measured_srcs_map = enmap.zeros(imap.shape, imap.wcs)
    if measured_positions_catalog is not None:
        unmeasured_srcs_cat = measured_positions_catalog.copy()
    else:
        unmeasured_srcs_cat = None
    
    for snr_threshold in snr_threshold_list:
        # keep iterating at this snr threshold until less than `min_num_srcs` found
        min_num_srcs = min_num_iter_sources_per_snr if (snr_threshold > min(snr_threshold_list)) else 1
        # initialize number of sources found for this snr threshold
        num_iter_srcs = min_num_srcs + 1
        while num_iter_srcs >= min_num_srcs:
            t0 = time.time()
            iter_num = inum # just used to print out info about this iter.
            iter_cat_fname = iter_sources_catalog_fname(inum, snr_threshold, iter_catalog_dir=iter_catalog_dir)
            if os.path.exists(iter_cat_fname):
                iter_cat = fgcatalogs.load_catalog(iter_cat_fname)
            else:
                # get the filtered/SN/RMS maps
                iter_filt_map, iter_sn_map, iter_rms_map = filt_obj.calc_filtered_maps(sub_sim)
                # get the catalog
                iter_cat = measure_sources_above_snr(iter_filt_map, iter_sn_map, snr_threshold, freq, 
                                                     beam_fwhm=beam_fwhm, beam_solid_angle_func=beam_solid_angle_func,
                                                     measured_positions_catalog=unmeasured_srcs_cat, 
                                                     nearby_source_radius=nearby_source_radius)
                iter_cat['iter_num'] = inum
                iter_cat['idx'] = list(range(num_measured_srcs + initial_source_id, 
                                             num_measured_srcs + len(iter_cat) + initial_source_id))
            num_iter_srcs = len(iter_cat)
            num_measured_srcs += num_iter_srcs
            if num_iter_srcs > 0:
                if not os.path.exists(iter_cat_fname):
                    iter_cat.to_csv(iter_cat_fname)
                # subtract measured sources from this iter 
                iter_srcs_map = maps.make_src_map(imap.shape, imap.wcs, [iter_cat], freq)
                measured_srcs_map += iter_srcs_map.copy()
                sub_sim -= maps.convolve_sim_with_beam(enmap.apply_window(iter_srcs_map), beam_fwhm)
                iter_catalogs[inum] = iter_cat.copy()
                if unmeasured_srcs_cat is not None:
                    unmeasured_srcs_cat = unmeasured_srcs_cat[~unmeasured_srcs_cat['idx'].isin(iter_cat['catalog_idx'].values)]
                inums.append(inum)
                inum += 1
            fgutils.print_msg((f"{utils.tmsg(time.time() - t0)} for iter {iter_num}: found {num_iter_srcs} with "
                               f"SNR >= {snr_threshold} ({num_measured_srcs} total so far)"), verbose=verbose, log=log)
    
    if num_measured_srcs > 0:
        catalog = fgcatalogs.combine_catalogs([iter_catalogs[inum] for inum in inums])
        catalog['remeasured'] = remeasured
    else:
        catalog = pd.DataFrame({col: [] for col in [*iter_cat.columns.values, 'iter_num', 'idx', 'remeasured']})
    # make sure that there is, at most, one source per pixel:
    srcs_in_shared_pixels = find_sources_in_shared_pixels(catalog, imap.shape, imap.wcs)
    if len(srcs_in_shared_pixels) > 0:
        catalog = combine_sources_in_shared_pixels(catalog, imap.shape, imap.wcs, sort_output_catalog_by_col='idx')
    fgutils.print_msg(f'{utils.tmsg(time.time() - t)} to measure {len(catalog)} sources', verbose=verbose, log=log)
    
    return catalog, sub_sim, measured_srcs_map


def find_missubtracted_sources(measured_srcs, sn_map, snr_threshold=5, min_npix=1, nearby_sources_radius=0,
                                 use_abs_sn=True, snr_threshold_below=None):
    # find locations w/ SNR above/below threshold
    objIDs, objPositions, objNumPix, segmentation_map = find_objs(sn_map, snr_threshold=snr_threshold, 
                                                                  snr_threshold_below=snr_threshold_below, 
                                                                  use_abs_sn=use_abs_sn, min_npix=min_npix)
    
    # get the catalog of measured sources
    measured_cat = measured_srcs.copy()
    # get pixel position of each source
    measured_cat = maps.add_pixel_coords_to_catalog(measured_cat, sn_map.shape, sn_map.wcs)
    # find sources that fall into the same pixel
    measured_srcs_in_shared_pixels = find_sources_in_shared_pixels(measured_cat.copy(), sn_map.shape, sn_map.wcs)
    
    # make a map of the source `'idx'` values
    # (if multiple sources fall into same pixel, the value of that pixel in 
    #  the `src_idx_map` will only have one `idx` ; we will correct for this below)
    src_idx_map = enmap.zeros(sn_map.shape, sn_map.wcs, dtype=int)
    x_pixels = measured_cat['x_pixel'].values.astype(int)
    y_pixels = measured_cat['y_pixel'].values.astype(int)
    src_idx_map_values = measured_cat['idx'].values.astype(int) + 1
    src_idx_map[y_pixels, x_pixels] = src_idx_map_values
    
    # loop through positions in SN map & find any measured sources at that location:
    src_idxs = [] # measured source indices
    sn_mask = enmap.ndmap(np.greater(segmentation_map, 0), sn_map.wcs)
    if nearby_sources_radius > 0:
        sn_mask = enmap.grow_mask(sn_mask, utils.arcmin2rad(nearby_sources_radius))
    src_idxs_in_missubtracted_regions = src_idx_map * sn_mask
    src_idxs = src_idxs_in_missubtracted_regions[src_idxs_in_missubtracted_regions > 0] - 1
            
    # if any missubtracted sources share a pixel with another src, add those to the list also:
    src_idxs = list(src_idxs)
    src_idxs_in_shared_pixels = [idx for idx in src_idxs if (idx in measured_srcs_in_shared_pixels['idx'].values)]
    for idx in src_idxs_in_shared_pixels:
        src_cat = measured_cat[measured_cat['idx'].eq(idx)]
        x_pixel = int(src_cat['x_pixel'].values[0])
        y_pixel = int(src_cat['y_pixel'].values[0])
        all_srcs_in_x_pixel = measured_cat[measured_cat['x_pixel'].eq(x_pixel)]
        all_srcs_in_pixel = all_srcs_in_x_pixel[all_srcs_in_x_pixel['y_pixel'].eq(y_pixel)]
        for idx in all_srcs_in_pixel['idx'].values:
            if idx not in src_idxs:
                src_idxs.append(int(idx))
                    
    return measured_srcs[measured_srcs['idx'].isin(list(set(src_idxs)))].copy()


def get_missubtracted_sources_catalog(sub_sim, catalog, min_snr, filt_obj, iter_num=None,
                                   freq=None, beam_solid_angle_func=None, 
                                   verbose=True, log=None):
    filt_map, sn_map, rms_map = filt_obj.calc_filtered_maps(sub_sim)   
    missubtracted_srcs = find_missubtracted_sources(catalog, sn_map, snr_threshold=min_snr, 
                                                    min_npix=1, nearby_sources_radius=0,
                                                    use_abs_sn=True, snr_threshold_below=None)
    iter_num = np.max(catalog['iter_num'].values) if (iter_num is None) else iter_num
    missubtracted_srcs['found_after_iter'] = iter_num
    missubtracted_srcs = maps.add_pixel_coords_to_catalog(missubtracted_srcs, sn_map.shape, sn_map.wcs)
    x_pixels = missubtracted_srcs['x_pixel'].values
    y_pixels = missubtracted_srcs['y_pixel'].values
    missubtracted_srcs['residual_SNR'] = sn_map[y_pixels, x_pixels]
    if (freq is not None) and (beam_solid_angle_func is not None):
        missubtracted_srcs['residual_deltaT_c'] = filt_map[y_pixels, x_pixels]
        missubtracted_srcs['residual_fluxmJy'] = utils.uK_to_mJy_per_str(missubtracted_srcs['residual_deltaT_c'].values, freq) * beam_solid_angle_func(np.round(missubtracted_srcs['decDeg'].values, 6))
    return missubtracted_srcs


def add_back_missubtracted_sources(sub_sim, measured_srcs_map, measured_srcs_catalog, 
                                   missubtracted_srcs_catalog, freq, beam_fwhm):
    """add missubtracted srcs back to source-subtracted sim, and remove them from catalog & map of measured srcs"""
    # remove missubtracted sources from catalog of measured sources
    srcs_to_keep = measured_srcs_catalog.copy()
    srcs_to_keep = srcs_to_keep[~srcs_to_keep['idx'].isin(missubtracted_srcs_catalog['idx'].values)]
    # make map of the missubtracted sources
    missubtracted_srcs_map = maps.make_src_map(sub_sim.shape, sub_sim.wcs, [missubtracted_srcs_catalog], freq)
    # remove the missubtracted srcs from the map of measured srcs:
    measured_srcs_to_keep_map = measured_srcs_map - missubtracted_srcs_map
    # add the misssubtracted srcs back to the source-subtracted sim:
    missubtracted_srcs_sim = maps.convolve_sim_with_beam(enmap.apply_window(missubtracted_srcs_map), beam_fwhm)
    new_sub_sim = sub_sim + missubtracted_srcs_sim
    return srcs_to_keep, new_sub_sim, measured_srcs_to_keep_map


def remeasure_missubtracted_sources(sub_sim, measured_srcs_map, measured_srcs_catalog, snr_threshold_list, 
                                    freq, beam_fwhm, apod_width, filter_obj, iter_catalog_dir,
                                    measured_positions_catalog=None, nearby_source_radius=None, 
                                    beam_solid_angle_func=None, missubtracted_srcs_catalog=None,
                                    snr_threshold=None, min_num_iter_sources_per_snr=10,
                                    max_ntimes_remeasure=25, verbose=True, log=None):
    if missubtracted_srcs_catalog is None:
        missubtracted_src_cols = [*measured_srcs_catalog.columns.values, 'found_after_iter', 
                                  'x_pixel', 'y_pixel', 'residual_SNR']
        if beam_solid_angle_func is not None:
            missubtracted_src_cols = [*missubtracted_src_cols, 'residual_deltaT_c', 'residual_fluxmJy']
        missubtracted_srcs_catalog = pd.DataFrame({col: [] for col in missubtracted_src_cols})
    if snr_threshold is None:
        snr_threshold = min(snr_threshold_list)
        
    # get un-apodized region (to count number of missubtracted sources)
    ra_ctr, dec_ctr, map_width, map_height = maps.get_map_ctr_extent(sub_sim.shape, sub_sim.wcs)
    width = map_width - 2 * apod_width
    height = map_height - 2 * apod_width
    # define a function to remove sources in apodized region from catalog:
    remove_apodized_sources = lambda icat: fgcatalogs.trim_catalog_positions(icat.copy(), ra_ctr=ra_ctr, dec_ctr=dec_ctr, width=width, height=height)

    t_start = time.time()
    remeasure = True
    ntimes_remeasured = 0
    prev_num_missubtracted_srcs = None
    while remeasure:
        t = time.time()
        # need to initialize `idx0`, `inum`, and `prev_inum` before we update the catalog by removing missubtracted srcs
        prev_inum = get_max_iter_num(iter_catalog_dir)
        inum = prev_inum + 1
        idx0 = int(max([*measured_srcs_catalog['idx'].values, *missubtracted_srcs_catalog['idx'].values])) + 1
        if ntimes_remeasured > 0:
            prev_num_missubtracted_srcs = num_missubtracted_srcs
        
        # find missubtracted srcs:
        t0 = time.time()
        fgutils.print_msg(f'getting catalog of missubtracted sources after iter {prev_inum}', verbose=verbose, log=log)
        if prev_inum in missubtracted_srcs_catalog['found_after_iter'].values:
            iter_missubtracted_srcs = missubtracted_srcs_catalog[missubtracted_srcs_catalog['found_after_iter'].eq(prev_inum)]
        else:
            iter_missubtracted_srcs = get_missubtracted_sources_catalog(sub_sim, measured_srcs_catalog, snr_threshold, filter_obj, 
                                                                     iter_num=prev_inum,
                                                                     freq=freq, beam_solid_angle_func=beam_solid_angle_func, 
                                                                     verbose=verbose, log=log)
            missubtracted_srcs_catalog = fgcatalogs.combine_catalogs([missubtracted_srcs_catalog, iter_missubtracted_srcs])
        # count the number of missubtracted srcs within the inner un-apodized region
        num_missubtracted_srcs = len(remove_apodized_sources(iter_missubtracted_srcs))
        fgutils.print_msg((f"{utils.tmsg(time.time() - t0)} to find {num_missubtracted_srcs} missubtracted sources "
                           f"within un-apodized region ({len(iter_missubtracted_srcs)} in full map)"), 
                          verbose=verbose, log=log)

        # remove them from the catalog and map of srcs in catalog ; add them back to the src-subtracted sim
        measured_srcs_catalog, sub_sim, measured_srcs_map = add_back_missubtracted_sources(sub_sim, measured_srcs_map, 
                                                                                           measured_srcs_catalog, 
                                                                                           iter_missubtracted_srcs,
                                                                                           freq, beam_fwhm)

        if measured_positions_catalog is not None:
            # update catalog of source positions that were measured at a different freq, but not yet found at this freq:
            unmeasured_srcs_cat = measured_positions_catalog[~measured_positions_catalog['idx'].isin(measured_srcs_catalog['catalog_idx'].values)].copy()
        else:
            unmeasured_srcs_cat = None
        
        # re-measure the sources that remain:
        remeasured_srcs_catalog, sub_sim, remeasured_srcs_map = iteratively_measure_sources(sub_sim, freq, beam_fwhm, filter_obj, 
                                                                                            snr_threshold_list, iter_catalog_dir,
                                                                                            initial_iter_num=inum, initial_source_id=idx0, 
                                                                                            remeasured=True,
                                                                                            min_num_iter_sources_per_snr=min_num_iter_sources_per_snr,
                                                                                            beam_solid_angle_func=beam_solid_angle_func,
                                                                                            measured_positions_catalog=unmeasured_srcs_cat,
                                                                                            nearby_source_radius=nearby_source_radius,
                                                                                            verbose=verbose, log=log)
        measured_srcs_map += remeasured_srcs_map
        num_measured_srcs = len(remeasured_srcs_catalog)
        if num_measured_srcs > 0:
            measured_srcs_catalog = fgcatalogs.combine_catalogs([measured_srcs_catalog, remeasured_srcs_catalog])
            # make sure that there is, at most, one source per pixel:
            srcs_in_shared_pixels = find_sources_in_shared_pixels(measured_srcs_catalog, sub_sim.shape, sub_sim.wcs)
            if len(srcs_in_shared_pixels) > 0:
                measured_srcs_catalog = combine_sources_in_shared_pixels(measured_srcs_catalog, sub_sim.shape, sub_sim.wcs, sort_output_catalog_by_col='idx')

        fgutils.print_msg(f'{utils.tmsg(time.time() - t)} to remeasure {num_measured_srcs} sources', verbose=verbose, log=log)
        ntimes_remeasured += 1
        if (ntimes_remeasured >= max_ntimes_remeasure) or (num_missubtracted_srcs in [0, prev_num_missubtracted_srcs]) or (num_measured_srcs == 0):
            remeasure = False
        
    fgutils.print_msg(f'{utils.tmsg(time.time() - t_start)} total to remeasure', verbose=verbose, log=log)
    return measured_srcs_catalog, missubtracted_srcs_catalog, sub_sim, measured_srcs_map


def measured_sources_catalog_2freqs(freq1, freq2, catalog1, catalog2):
    """
    `catalog1` is catalog of sources measured at `freq1`, using the source positions in `catalog2` measured at a different frequency `freq2`
    """
    # find sources measured at both `freq1` and another freq:
    measured_srcs1 = catalog1[~catalog1['method'].eq('map')].copy()
    measured_srcs2 = catalog2[catalog2['idx'].isin(catalog1['catalog_idx'].values)].copy()
    # only keep sources also measured at `freq2`:
    measured_srcs2 = measured_srcs2[measured_srcs2['freq'].eq(freq2)]
    measured_srcs1 = measured_srcs1[measured_srcs1['catalog_idx'].isin(measured_srcs2['idx'].values)]
    # make a single multi-frequency catalog
    measured_srcs1 = measured_srcs1.sort_values('catalog_idx')
    measured_srcs2 = measured_srcs2.sort_values('idx')
    multifreq_catalog_dict = {'RADeg': measured_srcs1['RADeg'].values, 'decDeg': measured_srcs1['decDeg'].values,
                              'idx': measured_srcs1['catalog_idx'].values,
                              f'fluxmJy_{freq1}GHz': measured_srcs1['fluxmJy'].values, 
                              f'SNR_{freq1}GHz': measured_srcs1['SNR'].values,
                              f'idx_{freq1}GHz': measured_srcs1['idx'].values,
                              f'fluxmJy_{freq2}GHz': measured_srcs2['fluxmJy'].values, 
                              f'SNR_{freq2}GHz': measured_srcs2['SNR'].values,
                              f'idx_{freq2}GHz': measured_srcs2['orig_idx'].values,
                             }
    multifreq_catalog = pd.DataFrame(multifreq_catalog_dict)
    return multifreq_catalog


def measure_avg_spectral_indices(freq1, freq2, catalog1, catalog2, min_snr4index=10, nsigma_for_index=3,
                                 return_catalogs=False, verbose=True, log=None):
    """measure the average CIB and radio spectral indices to extrapolate source fluxes from `freq2` to `freq1`
    
    i.e., `catalog1` is catalog of sources measured at `freq1`, using the source positions in `catalog2` measured at a different frequency `freq2`
    """
    # get catalog of sources measured at both `freq1` and `freq2`:
    catalog_for_index = measured_sources_catalog_2freqs(freq1, freq2, catalog1, catalog2)
    # only use sources with high SNR:
    catalog_for_index = catalog_for_index[catalog_for_index[f'SNR_{freq1}GHz'].ge(min_snr4index)]
    catalog_for_index = catalog_for_index[catalog_for_index[f'SNR_{freq2}GHz'].ge(min_snr4index)]
    # measure spectral index of each source
    catalog_for_index['index'] = fgutils.get_spectral_index(catalog_for_index[f'fluxmJy_{freq1}GHz'].values,
                                                    catalog_for_index[f'fluxmJy_{freq2}GHz'].values, freq1, freq2)
    # calculate average radio spectral index
    catalog_for_radio_index = catalog_for_index[catalog_for_index['index'].le(0)].copy()
    radio_index_mean, radio_index_std, cat_for_radio_index = fgutils.calc_avg_spectral_index(catalog_for_radio_index, freq1, freq2, 
                                                                                     nsigma_to_remove=nsigma_for_index, 
                                                                                     verbose=verbose, log=log)
    # calculate average CIB spectral index
    catalog_for_cib_index = catalog_for_index[catalog_for_index['index'].ge(0)].copy()
    cib_index_mean, cib_index_std, cat_for_cib_index = fgutils.calc_avg_spectral_index(catalog_for_cib_index, freq1, freq2, 
                                                                               nsigma_to_remove=nsigma_for_index, 
                                                                               verbose=verbose, log=log)
    if return_catalogs:
        catalogs_dict = {'all': catalog_for_index, 'radio': cat_for_radio_index, 'cib': cat_for_cib_index}
        return radio_index_mean, radio_index_std, cib_index_mean, cib_index_std, catalogs_dict
    else:
        return radio_index_mean, radio_index_std, cib_index_mean, cib_index_std



def identify_sources(freq, measured_srcs_catalog, measured_positions_catalog,
                              radio_index_mean, radio_index_std, cib_index_mean, cib_index_std,
                              freq_for_radio=90, freq_for_cib=277,
                             ):
    # find sources in the catalog that were only measured at this frequency:
    src_idxs_1freq = measured_srcs_catalog[measured_srcs_catalog['method'].eq('map')]['idx'].values.copy()
    # if this is the frequency used to find radio sources, then assume these sources are radio:
    radio_src_idxs = src_idxs_1freq if (freq == freq_for_radio) else []
    # if this is the frequency used to find CIB sources, then assume these sources are CIB:
    cib_src_idxs = src_idxs_1freq if (freq == freq_for_cib) else []
    # otherwise, we don't know what kind of sources they are:
    unknown_src_idxs = src_idxs_1freq if (freq not in [freq_for_radio, freq_for_cib]) else []
    
    # find index at which abs(radio_index - index) / radio_index_err = abs(cib_index - index) / cib_index_error
    index0 = (radio_index_mean * cib_index_std + cib_index_mean * radio_index_std) / (cib_index_std + radio_index_std)
    # calculate spectral index of sources also measured at another freq, and compare it to `index0` to determine if cib or radio:
    for freq2 in set(measured_positions_catalog['freq'].values):
        measured_srcs_2freqs = measured_sources_catalog_2freqs(freq, freq2, measured_srcs_catalog, measured_positions_catalog)
        measured_srcs_2freqs['index'] = fgutils.get_spectral_index(measured_srcs_2freqs[f'fluxmJy_{freq}GHz'].values,
                                                           measured_srcs_2freqs[f'fluxmJy_{freq2}GHz'].values,
                                                           freq, freq2)
        radio_srcs_2freqs = measured_srcs_2freqs[measured_srcs_2freqs['index'].le(index0)]
        radio_src_idxs = [*radio_src_idxs, *radio_srcs_2freqs[f'idx_{freq}GHz'].values]
        cib_srcs_2freqs = measured_srcs_2freqs[measured_srcs_2freqs['index'].gt(index0)]
        cib_src_idxs = [*cib_src_idxs, *cib_srcs_2freqs[f'idx_{freq}GHz'].values]
    
    measured_radio_srcs = measured_srcs_catalog[measured_srcs_catalog['idx'].isin(radio_src_idxs)].copy()
    measured_radio_srcs['component'] = 'radio'
    measured_cib_srcs = measured_srcs_catalog[measured_srcs_catalog['idx'].isin(cib_src_idxs)].copy()
    measured_cib_srcs['component'] = 'cib'
    measured_unknown_srcs = measured_srcs_catalog[measured_srcs_catalog['idx'].isin(unknown_src_idxs)].copy()
    measured_unknown_srcs['component'] = 'unknown'

    catalog = fgcatalogs.combine_catalogs([measured_radio_srcs,  measured_cib_srcs, measured_unknown_srcs])
    catalog = catalog.sort_values('idx').reset_index(drop=True)
    return catalog


def extrapolate_unmeasured_source_fluxes(freq, measured_srcs_catalog, measured_positions_catalog,
                                        radio_index=None, cib_index=None,
                                        freq_for_radio=90, freq_for_cib=277,
                                        radio_idxs_to_exclude=None, cib_idxs_to_exclude=None,
                                        min_extrap_snr=None,
                                        ):
    """
    `measured_srcs_catalog` = catalog of srcs measured at `freq`
    `measured_positions_catalog` = catalog of srcs measured at different freq(s) ; used to locate srcs in map at `freq` 
    
    `idxs_to_exclude` are the `idx` in `measured_positions_catalog` for missubtracted srcs at other freq 
        (`catalog_idx` in the `measured_srcs_catalog`); these won't be extrapolated
    """
    # sources measured at a different frequency that will be extrapolated:
    srcs_to_extrap = measured_positions_catalog[~measured_positions_catalog['idx'].isin(measured_srcs_catalog['catalog_idx'].values)]
    if min_extrap_snr is not None:
        srcs_to_extrap = srcs_to_extrap[srcs_to_extrap['SNR'].ge(min_extrap_snr)]

    # make a catalog of these sources w/ flux extrapolated to this `freq`:
    extrapolated_src_catalogs = []
    
    if (freq_for_radio in srcs_to_extrap['freq'].values) and (radio_index is not None):
        # get catalog of radio sources measured at the `freq_for_radio`:
        radio_srcs_to_extrap = srcs_to_extrap[srcs_to_extrap['freq'].eq(freq_for_radio)].copy()
        if radio_idxs_to_exclude is not None:
            radio_srcs_to_extrap = radio_srcs_to_extrap[~radio_srcs_to_extrap['orig_idx'].isin(radio_idxs_to_exclude)]
        # make catalog of these radio sources w/ flux extrapolated to `freq`:
        extrapolated_radio_srcs = radio_srcs_to_extrap[['RADeg', 'decDeg']].copy()
        extrapolated_radio_srcs['catalog_idx'] = radio_srcs_to_extrap['idx'].values.copy()
        extrapolated_radio_srcs['fluxmJy'] = fgutils.extrapolate_flux(radio_srcs_to_extrap['fluxmJy'].values, freq_for_radio, freq, radio_index)
        extrapolated_radio_srcs['component'] = 'radio'
        extrapolated_src_catalogs.append(extrapolated_radio_srcs)
    
    if (freq_for_cib in srcs_to_extrap['freq'].values) and (cib_index is not None):
        # get catalog of cib sources measured at the `freq_for_cib`:
        cib_srcs_to_extrap = srcs_to_extrap[srcs_to_extrap['freq'].eq(freq_for_cib)].copy()
        if cib_idxs_to_exclude is not None:
            cib_srcs_to_extrap = cib_srcs_to_extrap[~cib_srcs_to_extrap['orig_idx'].isin(cib_idxs_to_exclude)]
        # make catalog of these cib sources w/ flux extrapolated to `freq`:
        extrapolated_cib_srcs = cib_srcs_to_extrap[['RADeg', 'decDeg']].copy()
        extrapolated_cib_srcs['catalog_idx'] = cib_srcs_to_extrap['idx'].values.copy()
        extrapolated_cib_srcs['fluxmJy'] = fgutils.extrapolate_flux(cib_srcs_to_extrap['fluxmJy'].values, freq_for_cib, freq, cib_index)
        extrapolated_cib_srcs['component'] = 'cib'
        extrapolated_src_catalogs.append(extrapolated_cib_srcs)
        
    if len(extrapolated_src_catalogs) > 0:
        extrapolated_srcs = fgcatalogs.combine_catalogs(extrapolated_src_catalogs)
        extrapolated_srcs = extrapolated_srcs.sort_values('catalog_idx').reset_index(drop=True)
    else:
        extrapolated_srcs = pd.DataFrame({col: [] for col in ['RADeg', 'decDeg', 'catalog_idx', 'fluxmJy', 'component']})
    
    return extrapolated_srcs


def add_extrapolated_sources_to_catalog(sub_sim, measured_srcs_catalog, measured_positions_catalog,
                                        freq, filter_obj, #extrap_iter_num,
                                        radio_index=None, cib_index=None,
                                        freq_for_radio=90, freq_for_cib=277,
                                        radio_idxs_to_exclude=None, cib_idxs_to_exclude=None,
                                        min_extrap_snr=None,
                                        iter_catalog_dir=None,
                                       ):
    
    extrapolated_srcs_catalog = extrapolate_unmeasured_source_fluxes(freq, measured_srcs_catalog, measured_positions_catalog,
                                                                     radio_index=radio_index, cib_index=cib_index,
                                                                     freq_for_radio=freq_for_radio, freq_for_cib=freq_for_cib,
                                                                     radio_idxs_to_exclude=radio_idxs_to_exclude, 
                                                                     cib_idxs_to_exclude=cib_idxs_to_exclude,
                                                                     min_extrap_snr=min_extrap_snr)
    
    # add columns for the SNR and amplitude in the map at positions of extrapolated sources:
    extrap_x_pixels, extrap_y_pixels = maps.get_pixel_positions(extrapolated_srcs_catalog['RADeg'].values,
                                                                extrapolated_srcs_catalog['decDeg'].values, 
                                                                sub_sim.shape, sub_sim.wcs)
    filt_map, sn_map, rms_map = filter_obj.calc_filtered_maps(sub_sim)
    extrapolated_srcs_catalog['deltaT_c'] = filt_map[extrap_y_pixels, extrap_x_pixels]
    extrapolated_srcs_catalog['SNR'] = sn_map[extrap_y_pixels, extrap_x_pixels]
    extrapolated_srcs_catalog = extrapolated_srcs_catalog.sort_values('SNR', ascending=False).reset_index(drop=True)
    
    # fill in other columns:
    if iter_catalog_dir is not None:
        extrap_inum = get_max_iter_num(iter_catalog_dir) + 1
        extrap_idx0 = get_max_source_idx(iter_catalog_dir) + 1
    else:
        extrap_inum = np.max(measured_srcs_catalog['iter_num'].values) + 1
        extrap_idx0 = np.max(measured_srcs_catalog['idx'].values) + 1
    extrapolated_srcs_catalog['iter_num'] = extrap_inum
    extrapolated_srcs_catalog['idx'] = list(range(extrap_idx0, extrap_idx0 + len(extrapolated_srcs_catalog)))
    extrapolated_srcs_catalog['method'] = 'extrap'
    extrapolated_srcs_catalog['snr_threshold'] = 0
    extrapolated_srcs_catalog['numSigPix'] = 0
    extrapolated_srcs_catalog['remeasured'] = False
    
    # combine catalogs of measured & extrapolated sources:
    catalog = fgcatalogs.combine_catalogs([measured_srcs_catalog, extrapolated_srcs_catalog])
    
    return catalog


def extrapolate_and_subtract_unmeasured_sources(sub_sim, measured_srcs_map,
                                                measured_srcs_catalog, measured_positions_catalog,
                                                freq, beam_fwhm, 
                                                filter_obj,
                                                radio_index=None, cib_index=None,
                                                freq_for_radio=90, freq_for_cib=277,
                                                radio_idxs_to_exclude=None, cib_idxs_to_exclude=None,
                                                min_extrap_snr=None,
                                                iter_catalog_dir=None,
                                               ):
    catalog = add_extrapolated_sources_to_catalog(sub_sim, measured_srcs_catalog, measured_positions_catalog,
                                                  freq, filter_obj,
                                                  radio_index=radio_index, cib_index=cib_index,
                                                  freq_for_radio=freq_for_radio, freq_for_cib=freq_for_cib,
                                                  radio_idxs_to_exclude=radio_idxs_to_exclude, 
                                                  cib_idxs_to_exclude=cib_idxs_to_exclude,
                                                  min_extrap_snr=min_extrap_snr,
                                                  iter_catalog_dir=iter_catalog_dir,
                                                 )
    # make a map of extrapolated sources:
    extrapolated_srcs_catalog = catalog[catalog['method'].eq('extrap')]
    extrapolated_srcs_map = maps.make_src_map(sub_sim.shape, sub_sim.wcs, [extrapolated_srcs_catalog], freq)
    # add it to map of all subtracted sources:
    subtracted_srcs_map = measured_srcs_map + extrapolated_srcs_map
    # subtract the extrapolated sources from the source-subtracted map:
    sub_sim -= maps.convolve_sim_with_beam(enmap.apply_window(extrapolated_srcs_map.copy()), beam_fwhm)
    return catalog, sub_sim, subtracted_srcs_map



# ---------- clusters ----------

def iter_clusters_catalog_fname(iter_num, snr_threshold, iter_catalog_dir=None):
    fname = f'iter{iter_num:03d}_SNRabove{simutils.round_str(snr_threshold)}_subtracted_clusters.csv'
    if iter_catalog_dir is not None:
        fname = os.path.join(iter_catalog_dir, fname)
    return fname


def all_iter_clusters_catalog_fname(iter_num, snr_threshold, iter_catalog_dir=None):
    fname = f'iter{iter_num:03d}_SNRabove{simutils.round_str(snr_threshold)}_all_measured_clusters.csv'
    if iter_catalog_dir is not None:
        fname = os.path.join(iter_catalog_dir, fname)
    return fname


def measure_clusters_in_map_above_snr(filtered_map, sn_map, snr_threshold, min_npix=1):
    obj_ids, obj_positions, num_pix, segmentation_map = find_objs(sn_map, snr_threshold=snr_threshold, 
                                                                  min_npix=min_npix)
    if len(obj_ids) > 0:
        y_pixels, x_pixels = np.array(obj_positions).T
        ras, decs = maps.get_coord_positions(x_pixels, y_pixels, sn_map.shape, sn_map.wcs)
        snrs = sn_map[y_pixels, x_pixels]
        ymaxs = filtered_map[y_pixels, x_pixels]
        ocat = pd.DataFrame({'RADeg': ras, 'decDeg': decs, 'y_c': ymaxs, 'SNR': snrs, 'numSigPix': num_pix})
        ocat['snr_threshold'] = snr_threshold
        ocat = ocat.sort_values('SNR', ascending=False)
        ocat = ocat.reset_index(drop=True)
    else:
        ocat = pd.DataFrame({col: [] for col in ['RADeg', 'decDeg', 'y_c' 'SNR', 'numSigPix', 'snr_threshold']})
    return ocat
        

def measure_clusters_in_maps_above_snr(filtered_maps, sn_maps, snr_threshold, min_npix=1):
    catalogs = {}
    profile_names = list(filtered_maps.keys())
    for profile_name in profile_names:
        catalogs[profile_name] = measure_clusters_in_map_above_snr(filtered_maps[profile_name], sn_maps[profile_name], 
                                                                   snr_threshold, min_npix=min_npix)
        catalogs[profile_name]['template'] = profile_name
    all_clusters = fgcatalogs.combine_catalogs([catalogs[profile_name] for profile_name in profile_names])
    all_clusters['iter_idx'] = list(range(len(all_clusters)))
    if len(all_clusters) > 0:
        all_clusters = all_clusters.sort_values('SNR', ascending=False).reset_index(drop=True)
    return all_clusters


def measure_clusters_above_snr(filtered_maps, sn_maps, snr_threshold, match_radius=1, min_idx=0, min_npix=1):
    all_clusters = measure_clusters_in_maps_above_snr(filtered_maps, sn_maps, snr_threshold, min_npix=min_npix)

    '''match_radius in arcmin'''
    # in the original catalog of all clusters, keep track of the final 'idx' that is assigned to each individual cluster (same across all profiles)
    all_clusters['idx'] = -1  # placeholder
    idx = min_idx
    # keep track of which profile to keep in 'final' catalog (the ones w/ highest SNR) for each cluster
    idxs_to_keep = []
    idxs_to_remove = []
    # loop through catalog of all clusters from this iter ; already sorted so max SNR is first
    all_clusters = measure_clusters_in_maps_above_snr(filtered_maps, sn_maps, snr_threshold, min_npix=min_npix)
    unmatched_clusters = all_clusters.copy()
    while len(unmatched_clusters) > 0:
        index = unmatched_clusters.index.values[0]
        cluster = unmatched_clusters.loc[index] # highest SNR measurement at this position
        # find all measurements (using different profiles/filters) at this location:
        nearby_clusters = unmatched_clusters.copy()
        nearby_clusters['dist'] = maps.angular_distance(cluster['RADeg'], cluster['decDeg'], 
                                                        nearby_clusters['RADeg'].values, nearby_clusters['decDeg'].values)
        nearby_clusters = nearby_clusters[nearby_clusters['dist'].le(match_radius)]
        iter_idxs = nearby_clusters['iter_idx'].values
        # keep highest SNR measurment of this cluster
        idxs_to_keep.append(iter_idxs[0]) 
        idxs_to_remove = [*idxs_to_remove, *iter_idxs[1:]] # other measurments to remove from final, combined catalog
        # update catalogs measured from each profile:
        index_list = all_clusters[all_clusters['iter_idx'].isin(iter_idxs)].index.values
        all_clusters.loc[index_list, 'idx'] = idx
        unmatched_clusters = unmatched_clusters[~unmatched_clusters['iter_idx'].isin(iter_idxs)]
        idx += 1
    combined_cat = all_clusters[all_clusters['iter_idx'].isin(idxs_to_keep)].copy()
    if len(combined_cat) > 0:
        combined_cat = combined_cat.sort_values('SNR', ascending=False).reset_index(drop=True)
    return combined_cat, all_clusters


def iteratively_measure_clusters(imaps, freqs, beam_fwhms, cluster_filters, snr_threshold_list, iter_catalog_dir,
                                 iter_match_radius=1, num_iter_clusters_threshold=1, verbose=True, log=None):
    '''
    `cluster_filters` is an instance of `ClusterFilters`
    `iter_match_radius` in arcmin ; used
    '''
    t = time.time()
    iter_sub_sims = imaps.copy()
    shape = imaps.shape
    wcs = imaps.wcs
    subtracted_clusters_sims = enmap.zeros(shape, wcs)
    inum = 1
    idx0 = 0
    num_measured_clusters = 0
    iter_catalogs = {}
    for snr_threshold in snr_threshold_list:
        min_num_clusters = num_iter_clusters_threshold if (snr_threshold > min(snr_threshold_list)) else 1
        num_iter_clusters = min_num_clusters + 1
        while num_iter_clusters >= min_num_clusters:
            t0 = time.time()
            iter_num = inum # just used to print out info about this iter.

            iter_clusters_cat_fname = iter_clusters_catalog_fname(inum, snr_threshold,
                                                                  iter_catalog_dir=iter_catalog_dir)
            all_iter_clusters_cat_fname = all_iter_clusters_catalog_fname(inum, snr_threshold,
                                                                          iter_catalog_dir=iter_catalog_dir)
            if os.path.exists(iter_clusters_cat_fname):
                iter_clusters_cat = fgcatalogs.load_catalog(iter_clusters_cat_fname)
                all_iter_clusters_cat = fgcatalogs.load_catalog(all_iter_clusters_cat_fname)
            else:
                iter_filt_maps, iter_sn_maps, iter_rms_maps = cluster_filters.get_filtered_maps(iter_sub_sims)
                iter_clusters_cat, all_iter_clusters_cat = measure_clusters_above_snr(iter_filt_maps, iter_sn_maps,
                                                                                      snr_threshold, min_idx=idx0,
                                                                                      match_radius=iter_match_radius)
                iter_clusters_cat['iter_num'] = inum
                all_iter_clusters_cat['iter_num'] = inum

            num_iter_clusters = len(iter_clusters_cat)
            num_measured_clusters += num_iter_clusters
            idx0 += num_iter_clusters
            if num_iter_clusters > 0:
                # save the catalogs
                if not os.path.exists(iter_clusters_cat_fname):
                    iter_clusters_cat.to_csv(iter_clusters_cat_fname)
                if not os.path.exists(all_iter_clusters_cat_fname):
                    all_iter_clusters_cat.to_csv(all_iter_clusters_cat_fname)
                # subtract the measured clusters
                iter_clusters_sims = fgmaps.make_cluster_sims_from_catalog(shape[-2:], wcs, iter_clusters_cat,
                                                                           cluster_filters.profiles, freqs,
                                                                           convolve_pixwin=True,
                                                                           convolve_beam=True, beam_fwhms=beam_fwhms)
                iter_clusters_sims = fgmaps._multifreq_mapdict_to_maps(iter_clusters_sims, freqs=freqs, copy=False)
                iter_sub_sims -= iter_clusters_sims.copy()
                subtracted_clusters_sims += iter_clusters_sims.copy()
                iter_catalogs[inum] = iter_clusters_cat.copy()
                inum += 1

            fgutils.print_msg((f"{utils.tmsg(time.time() - t0)} for iter {iter_num}: found {num_iter_clusters} with "
                               f"SNR >= {snr_threshold} ({num_measured_clusters} total so far)"), verbose=verbose, log=log)

    if num_measured_clusters > 0:
        cluster_catalog = fgcatalogs.combine_catalogs([iter_catalogs[inum] for inum in sorted(list(iter_catalogs.keys()))])
    else:
        cluster_catalog = pd.DataFrame({col: [] for col in [*iter_clusters_cat.columns.values, 'iter_num']})
    measured_clusters_sims = fgmaps._multifreq_maps_to_mapdict(subtracted_clusters_sims, freqs, copy=False)
    sub_sims =  fgmaps._multifreq_maps_to_mapdict(iter_sub_sims, freqs, copy=False)
    fgutils.print_msg(f'{utils.tmsg(time.time() - t)} to measure {len(cluster_catalog)} clusters', verbose=verbose, log=log)
    return cluster_catalog, sub_sims, measured_clusters_sims



