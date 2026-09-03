import numpy as np
from pixell import enmap
from hdsims import utils, simutils, fgcatalogs, maps
from . import fgutils, fgmaps, iterfgclean


def make_binary_snr_mask(sn_map, snr_threshold=5, use_abs_sn=True, snr_threshold_below=None, min_npix=1, verbose=False):
    # get map that is non-zero only in regions that exceed the SNR threshold:
    _, _, _, obj_seg_map = iterfgclean.find_objs(sn_map, snr_threshold=snr_threshold,
                                                 snr_threshold_below=snr_threshold_below,
                                                 use_abs_sn=use_abs_sn, min_npix=min_npix)
    mask = enmap.ones(sn_map.shape, sn_map.wcs) * np.equal(obj_seg_map, 0)
    return mask


def apodize_mask(mask, apod_width):
    '''apod_width in arcmin'''
    if apod_width > 0:
        mask = enmap.apod_mask(mask, width=utils.arcmin2rad(apod_width), edge=False, profile=enmap.apod_profile_cos)
    return mask


def make_source_mask(imap, source_filter, shape=None, wcs=None, apod_width=1, snr_threshold=5,
                     use_abs_sn=True, snr_threshold_below=None, min_npix=1, verbose=False, log=None):
    '''shape, wcs are for mask ; may be a smaller region than the input `imap`
    '''
    _, sn_map, _ = source_filter.calc_filtered_maps(imap)
    if (shape is not None) and (wcs is not None):
        sn_map = enmap.project(sn_map, shape, wcs)
    mask = make_binary_snr_mask(sn_map, snr_threshold=snr_threshold, use_abs_sn=use_abs_sn,
                                snr_threshold_below=snr_threshold_below, min_npix=min_npix, verbose=verbose)
    if apod_width > 0:
        mask = apodize_mask(mask, apod_width)
    if verbose:
        masked_npix = np.less(mask, 1).sum()
        map_npix = sn_map.shape[-1] * sn_map.shape[-2]
        masked_frac = simutils.round_str(100 * masked_npix / map_npix)
        fgutils.print_msg(f"{masked_frac}% of the map is masked due to point sources", verbose=verbose, log=log)
    return mask


def make_cluster_mask(imaps, cluster_filters, profile_names=None, shape=None, wcs=None, apod_width=1, snr_threshold=5,
                      use_abs_sn=True, snr_threshold_below=None, min_npix=1, verbose=False, log=None):
    '''shape, wcs are for mask ; may be a smaller region than the input `imaps`
    profile_names can be subset of profiles in the cluster_filters instance
    imaps should be a dict
    '''
    profile_names = cluster_filters.profile_names if (profile_names is None) else profile_names
    maps_to_filter = fgmaps._multifreq_mapdict_to_maps(imaps, freqs=cluster_filters.freqs)
    _, sn_maps, _ = cluster_filters.get_filtered_maps(maps_to_filter, profile_names=profile_names)
    if (shape is None) or (wcs is None):
        shape = cluster_filters.shape
        wcs = cluster_filters.wcs
    mask = enmap.ones(shape, wcs)
    for profile_name  in profile_names:
        sn_map = enmap.project(sn_maps[profile_name], shape, wcs)
        mask *= make_binary_snr_mask(sn_map, snr_threshold=snr_threshold, use_abs_sn=use_abs_sn,
                                     snr_threshold_below=snr_threshold_below, min_npix=min_npix, verbose=verbose)
    if apod_width > 0:
        mask = apodize_mask(mask, apod_width)
    if verbose:
        masked_npix = np.less(mask, 1).sum()
        map_npix = sn_map.shape[-1] * sn_map.shape[-2]
        masked_frac = simutils.round_str(100 * masked_npix / map_npix)
        fgutils.print_msg(f"{masked_frac}% of the map is masked due to clusters", verbose=verbose, log=log)
    return mask


def make_fg_masks(imaps, freqs=None, source_filters=None, cluster_filters=None, profile_names=None,
                  shape=None, wcs=None, apod_width=1, snr_threshold=5, use_abs_sn=True,
                  snr_threshold_below=None, min_npix=1, verbose=False, log=None):
    '''
    imaps is dict of maps to mask
    freqs is list of freqs to make mask for
    source_filters is dict of instances of src filt class for each freq
    cluster_filters is instance of cluster filt class

    note: if making mask for clusters, need to pass maps at all freqs used for cluster filter(s), even if not making mask for each freq

    note: does not include the apodization window used to apodize edges of map
    '''
    freqs = sorted(list(imaps.keys())) if (freqs is None) else freqs
    # make the (binary) mask for the clusters (same for all frequencies):
    if cluster_filters is not None:
        cluster_mask = make_cluster_mask(imaps, cluster_filters, profile_names=profile_names, shape=shape, wcs=wcs,
                                          apod_width=0, snr_threshold=snr_threshold,
                                          use_abs_sn=use_abs_sn, snr_threshold_below=snr_threshold_below,
                                          min_npix=min_npix, verbose=verbose, log=log)
    else:
        mask_shape = imaps[freqs[0]].shape[-2:] if (shape is None) else shape
        mask_wcs = imaps[freqs[0]].wcs if (wcs is None) else wcs
        cluster_mask = enmap.ones(mask_shape, mask_wcs)
    # get mask for each freq:
    masks = {}
    for freq in freqs:
        masks[freq] = cluster_mask.copy()
        if source_filters is not None:
            # mask the (binary) mask for the point sources at this freq:
            masks[freq] *= make_source_mask(imaps[freq], source_filters[freq], shape=shape, wcs=wcs,
                                            apod_width=0, snr_threshold=snr_threshold,
                                            use_abs_sn=use_abs_sn, snr_threshold_below=snr_threshold_below,
                                            min_npix=min_npix, verbose=verbose, log=log)

        if apod_width > 0: # apodize the holes in the mask
            masks[freq] = apodize_mask(masks[freq], apod_width)
        if verbose:
            masked_npix = np.less(masks[freq], 1).sum()
            map_npix = masks[freq].shape[-1] * masks[freq].shape[-2]
            masked_frac = simutils.round_str(100 * masked_npix / map_npix)
            fgutils.print_msg(f"{simutils.round_str(freq)} GHz: {masked_frac}% of the map is masked", verbose=verbose, log=log)
    return masks


def make_binary_mask_from_catalog(shape, wcs, catalog, ra_col='RADeg', dec_col='decDeg', radius_col='mask_radius'):
    '''RA, dec in degrees ; radius in arcmin'''
    mask = enmap.ones(shape, wcs)
    for r in set(catalog[radius_col].values):
        mask *= fgmaps.make_binary_mask(catalog[catalog[radius_col].eq(r)][ra_col].values, catalog[catalog[radius_col].eq(r)][dec_col].values, r, shape, wcs)
    return mask


def make_mask_from_catalog(shape, wcs, catalog, ra_col='RADeg', dec_col='decDeg', radius_col='mask_radius', apod_width_col='apod_width'):
    '''RA, dec in degrees ; radius in arcmin'''
    mask = enmap.ones(shape, wcs)
    for apod_width in set(catalog[apod_width_col].values):
        mask *= apodize_mask(make_binary_mask_from_catalog(shape, wcs, catalog[catalog[apod_width_col].eq(apod_width)], 
                                                           ra_col=ra_col, dec_col=dec_col, radius_col=radius_col), apod_width)
    return mask


def get_masked_pixels(mask, min_unmasked_value=1):
    '''
    NOTE: should be a binary mask
    
    OR, for apod. mask, consider a pixel "masked" if its value in the mask is less than `min_unmasked_value` 
    '''
    ny, nx = mask.shape[-2:]
    max_pixel_num = nx * ny - 1
    all_pixel_nums = np.arange(max_pixel_num+1, dtype=int)
    all_x_pixels, all_y_pixels = maps.get_pixel_from_num(all_pixel_nums, mask.shape)
    all_pixels_map = enmap.zeros(mask.shape, mask.wcs)
    all_pixels_map[all_y_pixels, all_x_pixels] = all_pixel_nums + 1 # add one so all pixels > 0 initially
    
    binary_mask = enmap.ones(mask.shape, mask.wcs) * np.greater_equal(mask, min_unmasked_value)
    masked_pixels_map = all_pixels_map * (1 - binary_mask)
    masked_pixel_nums = masked_pixels_map[masked_pixels_map > 0].astype(int) - 1
    masked_x_pixels, masked_y_pixels = maps.get_pixel_from_num(masked_pixel_nums, mask.shape)
    
    return masked_x_pixels, masked_y_pixels, masked_pixel_nums


def add_mask_info_to_catalog(catalog, mask, min_unmasked_value=1, col_name='masked'):
    '''
    NOTE: should be a binary mask
    
    OR, for apod. mask, consider a pixel "masked" if its value in the mask is less than `min_unmasked_value` 
    '''
    icat_cols = catalog.columns.values.copy()
    icat = catalog.copy()
    # add column w/ label for each row, so we can return catalog w/ rows in same order as they were passed in:
    icat['icat_row_num'] = list(range(len(icat)))
    icat = maps.add_pixel_coords_to_catalog(icat, mask.shape, mask.wcs, add_pixel_num=True)
    if col_name not in icat_cols:
        icat[col_name] = False
    
    _, _, masked_pixel_nums = get_masked_pixels(mask, min_unmasked_value=min_unmasked_value)
    unmasked_cat = icat[~icat['pixel_num'].isin(masked_pixel_nums)].copy()
    masked_cat = icat[icat['pixel_num'].isin(masked_pixel_nums)].copy()
    masked_cat[col_name] = True
    
    ocat = fgcatalogs.combine_catalogs([unmasked_cat, masked_cat]).sort_values('icat_row_num').reset_index(drop=True)
    ocat_cols = [*icat_cols, col_name]
    ocat = ocat[ocat_cols].copy()
    return ocat

