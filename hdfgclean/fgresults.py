import time
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from hdsims import utils, fgcatalogs, maps
from . import fgutils


def match_subtracted_to_true_sources(subtracted_sources_catalog, true_sources_catalog, 
                                     match_radius, max_abs_flux_diff_vs_err=3,
                                     verbose=False, log=None):
    '''
    NOTE : true src cat should just have a col for fluxmJy (not `'fluxmJy_{freq}GHz'`)
    '''
    fgutils.print_msg(f"subtracted {len(subtracted_sources_catalog)} sources; {len(true_sources_catalog)} true sources in map", verbose=verbose, log=log)
    
    true_srcs = true_sources_catalog.copy()
    true_srcs = true_srcs.sort_values('fluxmJy', ascending=False).reset_index(drop=True)
    if 'idx' not in true_srcs.columns.values:
        true_srcs['idx'] = list(range(len(true_srcs)))
    
    # add column for flux error
    sub_srcs = subtracted_sources_catalog.copy()
    if 'extrap' in sub_srcs['method'].values:
        # calculate an error bar for the flux measurements:
        # for sources that weren't extrapolated, use the SNR:
        measured_srcs = sub_srcs[~sub_srcs['method'].eq('extrap')].copy()
        measured_srcs['fluxmJy_err'] = measured_srcs['fluxmJy'].values / measured_srcs['SNR'].values
        # for extrapolated srcs, use avg. error for measured dim srcs
        avg_flux_err = np.mean(measured_srcs[measured_srcs['SNR'].lt(5)]['fluxmJy_err'].values)
        extrap_srcs = sub_srcs[sub_srcs['method'].eq('extrap')].copy()
        extrap_srcs['fluxmJy_err'] = avg_flux_err
        # put all srcs back in single catalog:
        sub_srcs = fgcatalogs.combine_catalogs([measured_srcs, extrap_srcs])
    else:
        sub_srcs['fluxmJy_err'] = sub_srcs['fluxmJy'].values / sub_srcs['SNR'].values
    sub_srcs = sub_srcs.sort_values('SNR', ascending=False).reset_index(drop=True)
    
    unmatched_true_srcs = true_srcs.copy()
    matched_sub_idxs = []
    matched_true_idxs = []
    t0 = time.time()
    for _, sub_src in sub_srcs.iterrows():
        sub_idx = sub_src['idx']
        sub_ra = sub_src['RADeg']
        sub_dec = sub_src['decDeg']
        # find true sources (that haven't yet been matched) within the match radius
        nearby_true_srcs = unmatched_true_srcs.copy()
        nearby_true_srcs['dist'] = maps.angular_distance(sub_ra, sub_dec, nearby_true_srcs['RADeg'].values, nearby_true_srcs['decDeg'].values)
        nearby_true_srcs = nearby_true_srcs[nearby_true_srcs['dist'].le(match_radius)]
        if len(nearby_true_srcs) > 0:
            # compare flux of each true source to the subtracted source
            nearby_true_srcs['abs_flux_diff_vs_err'] = np.abs(nearby_true_srcs['fluxmJy'].values - sub_src['fluxmJy']) / sub_src['fluxmJy_err']
            nearby_true_srcs = nearby_true_srcs[nearby_true_srcs['abs_flux_diff_vs_err'].le(max_abs_flux_diff_vs_err)]
            if len(nearby_true_srcs) > 0:
                # keep the closest match
                true_idx = nearby_true_srcs['idx'].values[0]
                matched_true_idxs.append(true_idx)
                unmatched_true_srcs = unmatched_true_srcs[~unmatched_true_srcs['idx'].eq(true_idx)].copy()
                matched_sub_idxs.append(sub_idx)
    fgutils.print_msg(f"{utils.tmsg(time.time() - t0)} to match {len(matched_sub_idxs)} sources", verbose=verbose, log=log)
            
    # return un-modified catalogs for unmatched sources
    unmatched_sub_srcs = subtracted_sources_catalog[~subtracted_sources_catalog['idx'].isin(matched_sub_idxs)].copy()
    unmatched_true_srcs = true_sources_catalog[~true_sources_catalog['idx'].isin(matched_true_idxs)].copy()
    fgutils.print_msg(f"{len(unmatched_sub_srcs)} subtracted sources (out of {len(subtracted_sources_catalog)} total subtracted) were not matched", verbose=verbose, log=log)
    fgutils.print_msg(f"{len(unmatched_true_srcs)} true sources (out of {len(true_sources_catalog)} total in map) were not matched", verbose=verbose, log=log)
    
    # make a single catalog of matched sources
    matched_srcs = pd.DataFrame({'idx': matched_sub_idxs, 'true_idx': matched_true_idxs})
    matched_srcs = matched_srcs.sort_values('idx')
    matched_sub_srcs = subtracted_sources_catalog[subtracted_sources_catalog['idx'].isin(matched_sub_idxs)].sort_values('idx').copy()
    for col in matched_sub_srcs.columns.values:
        if col != 'idx':
            matched_srcs[col] = matched_sub_srcs[col].values.copy()
    matched_srcs = matched_srcs.sort_values('true_idx')
    matched_true_srcs = true_srcs[true_srcs['idx'].isin(matched_true_idxs)].sort_values('idx').copy()
    for col in matched_true_srcs.columns.values:
        if col != 'idx':
            matched_srcs[f'true_{col}'] = matched_true_srcs[col].values.copy()
    if 'idx' not in true_sources_catalog.columns.values:
        matched_srcs = matched_srcs[[col for col in matched_srcs.columns.values if (col != 'true_idx')]]
    matched_srcs = matched_srcs.sort_values('SNR', ascending=False).reset_index(drop=True)
    return matched_srcs, unmatched_sub_srcs, unmatched_true_srcs
    
    
def match_subtracted_to_true_clusters(subtracted_clusters_catalog, true_clusters_catalog, 
                                      match_radius=1, remove_matches=False, verbose=False, log=None):
    '''
    match_radius in arcmin
    '''
    fgutils.print_msg(f"subtracted {len(subtracted_clusters_catalog)} clusters ; {len(true_clusters_catalog)} true clusters in map", verbose=verbose, log=log)
    t = time.time()
    
    true_clusters = true_clusters_catalog.copy()
    if 'idx' not in true_clusters.columns.values:
        true_clusters['idx'] = list(range(len(true_clusters)))
    
    matched_idxs = {} # key = true cluster idx ; value = list of subtracted cluster idxs matched to this true cluster
    matched_sub_idxs = [] # list of all subtracted cluster idx's that were matched to a true cluster
    unmatched_sub_clusters = subtracted_clusters_catalog.copy()
    for _, cluster in true_clusters.iterrows():
        unmatched_clusters = unmatched_sub_clusters.copy()
        unmatched_clusters['dist'] = maps.angular_distance(cluster['RADeg'], cluster['decDeg'], unmatched_clusters['RADeg'].values, unmatched_clusters['decDeg'].values)
        nearby_clusters = unmatched_clusters[unmatched_clusters['dist'].le(match_radius)].sort_values('dist').copy()
        if len(nearby_clusters) > 0:
            matched_idxs[int(cluster['idx'])] = nearby_clusters['idx'].values.tolist()
            matched_sub_idxs = [*matched_sub_idxs, *nearby_clusters['idx'].values]
            if remove_matches:
                unmatched_sub_clusters = unmatched_sub_clusters[~unmatched_sub_clusters['idx'].isin(matched_sub_idxs)]
            
    matched_true_idxs = list(matched_idxs.keys())
    matched_true_cat = true_clusters[true_clusters['idx'].isin(matched_true_idxs)].copy()
    unmatched_true_cat = true_clusters[~true_clusters['idx'].isin(matched_true_idxs)].copy()
    matched_sub_cat = subtracted_clusters_catalog[subtracted_clusters_catalog['idx'].isin(matched_sub_idxs)].copy()
    unmatched_sub_cat = subtracted_clusters_catalog[~subtracted_clusters_catalog['idx'].isin(matched_sub_idxs)].copy()
    
    fgutils.print_msg(f"{utils.tmsg(time.time() - t)} to match {len(matched_sub_cat)} subtracted clusters to {len(matched_true_cat)} true clusters", verbose=verbose, log=log)
    fgutils.print_msg(f"{len(unmatched_sub_cat)} subtracted clusters not matched, {len(unmatched_true_cat)} true clusters not matched", verbose=verbose, log=log)
    return matched_sub_cat, unmatched_sub_cat, matched_true_cat, unmatched_true_cat, matched_idxs

    
def get_source_completeness(matched_true_fluxes, all_true_fluxes, cumulative=True,
                          flux_bin_edges=None, num_flux_bins=25, log_flux_bins=True,
                         ):
    if flux_bin_edges is None:
        min_true_flux = np.min(all_true_fluxes)
        max_true_flux = np.max(all_true_fluxes)
        flux_bin_edges, flux_bin_ctrs = fgutils.make_uniform_bins(min_true_flux, max_true_flux, num_flux_bins, log=log_flux_bins)
    else:
        num_flux_bins = len(flux_bin_edges) - 1
        flux_bin_ctrs = fgutils.get_bin_centers(flux_bin_edges, log=log_flux_bins)
    lower_flux_bin_edges = flux_bin_edges[:-1]
    upper_flux_bin_edges = flux_bin_edges[1:]
            
    num_true_per_bin = np.zeros(num_flux_bins) # all true sources
    num_matched_per_bin = np.zeros(num_flux_bins) # detected sources matched to a true source
    for i, (min_flux, max_flux) in enumerate(zip(lower_flux_bin_edges, upper_flux_bin_edges)):
        if cumulative or (i == (num_flux_bins - 1)): # count number of true/matched sources above lower flux bin edge:
            num_true_per_bin[i] = len(all_true_fluxes[all_true_fluxes >= min_flux])
            num_matched_per_bin[i] = len(matched_true_fluxes[matched_true_fluxes >= min_flux])
        else:
            num_true_per_bin[i] = len(all_true_fluxes[(all_true_fluxes >= min_flux) & (all_true_fluxes < max_flux)])
            num_matched_per_bin[i] = len(matched_true_fluxes[(matched_true_fluxes >= min_flux) & (matched_true_fluxes < max_flux)])
    
    completeness = num_matched_per_bin / num_true_per_bin
    
    return completeness, flux_bin_ctrs, flux_bin_edges, num_matched_per_bin, num_true_per_bin


def get_source_purity(matched_measured_fluxes, all_measured_fluxes, cumulative=True,
                          flux_bin_edges=None, num_flux_bins=25, log_flux_bins=True,
                         ):
    if flux_bin_edges is None:
        min_true_flux = np.min(all_measured_fluxes)
        max_true_flux = np.max(all_measured_fluxes)
        flux_bin_edges, flux_bin_ctrs = fgutils.make_uniform_bins(min_true_flux, max_true_flux, num_flux_bins, log=log_flux_bins)
    else:
        num_flux_bins = len(flux_bin_edges) - 1
        flux_bin_ctrs = fgutils.get_bin_centers(flux_bin_edges, log=log_flux_bins)
    lower_flux_bin_edges = flux_bin_edges[:-1]
    upper_flux_bin_edges = flux_bin_edges[1:]

    num_per_bin = np.zeros(num_flux_bins) # all detected sources
    num_matched_per_bin = np.zeros(num_flux_bins) # detected sources matched to a true source
    for i, (min_flux, max_flux) in enumerate(zip(lower_flux_bin_edges, upper_flux_bin_edges)):
        if cumulative or (i == (num_flux_bins - 1)): # count number of true/matched sources above lower flux bin edge:
            num_per_bin[i] = len(all_measured_fluxes[all_measured_fluxes >= min_flux])
            num_matched_per_bin[i] = len(matched_measured_fluxes[matched_measured_fluxes >= min_flux])
        else:
            num_per_bin[i] = len(all_measured_fluxes[(all_measured_fluxes >= min_flux) & (all_measured_fluxes < max_flux)])
            num_matched_per_bin[i] = len(matched_measured_fluxes[(matched_measured_fluxes >= min_flux) & (matched_measured_fluxes < max_flux)])
    
    purity = num_matched_per_bin / num_per_bin
    
    return purity, flux_bin_ctrs, flux_bin_edges, num_matched_per_bin, num_per_bin


def get_cluster_completeness(all_true_clusters, matched_true_clusters, num_mass_bins=10, mass_bin_edges=None,
                             cumulative=True, mass_col='M500'):
    if mass_bin_edges is None:
        mass_bin_edges = np.logspace(np.log10(np.min(all_true_clusters[mass_col].values)), np.log10(np.max(all_true_clusters[mass_col].values)), num_mass_bins+1)
    else:
        num_mass_bins = len(mass_bin_edges) - 1
    lower_mass_bin_edges = mass_bin_edges[:-1]
    upper_mass_bin_edges = mass_bin_edges[1:]
    masses = fgutils.get_bin_centers(mass_bin_edges, log=True)

    num_true = np.zeros(num_mass_bins)
    num_matched = np.zeros(num_mass_bins)
    completeness = np.zeros(num_mass_bins)
    for i, (min_M500, max_M500) in enumerate(zip(lower_mass_bin_edges, upper_mass_bin_edges)):
        if cumulative:
            num_true[i] = len(all_true_clusters[all_true_clusters[mass_col].ge(min_M500)])
            num_matched[i] = len(matched_true_clusters[matched_true_clusters[mass_col].ge(min_M500)])
        else:
            num_true[i] = len(all_true_clusters[all_true_clusters[mass_col].between(min_M500, max_M500)])
            num_matched[i] = len(matched_true_clusters[matched_true_clusters[mass_col].between(min_M500, max_M500)])

        if num_true[i] == 0:
            completeness[i] = 0 if (num_matched[i] > 0) else 1
        else:
            completeness[i] = num_matched[i] / num_true[i]

    return masses, completeness, mass_bin_edges, num_true, num_matched



def get_coadded_noise(noise):
    '''
    noise is dict of noise spectra (at each ell) w/ keys for freqs
    '''
    all_freqs = list(noise.keys())
    # make sure we're not dividing by zero
    inv_noise = {}
    for f in all_freqs:
        if not np.all(noise[f] == 0):
            inv_noise[f] = 1. / noise[f]
            # if we divided by zero anywhere, set that element to zero
            inv_noise[f][~np.isfinite(inv_noise[f])] = 0.
    freqs = list(inv_noise.keys()) # only those with nonzero noise
    if len(freqs) > 0:
        # total weight = sum of inverse noise weights
        tot_weight = np.sum([inv_noise[f] for f in freqs], axis=0)
        # divide inverse-noise weights by total
        weights = {}
        for f in freqs:
            weights[f] = inv_noise[f] / tot_weight
            weights[f][~np.isfinite(weights[f])] = 0.
        # coadd the noise using these weights
        coadded = np.sum([noise[f] * weights[f]**2 for f in freqs], axis=0)
    else: # nothing left to coadd - just return zeros
        coadded = np.zeros(noise[all_freqs[0]].shape)
    return coadded


def interp_spectra(ells, cls, lmin=None, lmax=None, kind='cubic'):
    '''
    NOTE : if lmin, lmax fall outside range of `ells`, the interpolated power will be set to zero there
    '''
    # interpolate:
    ell_min = int(ells[0])
    if ell_min < ells[0]:
        ell_min += 1
    ell_max = int(ells[-1])
    ells_for_interp = np.arange(ell_min, ell_max+1)
    interpolated = interp1d(ells, cls, kind=kind)(ells_for_interp)
    # return interpolated power between lmin and lmax:
    lmin = lmin if (lmin is not None) else ell_min
    lmax = lmax if (lmax is not None) else ell_max
    ell_vals = np.arange(lmin, lmax+1, dtype=int)
    interp_cls = np.zeros(len(ell_vals))
    interp_cls[np.isin(ell_vals, ells_for_interp)] = interpolated[np.isin(ells_for_interp, ell_vals)].copy()
    return ell_vals, interp_cls


def calc_coadded_noise(ells, noise, interp=True, lmin=None, lmax=None, ells_to_fill=None, fill_vals=None, fill_val_ells=None):
    lmax = int(ells[-1]) if (lmax is None) else int(lmax)
    lmin = 0 if (lmin is None) else int(lmin)
    noise_spectra = deepcopy(noise)
    noise_ells = ells.copy()
    num_noise_ells = len(noise_ells)
    noise_lmin = int(noise_ells[0])
    if noise_lmin < noise_ells[0]:
        noise_lmin += 1
    noise_lmax = int(noise_ells[-1])

    if fill_vals is not None:
        if ells_to_fill is None:
            raise ValueError("If `fill_vals` is not `None`, you must also pass a list of multipoles to `ells_to_fill`")
        fill_ells = np.atleast_1d(ells_to_fill).astype(int)
        fill_ells_below_noise_lmin = fill_ells[(fill_ells < noise_lmin) & (fill_ells >= lmin)]
        fill_ells_above_noise_lmax = fill_ells[(fill_ells > noise_lmax) & (fill_ells <= lmax)]

        num_fill_ells_below = len(fill_ells_below_noise_lmin)
        num_fill_ells_above = len(fill_ells_above_noise_lmax)
        num_fill_ells = num_fill_ells_below + num_fill_ells_above
        if num_fill_ells > 0:
            noise_ells = np.array([*fill_ells_below_noise_lmin, *noise_ells, *fill_ells_above_noise_lmax], dtype=int)
            for freq in fill_vals.keys():
                filled_noise_power = np.zeros(num_noise_ells+num_fill_ells)
                filled_noise_power[num_fill_ells_below:num_fill_ells_below+num_noise_ells] = noise_spectra[freq].copy()
                if fill_val_ells is None:
                    fill_val_ells = np.arange(len(fill_vals[freq]), dtype=int)
                if num_fill_ells_below > 0:
                    loc_below = np.isin(fill_val_ells, fill_ells_below_noise_lmin)
                    filled_noise_power[:num_fill_ells_below] = fill_vals[freq][loc_below].copy()
                if num_fill_ells_above > 0:
                    loc_above = np.isin(fill_val_ells, fill_ells_above_noise_lmax)
                    filled_noise_power[-num_fill_ells_above:] = fill_vals[freq][loc_above].copy()
                noise_spectra[freq] = filled_noise_power.copy()

    coadded_noise_power = get_coadded_noise(noise_spectra)
    if interp:
        interp_ells, interp_coadded_noise = interp_spectra(noise_ells, coadded_noise_power)
        coadded_noise_ells = np.arange(lmin, lmax+1, dtype=int)
        coadded_noise = np.zeros(len(coadded_noise_ells))
        coadded_noise[np.isin(coadded_noise_ells, interp_ells)] = interp_coadded_noise[np.isin(interp_ells, coadded_noise_ells)].copy()
    else:
        loc = np.where((noise_ells >= lmin) & (noise_ells <= lmax))
        coadded_noise_ells = noise_ells[loc].copy()
        coadded_noise = coadded_noise_power[loc].copy()
    return coadded_noise_ells, coadded_noise

