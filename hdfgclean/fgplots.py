from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from pixell import colorize
from hdsims import utils, simutils, fgcatalogs, plots
from . import fgutils, fgresults

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


def plot_fgcleaned_maps_without_cmb(maps_before, maps_after, fg_label='tSZ+CIB+Radio',
                                    clims=50, dpi=250, show=True, fname=None):
    plt_freqs = sorted(list(maps_before.keys()))
    if not isinstance(clims, dict):
        plt_clims = {freq: clims for freq in plt_freqs} # single color bar range for all freqs
    plt_titles = [f'Before {fg_label} subtraction', f'Measured {fg_label}', f'After {fg_label} subtraction']
    plt_maps = []
    for freq in plt_freqs:
        plt_maps.append(maps_before[freq])
        plt_maps.append(maps_before[freq] - maps_after[freq])
        plt_maps.append(maps_after[freq])

    ncol = 3
    nrow = len(plt_freqs)
    nplt = ncol * nrow
    cax_frac = 0.05
    plt_wspace = 0.03
    plt_height = 4
    plt_width = plt_height
    fig_width = plt_width * (ncol - 1) + plt_width * (1 + plt_wspace + cax_frac)
    fig_height = nrow * plt_height

    fig, axs = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi, nrows=nrow, ncols=ncol+1, width_ratios=[1,1,1,cax_frac])
    for i in range(nrow):
        freq = plt_freqs[i]
        clim = plt_clims[freq]
        for j in range(ncol):
            if (nrow > 1) and (ncol > 1):
                plt_num = i * ncol + j
                ax = axs[i, j]
            else:
                plt_num = i + j
                ax = axs[plt_num]

            if plt_num < nplt:
                if j == ncol - 1: # put colorbar in last column
                    if (nrow > 1) and (ncol > 1):
                        cax = axs[i, j+1]
                    else:
                        cax = axs[plt_num+1]
                    fig, ax, cax = plots.plot_map(plt_maps[plt_num], fig=fig, ax=ax, cax=cax, show=False, vmin=-clim, vmax=clim, grid=False)
                    cax.grid(visible=False)
                else:
                    fig, ax, _ = plots.plot_map(plt_maps[plt_num], fig=fig, ax=ax, colorbar=False, show=False, vmin=-clim, vmax=clim, grid=False)

                ax.tick_params(labelsize=11)
                # label for the frequency:
                ax.text(0.05, 0.95, f'{simutils.round_str(freq)} GHz', fontsize=13,
                        bbox={'facecolor': 'w', 'alpha': 0.75, 'lw': 1.25, 'boxstyle': 'round'},
                        ha='left', va='top', transform=ax.transAxes)

                # only place x-axis ticks and labels in last row:
                first_row = (i == 0)
                last_row = (i == (nrow - 1))
                if first_row:
                    ax.set_title(plt_titles[j], fontsize=13)
                if last_row:
                    ax.set_xlabel('RA [degrees]', fontsize=12)
                else:
                    xlims = ax.get_xlim()
                    ax.set_xticks(ax.get_xticks())
                    ax.set_xticklabels(['' for tick in ax.get_xticklabels()])
                    ax.set_xlim(xlims)
                    ax.set_xlabel('')
                # only place y-axis ticks and labels in first column:
                if j > 0:
                    ylims = ax.get_ylim()
                    ax.set_yticks(ax.get_yticks())
                    ax.set_yticklabels(['' for tick in ax.get_yticklabels()])
                    ax.set_ylim(ylims)
                    ax.set_ylabel('')
                else:
                    ax.set_ylabel('dec [degrees]', fontsize=12)

    plt.subplots_adjust(wspace=plt_wspace, hspace=plt_wspace)

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, axs


def plot_fgcleaned_maps_with_cmb(map_before_with_cmb, map_after_with_cmb,
                                 maps_before_without_cmb, maps_after_without_cmb,
                                 cmb_map_freq, maps_without_cmb_label='CMB & kSZ', 
                                 clims=50, cmb_clim=300, dpi=250, show=True, fname=None):
    freqs = sorted(list(maps_before_without_cmb.keys()))
    freqs_without_cmb = [freq for freq in freqs if (freq != cmb_map_freq)]
    plt_freqs = [cmb_map_freq, *freqs_without_cmb]
    if not isinstance(clims, dict):
        clims = {freq: clims for freq in plt_freqs} # single color bar range for all freqs
        
    plt_titles = ['Before foreground removal', 'After foreground removal']
    plt_maps = [map_before_with_cmb, map_after_with_cmb]
    plt_labels = [f'{cmb_map_freq} GHz', f'{cmb_map_freq} GHz']
    plt_clims = [cmb_clim, cmb_clim]
    for freq in plt_freqs:
        plt_maps.append(maps_before_without_cmb[freq])
        plt_maps.append(maps_after_without_cmb[freq])
        for i in range(2):
            plt_labels.append(f'{freq} GHz')
            plt_clims.append(clims[freq])
      
    ncol = 2
    nrow = len(plt_freqs) + 1
    nplt = ncol * nrow
    cax_frac = 0.05
    plt_wspace = 0.03
    plt_height = 4
    plt_width = plt_height 
    fig_width = plt_width * (ncol - 1) + plt_width * (1 + plt_wspace + cax_frac)
    fig_height = nrow * plt_height
    
    fig, axs = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi, nrows=nrow, ncols=ncol+1, width_ratios=[1,1,cax_frac])
    for i in range(nrow):
        for j in range(ncol):
            if (nrow > 1) and (ncol > 1):
                plt_num = i * ncol + j
                ax = axs[i, j]
            else:
                plt_num = i + j
                ax = axs[plt_num]
                
            clim = plt_clims[plt_num]
            if plt_num < nplt:
                if j == ncol - 1: # put colorbar in last column
                    cax = axs[i, j+1]
                    fig, ax, cax = plots.plot_map(plt_maps[plt_num], fig=fig, ax=ax, cax=cax, show=False, vmin=-clim, vmax=clim, grid=False)
                    cax.grid(visible=False)
                else:
                    fig, ax, _ = plots.plot_map(plt_maps[plt_num], fig=fig, ax=ax, colorbar=False, show=False, vmin=-clim, vmax=clim, grid=False)
                
                ax.tick_params(labelsize=11)
                # label for the frequency:
                ax.text(0.05, 0.95, plt_labels[plt_num], fontsize=13,
                        bbox={'facecolor': 'w', 'alpha': 0.75, 'lw': 1.25, 'boxstyle': 'round'}, 
                        ha='left', va='top', transform=ax.transAxes)
                if i > 0:
                    ax.text(0.5, 0.05, f'{maps_without_cmb_label} subtracted for clarity', 
                            bbox=dict(facecolor='w', alpha=0.75), fontsize=10,
                            horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
                # only place x-axis ticks and labels in last row:
                if i < nrow - 1: 
                    xlims = ax.get_xlim()
                    ax.set_xticks(ax.get_xticks())
                    ax.set_xticklabels(['' for tick in ax.get_xticklabels()])
                    ax.set_xlim(xlims)
                    ax.set_xlabel('')
                    if i == 0: # put titles above first row
                        ax.set_title(plt_titles[j], fontsize=14)
                else: 
                    ax.set_xlabel('RA [degrees]', fontsize=12)
                # only place y-axis ticks and labels in first column:
                if j > 0: 
                    ylims = ax.get_ylim()
                    ax.set_yticks(ax.get_yticks())
                    ax.set_yticklabels(['' for tick in ax.get_yticklabels()])
                    ax.set_ylim(ylims)
                    ax.set_ylabel('')
                else: 
                    ax.set_ylabel('dec [degrees]', fontsize=12)
                    
    plt.subplots_adjust(wspace=plt_wspace, hspace=plt_wspace)
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, axs


def _axis_limits(plot_data, log=False, pad_frac=0.01):
    '''
    plot_data : 1d array of data being plotted (on x / y axis)
    log : whether it will be plotted on a log scale
    pad_frac : controls amount of padding to add beyond min/max of `plot_data`
        e.g. if all `plot_data` values are positive, default range is [0.99 * min(plot_data), 1.01*max(plot_data)]
    '''
    if log:
        data_min = np.min(np.log10(plot_data[plot_data > 0]))
        data_max = np.max(np.log10(plot_data[plot_data > 0]))
    else:
        data_min = np.min(plot_data)
        data_max = np.max(plot_data)
    min_fact = (1 - pad_frac) if (data_min > 0) else (1 + pad_frac)
    max_fact = (1 + pad_frac) if (data_max > 0) else (1 - pad_frac)
    ax_min = min_fact * data_min
    ax_max = max_fact * data_max
    if log:
        ax_min = 10**ax_min
        ax_max = 10**ax_max
    return ax_min, ax_max


def _setup_multifreq_spectra_plot(freqs=None, spec_type='dl', lmin=None, lmax=20000, logy=True,
                                  plot_cmb=False, cmb_spectra=None, cmb_label='Lensed CMB',
                                  freq_label=True, freq_label_pos=None, freq_label_va='top', freq_label_ha='left', 
                                  fig=None, axs=None, figsize=None, dpi=500,
                                  labelsize=11, legendsize=9, ticksize=10, textsize=12, lw=1.25,
                                 ):
    """
    cmb_spectra is a dict w/ keys 'ells' and 'dltt' and/or 'cltt'
        if cmb_spectra passed but plot_cmb is False, cmb_spectra is used to set the y-axis limits
        if cmb_spectra is None, returned ylims will be (None, None)
        
    freq_label_pos is (x, y) position of label for freq (in axis units) ; if `None`, use default
    
    returns fig, axs, legend_kwargs, ylims
    """
    ylabel = r"$C_\ell$ [$\mu$K$^2$]" if ('cl' in spec_type.lower()) else r"$\frac{\ell(\ell+1)}{2\pi} C_\ell$ [$\mu$K$^2$]"
    legend_kwargs = {'fontsize': legendsize, 'loc': 'upper right', 'bbox_to_anchor': (1, 1.01),
                     'markerfirst': False, 'alignment': 'right', 
                     'handlelength': 1.7, 'handletextpad': 0.65, 'frameon': False}
    if freq_label_pos is None:
        flabel_x = 0.12
        flabel_y = 0.95
    else:
        flabel_x, flabel_y = freq_label_pos
    
    if cmb_spectra is not None:
        cmb_ells = cmb_spectra['ells']
        cmb_power = cmb_spectra[f'{spec_type[:2].lower()}tt']
        if lmax is not None:
            cmb_ells, cmb_power = utils.trim_spectrum_ell_range(cmb_ells, cmb_power, lmin=lmin, lmax=lmax)
        # get y-axis limits
        ylims = _axis_limits(cmb_power, log=logy, pad_frac=0.02)
    else:
        ylims = (None, None)
        
    # set up the plot
    nplt = 1 if (freqs is None) else len(freqs)
    if fig is None:
        if figsize is None:
            figsize = (nplt*4, 4) if (nplt > 1) else (4.5, 4) 
        fig, axs = plt.subplots(figsize=figsize, ncols=nplt, dpi=dpi, layout='tight')
    elif axs is None:
        axs = fig.subplots(ncols=nplt)
    if nplt == 1:
        axs = [axs] # put single axis into a list
    
    for i in range(nplt):
        axs[i].margins(0.01)
        if logy:
            axs[i].set_yscale('log')  
            
        if plot_cmb and (cmb_spectra is not None):
            axs[i].plot(cmb_ells, cmb_power, lw=lw, color='k', label=cmb_label, zorder=2)

        axs[i].set_xlabel(r'Multipole, $\ell$', fontsize=labelsize)
        axs[i].set_ylabel(ylabel, fontsize=labelsize, labelpad=2.0)
        axs[i].tick_params(which='both', right=True, labelsize=ticksize)
        axs[i].grid(alpha=0.1)
        if freq_label and (freqs is not None):
            axs[i].text(flabel_x, flabel_y, f'{freqs[i]} GHz', va=freq_label_va, ha=freq_label_ha, 
                        transform=axs[i].transAxes, color='k', fontsize=textsize, 
                        bbox={'facecolor': 'w', 'edgecolor': 'k', 'boxstyle': 'round', 'pad': 0.25})

    return fig, axs, legend_kwargs, ylims


def _plot_multifreq_spectra(freqs, spectra, spec_type='dl', lmin=None, lmax=20000, show=True, fname=None,
                            spectra_index_for_ylims=None, logy=True, freq_label=True, freq_label_pos=None, 
                            freq_label_va='top', freq_label_ha='left', 
                            fig=None, axs=None, figsize=None, dpi=500, 
                            labelsize=11, legendsize=9, ticksize=10, textsize=12):
    """
    spectra is a dict w/ key for each freq
        spectra[freq] contains a list ; each element in list is tuple of (array_of_ells, array_of_power_spectrum, dict_of_kwargs)
        e.g. `spectra[90][0] = (ells, cls, kwargs)` is plotted with `ax.plot(ells, cls, **kwargs)`
        
    if spectra_index_for_ylims is not None: 
        set y-axis limits based on the range of the power spectrum at index `spectra_index_for_ylims` in the list `spectra[freq]` at each freq
    
    cmb_spectra is a dict w/ keys 'ells' and 'dltt' and/or 'cltt'
        if cmb_spectra passed but plot_cmb is False, cmb_spectra is used to set the y-axis limits
        if cmb_spectra is None, returned ylims will be (None, None)
        
    freq_label_pos is (x, y) position of label for freq (in axis units) ; if `None`, use default
    
    returns fig, axs, legend_kwargs
    """
    fig, axs, legend_kwargs, _ = _setup_multifreq_spectra_plot(freqs=freqs, spec_type=spec_type, lmin=lmin, lmax=lmax, 
                                                               freq_label=freq_label, freq_label_pos=freq_label_pos,
                                                               freq_label_va=freq_label_va, freq_label_ha=freq_label_ha,
                                                               logy=logy, figsize=figsize, dpi=dpi, 
                                                               labelsize=labelsize, legendsize=legendsize, 
                                                               ticksize=ticksize, textsize=textsize,
                                                               fig=fig, axs=axs)
    for i, freq in enumerate(freqs):
        ylims = (None, None)
        for j, (xdata, ydata, kwargs) in enumerate(spectra[freq]):
            xdata, ydata = utils.trim_spectrum_ell_range(xdata, ydata, lmin=lmin, lmax=lmax)
            if j == spectra_index_for_ylims:
                ylims = _axis_limits(ydata, log=logy, pad_frac=0.02)
            axs[i].plot(xdata, ydata, **kwargs)
        axs[i].set_ylim(ylims)
        axs[i].legend(**legend_kwargs)
            
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, axs, legend_kwargs


def plot_fgcleaned_spectra(freqs, sim_spectra, previous_spectra, spec_type='dl',
                           sim_fg_components=['ksz', 'tsz', 'cib', 'radio'], sim_spectra_keys=['tsz', 'cib_radio'],
                           previous_spectra_keys=['cmb', 'noise', 'ksz'], plot_total_fg_noise=True, plot_total_fg=False,
                           show=True, fname=None, plt_lmax=20000, lw=1.25, sim_lw=1.75, 
                           labelsize=11, legendsize=9, ticksize=10, figsize=None, dpi=500, 
                           legend_labels={}, previous_spectra_colors={}, sim_spectra_colors={}):
    skey = f'{spec_type}tt'
    # list of all possible spectra, in correct order for plotting:
    total_spectra_order = ['total_fg', 'total_fg_noise']
    plt_spectra_order = ['cmb', 'noise', 'ksz', 'tsz', 'cib', 'radio', 'cib_radio', *total_spectra_order]
    # whether to plot total FG, with and/or without noise:
    plot_total_spectra = {'total_fg': plot_total_fg, 'total_fg_noise': plot_total_fg_noise}
    total_keys = [key for key in total_spectra_order if plot_total_spectra[key]]
    # which sim-based spectra and previous estimated spectra to plot:
    sim_spec_keys = [key for key in plt_spectra_order if (key in [*sim_spectra_keys, *total_keys])]
    prev_spec_keys = [key for key in plt_spectra_order if (key in [*previous_spectra_keys, *total_keys])]
    theo_keys = [key for key in prev_spec_keys if (key not in sim_spec_keys)]
    # list of all keys being used:
    all_plt_keys = list(set([*prev_spec_keys, *sim_spec_keys])) # all keys for spectra to be plotted
    ordered_plt_keys = [key for key in plt_spectra_order if (key in all_plt_keys)] # put them in order
    component_keys = [key for key in ordered_plt_keys if ('total_fg' not in key)] # for individual components
    
    # defaults for plotting
    default_labels = {'cmb': 'Lensed CMB', 'noise': 'Inst. noise', 'total_fg': 'Total FG',  'total_fg_noise': 'Total FG + Inst. Noise', 
                      'tsz': 'tSZ FG', 'ksz': 'kSZ FG', 'cib': 'CIB', 'radio': 'Radio', 'cib_radio': 'CIB + Radio FG'}
    default_colors = {'cmb': 'k', 'noise': 'tab:gray', 'total_fg_noise': '#ff7070', 'total_fg': '#ab95de', 
                      'tsz': '#97bd68', 'ksz': '#e6a22e', 'cib_radio': '#74b6e3'}
    default_sim_colors = {'total_fg_noise': '#b50202', 'total_fg': 'tab:purple', 'tsz': '#6a8a00', 'cib_radio': '#0261a1'}
    
    # if plotting a non-default set of spectra, update legend labels, and
    #  make sure we are comparing correct set of "previous" spectra to the "sim" spectra
    prev_spectra = deepcopy(previous_spectra)
    all_fg_components = ['tsz', 'ksz', 'cib', 'radio']
    if len(sim_fg_components) < len(all_fg_components):
        default_labels['total_fg'] = " + ".join([default_labels[component] for component in sim_fg_components])
        default_labels['total_fg_noise'] = f"{default_labels['total_fg']} + Inst. Noise"
        for freq in freqs:
            prev_spectra[freq]['total_fg'][skey] = np.zeros(len(prev_spectra[freq]['total_fg'][skey]))
            for component in sim_fg_components:
                prev_spectra[freq]['total_fg'][skey] += prev_spectra[freq][component][skey]
            prev_spectra[freq]['total_fg_noise'][skey] = prev_spectra[freq]['total_fg'][skey] + prev_spectra[freq]['noise'][skey]
    
    # override defaults if other values were passed
    labels = {**default_labels, **legend_labels}
    colors = {**default_colors, **previous_spectra_colors}
    sim_colors = {**default_sim_colors, **sim_spectra_colors}
    lines = {key: '--' for key in all_plt_keys}
    for key in ['cmb', *total_keys]:
        lines[key] = '-'
    
    # make lists of data to plot and kwargs for spectra at each freq:
    plt_spectra = {}
    spectra_index_for_ylims = None
    for freq in freqs:
        plt_spectra[freq] = []
        # first, plot all spectra in `previous_spectra` that don't have a corresponding key in `sim_spectra`; 
        # these should all be theory curves (cmb, ksz, white noise):
        theo_spectra_kwargs = {'lw': lw, 'zorder': 2}
        for key in theo_keys:
            if key == 'cmb':
                spectra_index_for_ylims = len(plt_spectra[freq])
            plt_spectra[freq].append((prev_spectra[freq][key]['ells'], prev_spectra[freq][key][skey], 
                                      {**theo_spectra_kwargs, 'ls': lines[key], 'color': colors[key], 'label': labels[key]}))
        # plot the sim-based spectra in `sim_spec_keys`, then plot the corresponding previous estimate
        for key in sim_spec_keys:
            if (key == 'cmb') and (spectra_index_for_ylims is None):
                spectra_index_for_ylims = len(plt_spectra[freq])
            plt_sim_lw = sim_lw - 0.25 if ('total_fg' not in key) else sim_lw
            label = f'{labels[key]} (Sim-based)'
            plt_spectra[freq].append((sim_spectra[freq][key]['ells'], sim_spectra[freq][key][skey], 
                                      {'lw': plt_sim_lw, 'ls': lines[key], 'color': sim_colors[key], 'label': label, 'zorder': 3}))
            if key in prev_spec_keys:
                plt_lw = lw if ('total_fg' not in key) else lw + 0.5
                label = f'{labels[key]} (Previous)' if (key in sim_spec_keys) else labels[key]
                plt_spectra[freq].append((prev_spectra[freq][key]['ells'], prev_spectra[freq][key][skey], 
                                          {'lw': plt_lw, 'ls': lines[key], 'color': colors[key], 'label': label, 'zorder': 2}))
                
    # make the plot
    plt_output = _plot_multifreq_spectra(freqs, plt_spectra, spec_type=spec_type, lmax=plt_lmax, 
                                         logy=True, spectra_index_for_ylims=spectra_index_for_ylims, show=show, fname=fname,
                                         figsize=figsize, dpi=dpi, labelsize=labelsize, legendsize=legendsize, ticksize=ticksize)
    if not show:
        return plt_output


def spectral_index_plot(fluxs1, fluxs2, freq1, freq2, component_name=None, true_index=None, 
                        xticks=None, yticks=None, labelsize=11, ticksize=10, legendsize=9, cmap='rainbow', 
                        fig=None, ax=None, cax=None, dpi=500, plot_width=4, show=True, fname=None):
    '''
    extrap fluxes *from* freq1 *to* freq2
    '''
    indices = fgutils.get_spectral_index(fluxs1, fluxs2, freq1, freq2)
    measured_index = np.mean(indices)

    # set up the plot:
    legend_title = 'Mean spectral index' if (component_name is None) else f'Mean {component_name} spectral index'
    lowflux_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    highflux_ticks = [0.1, 0.5, 1, 2, 5, 10, 20,  50, 100, 200, 500]
    if xticks is None:
        xticks = highflux_ticks if (np.max(fluxs1) > 20) else lowflux_ticks
    if yticks is None:
        yticks = highflux_ticks if (np.max(fluxs2) > 20) else lowflux_ticks
    
    cax_frac = 0.05
    if fig is None:
        fig, (ax, cax) = plt.subplots(figsize=(plot_width*(1+cax_frac), plot_width), ncols=2, 
                                      width_ratios=[1/(1+cax_frac), cax_frac/(1+cax_frac)], dpi=dpi)
    elif ax is None:
        ax, cax = fig.subplots(ncols=2, width_ratios=[1/(1+cax_frac), cax_frac/(1+cax_frac)])
    elif cax is None:
        cax = ax
    ax.set_xscale('log')
    ax.set_yscale('log')

    # plot individual fluxs:
    pts = ax.scatter(fluxs1, fluxs2, c=indices, cmap=cmap, s=20, linewidths=0.25, edgecolors=(1,1,1,0.25))
    cbar = fig.colorbar(pts, cax=cax)
    cbar.set_label('Index', fontsize=labelsize)

    # plot line for measured index, and true index if it was provided:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if true_index is not None:
        ax.plot([xmin, xmax], fgutils.extrapolate_flux(np.array([xmin, xmax]), freq1, freq2, true_index), 
                color='#fa5798', lw=1.5, label='True index')
    ax.plot([xmin, xmax], fgutils.extrapolate_flux(np.array([xmin, xmax]), freq1, freq2, measured_index), 
            color='k', lw=1.5, label='Measured index')
    if true_index is not None:
        ax.fill_between([xmin, xmax], fgutils.extrapolate_flux(np.array([xmin, xmax]), freq1, freq2, 0.95 * true_index), 
                        y2=fgutils.extrapolate_flux(np.array([xmin, xmax]), freq1, freq2, 1.05 * true_index), 
                        color=(0.5, 0.5, 0.5, 0.2), edgecolor='none', zorder=-3, label=r'True index $\pm$ 5%')
    ax.legend(title=legend_title, loc='upper left', bbox_to_anchor=(0, 1), frameon=False, fontsize=legendsize)

    ax.set_yticks(yticks)
    ax.set_yticklabels([simutils.round_str(tick) for tick in yticks])
    ax.set_xticks(xticks)
    ax.set_xticklabels([simutils.round_str(tick) for tick in xticks])
    ax.tick_params(labelsize=ticksize)
    cax.tick_params(labelsize=ticksize)
    
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel(f'{freq1} GHz flux [mJy]', fontsize=labelsize)
    ax.set_ylabel(f'{freq2} GHz flux [mJy]', fontsize=labelsize)

    ax.grid(visible=False)
    cax.grid(visible=False)
    plt.subplots_adjust(wspace=0.05)

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, ax, cax, cbar, pts


def measured_vs_true_flux_plot(matched_sources_catalog, freq=90,
                               true_flux_col='true_fluxmJy', measured_flux_col='fluxmJy', cbar_col='SNR', 
                               min_snr=None, # min SNR of measured source fluxes
                               min_flux=None, max_flux=None, # for axis ranges
                               cmap='rainbow', cbar_label=None, cbar_ticks=[5, 10, 20, 50, 100, 200, 500, 1000],
                               labelsize=11, ticksize=10, legendsize=9, textsize=12, plot_width=4, dpi=500,
                               fig=None, ax=None, cax=None, show=True, fname=None):
    catalog = matched_sources_catalog.copy()
    # min. SNR of measured fluxes:
    if min_snr is not None:
        catalog = catalog[catalog['SNR'].ge(min_snr)]
    else:
        min_snr = round(np.min(catalog['SNR'].values), 2)
    # approx. flux corresponding to the min SNR:
    min_cat_flux = np.min(catalog['fluxmJy'].values)
    min_snr_flux = round(min_cat_flux,2)
    if (min_cat_flux - min_snr_flux) > 0.0025:
        min_snr_flux += 0.01
        
    xlabel = f'True {simutils.round_str(freq)} GHz flux [mJy]'
    ylabel = f'Measured {simutils.round_str(freq)} GHz flux [mJy]'
    cbar_label = cbar_label if (cbar_label is not None) else cbar_col
    cbar_ticks = np.asarray(cbar_ticks) if (cbar_ticks is not None) else np.array([5, 10, 20, 50, 100, 200, 500, 1000])
    flux_tick_formatter = lambda x, pos : simutils.round_str(x, n=3)
    cbar_tick_formatter = lambda x, pos : simutils.round_str(x, n=2)
    vmin = np.min(catalog[cbar_col].values) 
    vmax = np.max(catalog[cbar_col].values)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    
    cax_frac = 0.05 # fraction of plot width used for color bar
    if fig is None:
        fig, (ax, cax) = plt.subplots(figsize=(plot_width*(1+cax_frac), plot_width), ncols=2, 
                                      width_ratios=[1/(1+cax_frac), cax_frac/(1+cax_frac)], dpi=dpi)
    elif ax is None:
        ax, cax = fig.subplots(ncols=2, width_ratios=[1/(1+cax_frac), cax_frac/(1+cax_frac)])
    elif cax is None:
        cax = ax
    ax.set_xscale('log')
    ax.set_yscale('log')
        
    pts = ax.scatter(catalog[true_flux_col].values, catalog[measured_flux_col].values, c=catalog[cbar_col].values, 
                     norm=norm, cmap=cmap, s=5)
    
    # get axis limits and then set same limits for both axes
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if min_flux is None:
        min_flux = min([xmin, ymin])
    if max_flux is None:
        max_flux = max([xmax, ymax])
        
    # shade region where |true flux - measured flux| <= `min_snr_flux` mJy (which is approx. the min. measured flux w/ SNR >= `min_snr`)
    flux_vals = np.logspace(np.log10(0.95*min_flux), np.log10(1.05*max_flux), 500)
    ax.fill_between(flux_vals, flux_vals - min_snr_flux, y2=flux_vals + min_snr_flux, 
                    color=(0.5, 0.5, 0.5, 0.2), edgecolor='none', zorder=-3) # shading under points
    ax.fill_between(flux_vals, flux_vals - min_snr_flux, y2=flux_vals + min_snr_flux, 
                    color=(1,1,1,0), edgecolor='w', ls=':', lw=0.75, zorder=4) # line above points
    # for legend label (fill between y=0 and some ymax that is below the `min_flux` on the y-axis):
    ax.fill_between(flux_vals, min_flux/10, color=(0.5, 0.5, 0.5, 0.2), edgecolor='w', ls=':', lw=0.75, zorder=-3, 
                    label = r'|true - measured| $\leq$ %s mJy' % simutils.round_str(min_snr_flux,n=3))
    # line and label for measured flux = `min_snr_flux` mJy
    ax.axhline(y=min_snr_flux, color='k', lw=0.5, ls='-.')
    ax.text(0.8*xmax, 1.1*min_snr_flux, r'SNR of %d $\approx$ %s mJy' % (min_snr, simutils.round_str(min_snr_flux,n=3)), ha='right')
    # measured flux = true flux line
    ax.plot(flux_vals, flux_vals, color='k', lw=0.8, label='measured = true', zorder=5)
    
    ax.legend(fontsize=legendsize, frameon=False, loc='upper left', bbox_to_anchor=(0, 0.9))
    ax.text(0.35, 0.95, f'{simutils.round_str(freq)} GHz', color='k', fontsize=textsize, 
            bbox={'facecolor': 'w', 'edgecolor': 'k', 'boxstyle': 'round', 'pad': 0.25},
            va='top', ha='left', transform=ax.transAxes)
    ax.grid(alpha=0.1)
    ax.set_xlim([min_flux, max_flux])
    ax.set_ylim([min_flux, max_flux])
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(flux_tick_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(flux_tick_formatter))

    cbar = fig.colorbar(pts, cax=cax, ticks=cbar_ticks[cbar_ticks <= vmax])
    cax.grid(visible=False)
    cax.set_ylabel(cbar_label, fontsize=labelsize)
    cax.tick_params(labelsize=ticksize)
    cax.yaxis.set_major_formatter(mticker.FuncFormatter(cbar_tick_formatter))

    plt.subplots_adjust(wspace=0.05)
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, ax, cax, cbar, pts


def true_flux_histogram_plot(matched_bright_true_fluxes, matched_extrap_true_fluxes=None, 
                             unmatched_true_fluxes=None, unmatched_measured_fluxes=None,
                             freq=None, min_detected_snr=None, detected_flux_lim=None, detected_flux_label=None,
                             num_flux_bins=250, xlabel='True flux [mJy]', ylabel='Number of sources',
                             fig=None, ax=None, plot_width=4.5, plot_height=4.25, dpi=500,
                             labelsize=11, ticksize=10, legendsize=8.75, textsize=12, show=True, fname=None):
    legend_kwargs = {'fontsize': legendsize, 'loc': 'upper right', 'bbox_to_anchor': (1, 0.87), 
                     'frameon': False, 'borderpad': 0.25, 'borderaxespad': 0.15,
                     'handlelength': 1, 'handletextpad': 0.4, 'handleheight': 1,
                     'markerfirst': False, 'labelspacing': 0.75}
    tick_formatter = lambda x, pos : simutils.round_str(x, n=4)
    bright_srcs_label = 'Detected sources' if (matched_extrap_true_fluxes is None) else 'Bright sources'
    if min_detected_snr is not None:
        bright_srcs_label = f"{bright_srcs_label} with SNR > {simutils.round_str(min_detected_snr)}"
    
    fluxes_to_plot = [matched_bright_true_fluxes]
    colors = ['tab:olive']
    labels = [bright_srcs_label]
    if matched_extrap_true_fluxes is not None:
        fluxes_to_plot.append(matched_extrap_true_fluxes)
        colors.append('tab:pink')
        labels.append("Dim sources extrapolated\nfrom another frequency")
    if unmatched_true_fluxes is not None:
        fluxes_to_plot.append(unmatched_true_fluxes)
        colors.append('tab:blue')
        labels.append("Sources not detected")
    min_flux = min([np.min(fluxes) for fluxes in fluxes_to_plot])
    max_flux = max([np.max(fluxes) for fluxes in fluxes_to_plot])
    flux_bins = np.logspace(np.log10(min_flux), np.log10(max_flux), num_flux_bins)
    
    # make the plot:
    if fig is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=dpi)
        else:
            ax = fig.subplots()
    # plot filled histograms   
    ax.hist(fluxes_to_plot, bins=flux_bins, stacked=True, label=labels, color=colors, alpha=0.65)
    # plot outline around each histogram
    ax.hist(fluxes_to_plot, bins=flux_bins, stacked=True, histtype='step', fill=False,  
            edgecolor='k', lw=0.25, zorder=4, alpha=0.5)
    # plot false detections:
    if unmatched_measured_fluxes is not None:
        ax.hist(unmatched_measured_fluxes, bins=flux_bins, histtype='step', label=f"False detections", 
                fill=False, edgecolor='tab:red', lw=1, zorder=5)
        ax.axhline(color='k', lw=1, zorder=5) # prevent red line appearing at bottom for bins with zero false detections
    # plot detected flux limit, if provided
    if detected_flux_lim is not None:
        ax.axvline(x=detected_flux_lim, color='k', ls=':', lw=1, label=detected_flux_label)
    # finalize the plot:
    ax.margins(0.025)
    ax.set_xscale('log')
    ax.grid(alpha=0.1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(tick_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(tick_formatter))
    ax.tick_params(which='both', labelsize=ticksize)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.legend(**legend_kwargs)
    if freq is not None:
        ax.text(0.9, 0.96, f'{freq} GHz', color='k', fontsize=textsize, 
                bbox={'facecolor': 'w', 'edgecolor': 'k', 'boxstyle': 'round', 'pad': 0.25},
                va='top', ha='right', transform=ax.transAxes)

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, ax, legend_kwargs


def true_flux_histogram_plots(freqs, matched_bright_true_fluxes, matched_extrap_true_fluxes=None, 
                              unmatched_true_fluxes=None, unmatched_measured_fluxes=None,
                              min_detected_snr=None, detected_flux_lims=None, detected_flux_label=None,
                              num_flux_bins=250, xlabel='True flux [mJy]', ylabel='Number of sources',
                              equal_ylims=True, fig=None, axs=None, plot_width=4.5, plot_height=4.25, dpi=500,
                              labelsize=11, ticksize=10, legendsize=8.75, textsize=12, show=True, fname=None):
    '''
    matched_bright_true_fluxes, detected_flux_lims, etc should be dicts w/ keys for each freq
    if axs provided, should be a list in same order as freqs
    '''
    if matched_extrap_true_fluxes is None:
        matched_extrap_true_fluxes = {freq: None for freq in freqs}
    if unmatched_true_fluxes is None:
        unmatched_true_fluxes = {freq: None for freq in freqs}
    if unmatched_measured_fluxes is None:
        unmatched_measured_fluxes = {freq: None for freq in freqs}
    if detected_flux_lims is None:
        detected_flux_lims = {freq: None for freq in freqs}
    
    nplt = len(freqs)
    if fig is None:
        fig, axs = plt.subplots(figsize=(plot_width*nplt, plot_height), ncols=nplt, dpi=dpi)
    elif axs is None:
        axs = fig.subplots(ncols=nplt)
    if nplt == 1:
        axs = [axs] # put single axis into a list
    
    for i, freq in enumerate(freqs):
        fig, axs[i], legend_kwargs =  true_flux_histogram_plot(matched_bright_true_fluxes[freq],
                                                               matched_extrap_true_fluxes=matched_extrap_true_fluxes[freq], 
                                                               unmatched_true_fluxes=unmatched_true_fluxes[freq], 
                                                               unmatched_measured_fluxes=unmatched_measured_fluxes[freq],
                                                               freq=freq, detected_flux_lim=detected_flux_lims[freq],
                                                               detected_flux_label=detected_flux_label, 
                                                               min_detected_snr=min_detected_snr, 
                                                               num_flux_bins=num_flux_bins, xlabel=xlabel, ylabel=ylabel,
                                                               plot_width=plot_width, plot_height=plot_height, dpi=dpi,
                                                               labelsize=labelsize, ticksize=ticksize, 
                                                               legendsize=legendsize, textsize=textsize,
                                                               fig=fig, ax=axs[i], show=False)
    if equal_ylims:
        ymin = min([ax.get_ylim()[0] for ax in axs])
        ymax = max([ax.get_ylim()[1] for ax in axs])
        for ax in axs:
            ax.set_ylim([ymin, ymax])
    plt.subplots_adjust(wspace=0.25)    
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, axs, legend_kwargs


def cluster_mass_vs_redshift_plot(matched_true_clusters, unmatched_true_clusters, 
                                  plot_mass_at_completeness=True, completeness=0.99, zkey='z', mass_key='M500', 
                                  mass_label=r'True $M_{500\mathrm{c}}~[M_{\odot}]$', redshift_label=r'True $z$',
                                  mass_axis_lims=[1e12, 2e15], redshift_axis_lims=None,
                                  labelsize=13, ticksize=12, dpi=500, figsize=(5,5), show=True, fname=None):
    legend_kwargs = {'loc': 'upper right', 'edgecolor': (0,0,0,0.25), 'markerscale': 1.5, 
                     'borderpad': 0.3, 'borderaxespad': 0.25, 'handlelength': 1.25, 'handletextpad': 0.5}
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_yscale('log')
    ax.plot(matched_true_clusters[zkey].values, matched_true_clusters[mass_key].values, '.', 
            markersize=4,  color='#ff7070', alpha=0.8, mew=0.5, mec='#b50202', label='True clusters detected')
    ax.plot(unmatched_true_clusters[zkey].values, unmatched_true_clusters[mass_key].values,  '.',  
            markersize=4, color='tab:cyan', alpha=0.8, mew=0.5, mec='tab:blue', label='True clusters not detected')
    if plot_mass_at_completeness:
        all_true_clusters = fgcatalogs.combine_catalogs([matched_true_clusters.copy(), unmatched_true_clusters.copy()])
        cluster_masses, cluster_completeness, _, _, _ = fgresults.get_cluster_completeness(all_true_clusters, 
                                                                                           matched_true_clusters, 
                                                                                           num_mass_bins=100, 
                                                                                           mass_col=mass_key)
        mass_limit = np.min(cluster_masses[cluster_completeness >= completeness])
        p = int(np.log10(mass_limit))
        num = round(mass_limit / (10**p))
        completeness_label = f"Mass for {simutils.round_str(100*completeness)}% "+r"completion $\approx$ %s$\times 10^{%d}$ $M_{\odot}$" % (simutils.round_str(num), p)
        ax.axhline(y=(num * 10**p), color='k', ls='--', lw=1.5,  label=completeness_label)
    ax.grid(alpha=0.1)
    ax.legend(**legend_kwargs)
    ax.set_ylabel(mass_label, fontsize=labelsize)
    ax.set_xlabel(redshift_label, fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    if redshift_axis_lims is not None:
        ax.set_xlim(redshift_axis_lims)
    if mass_axis_lims is not None:
        ax.set_ylim(mass_axis_lims)
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, ax, legend_kwargs


def cluster_completeness_per_mass_redshift_plot(all_true_clusters, matched_true_clusters, 
                                                num_mass_bins=15, mass_bin_edges=None, 
                                                zmin=0, zmax=3, num_z_bins=6, z_col='z',
                                                min_M500=1e13, max_M500=1.5e14, mass_col='M500', # for axis limits
                                                interp=True, npts_interp=50, ntimes_smooth=3,
                                                figsize=(4,4), dpi=500, labelsize=12, legendsize=11, colors=None, 
                                                xlabel=r'$M_{500\mathrm{c}}~[M_{\odot}]$', ylabel='Completeness', 
                                                fig=None, ax=None, show=True, fname=None):
    '''
    masses are in units of solar masses
    colors : 1st color used for highest z bin
    '''
    # we are only interested in mass bins above `min_M500`, but need to keep some lower bins for interpolation,
    # so remove very low-mass clusters from catalogs:
    true_massive_clusters = all_true_clusters[all_true_clusters[mass_col].ge(min_M500/10)].copy()
    matched_massive_clusters = matched_true_clusters[matched_true_clusters[mass_col].ge(min_M500/10)].copy()

    z_bin_edges = np.linspace(zmin, zmax, num_z_bins+1)
    z_bin_ctrs = (z_bin_edges[1:] + z_bin_edges[:-1]) / 2
    completeness_per_z = {}
    masses_per_z = {}
    for j, (zmin, zmax, z_ctr) in enumerate(zip(z_bin_edges[:-1], z_bin_edges[1:], z_bin_ctrs)):
        true_clusters_in_zbin = true_massive_clusters[true_massive_clusters[z_col].between(zmin, zmax)].copy()
        matched_true_clusters_in_zbin = matched_massive_clusters[matched_massive_clusters[z_col].between(zmin, zmax)].copy()
        masses, completeness, _, _, _ = fgresults.get_cluster_completeness(true_clusters_in_zbin, matched_true_clusters_in_zbin, 
                                                                             num_mass_bins=num_mass_bins, mass_bin_edges=mass_bin_edges,
                                                                             mass_col=mass_col, cumulative=False)
        if interp:
            masses_per_z[z_ctr] = np.logspace(np.log10(masses[0]), np.log10(masses[-1]), npts_interp)
            completeness_per_z[z_ctr] = interp1d(masses, completeness)(masses_per_z[z_ctr])
        else:
            masses_per_z[z_ctr] = masses
            completeness_per_z[z_ctr] = completeness
        for i in range(ntimes_smooth):
            smoothed_completeness = np.convolve(completeness_per_z[z_ctr], np.ones(3), 'valid') / 3
            completeness_per_z[z_ctr] = np.array([0, *smoothed_completeness, 1])
    
    legend_kwargs = {'frameon': False, 'fontsize': legendsize, 'reverse': True}
    # if using default z bins, also use default set of colors
    if (colors is None) and (num_z_bins <= 6):
        colors = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:blue', 'tab:purple']
    # plot from highest to lowest redshift
    plt_z_bin_edges = sorted(z_bin_edges)[::-1]
    plt_z_bin_ctrs = sorted(z_bin_ctrs)[::-1]

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    elif ax is None:
        ax = fig.subplots()
    for j, (zmax, zmin, z_ctr) in enumerate(zip(plt_z_bin_edges[:-1], plt_z_bin_edges[1:], plt_z_bin_ctrs)):
        color = None if (colors is None) else colors[j]
        loc = np.where(masses_per_z[z_ctr] >= 0.9 * min_M500)
        ax.plot(masses_per_z[z_ctr][loc], completeness_per_z[z_ctr][loc], color=color, alpha=0.9)
        # for the legend label
        ax.plot(masses_per_z[z_ctr][loc], completeness_per_z[z_ctr][loc], color=color, alpha=0.9, 
                lw=2, label=r"$%.1f < z < %.1f$" % (zmin, zmax))
    ax_ymin, _ = ax.get_ylim()
    ymin = max([round(ax_ymin,1) - 0.1, 0]) - 0.01
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([ymin, 1.01])
    ax.set_xlim([min_M500, max_M500])
    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=labelsize+1)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.legend(**legend_kwargs)
    ax.grid(alpha=0.2, which='both')
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, ax, legend_kwargs


