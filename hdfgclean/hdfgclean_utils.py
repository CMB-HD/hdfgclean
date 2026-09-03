import os
import numpy as np
from hd_mock_data import hd_data
from hdsims import hdsims, siminfo as si, utils, simutils, hdsimsutils
from . import fgclean_info as fgi


def _get_default_hdfgclean_repo_dir():
    # assuming the user hasn't moved the example notebooks from the hdfgclean repository
    return os.path.split(os.getcwd())[0]


def _command_to_reproduce_10x10(config_file, take_power=True, match=True, plots=True, 
                                num_mpi_processes=25, hdfgclean_repo_dir=None):

    if hdfgclean_repo_dir is None:
        hdfgclean_repo_dir = _get_default_hdfgclean_repo_dir()
    py_file = os.path.join(os.path.join(hdfgclean_repo_dir, 'reproduce_10x10'), 'reproduce_10x10.py')
    if num_mpi_processes > 1:
        cmd_list = [f'mpirun -np {num_mpi_processes} python {py_file}', config_file]
    else:
        cmd_list = [f'python {py_file}', config_file]
    if match:
        cmd_list.append('--match')
    if take_power:
        cmd_list.append('--spectra')
    if plots:
        cmd_list.append('--plots')
    return ' '.join(cmd_list)


def print_instructions_to_reproduce_10x10(config_file, num_mpi_processes_fgclean=25, num_mpi_processes_spectra=2, hdfgclean_repo_dir=None):
    # check if the sims have been saved:
    hd_sims_dir = utils.load_yaml(config_file)['hd_sims_dir']
    hdsims_are_saved = hdsims_files_are_saved(hd_sims_dir)
    maps_for_noise_are_saved = noise_map_files_are_saved(hd_sims_dir)
    sim_files_saved = hdsims_are_saved and maps_for_noise_are_saved
    if not sim_files_saved:
        raise FileNotFoundError("You must save the simulation files before running FG cleaning.")
    # if they have been, print out the instructions:
    print(f"Option 1 (recommended): run each step separately")
    print(f"  First, run the FG cleaning procedure on the maps to remove CIB and radio point sources and tSZ clusters:\n")
    print(f"      {_command_to_reproduce_10x10(config_file, take_power=False, match=False, plots=False, num_mpi_processes=num_mpi_processes_fgclean, hdfgclean_repo_dir=hdfgclean_repo_dir)}\n")
    print("  After the first command has successfully completed, match the catalogs of detected sources/clusters to the true catalogs by running\n")
    print(f"      {_command_to_reproduce_10x10(config_file, take_power=False, match=True, plots=False, num_mpi_processes=num_mpi_processes_fgclean, hdfgclean_repo_dir=hdfgclean_repo_dir)}\n")
    print("  and take the power spectra of the maps after FG cleaning and save the plots by running\n")
    print(f"      {_command_to_reproduce_10x10(config_file, take_power=True, match=False, plots=False, num_mpi_processes=num_mpi_processes_spectra, hdfgclean_repo_dir=hdfgclean_repo_dir)}\n")
    print(f"\nOption 2: run everything with the following command:\n")
    print(f"      {_command_to_reproduce_10x10(config_file, take_power=True, match=True, plots=True, num_mpi_processes=num_mpi_processes_fgclean, hdfgclean_repo_dir=hdfgclean_repo_dir)}\n")
    if num_mpi_processes_fgclean > num_mpi_processes_spectra:
        print("NOTE that taking the power spectra of the full maps requires much more memory than the other steps, so we recommend at least running "
              "that step separately, either by decreasing the number of MPI processes or increasing the number of compute nodes used.") 



def _missing_spectra_files_for_plots(fgcleanlib, verbose=False):
    # kwargs for spectra of full map:
    kwargs = {'beam': True, 'bin_dl': True, 'mask': True, 'subtract_sources': True, 'subtract_clusters': True}
    # will need point source spectra on a single patch for fig 6:
    patch_num = fgcleanlib.get_patch_num(ra=fgcleanlib.ra_ctr, dec=fgcleanlib.dec_ctr)
    patch_kwargs = {'beam': True, 'noise': False, 'components': ['cib', 'radio'],
                    'bin_dl': True, 'bin_cl': False, **fgcleanlib._patch_kwargs_for_spectra(patch_num=patch_num)}
    missing_files = {}
    for freq in fgi.spectra_freqs:
        # spectra of full maps needed for figs 1 and 2
        fg_noise_spectra_fname = fgcleanlib.get_sim_power_fname(freq=freq, components=['ksz', 'tsz', 'cib', 'radio'],
                                                                noise=True, **kwargs)
        if not os.path.exists(fg_noise_spectra_fname):
            missing_files[f'{freq} GHz residual FG + noise spectrum'] = fg_noise_spectra_fname
        tsz_spectra_fname = fgcleanlib.get_sim_power_fname(freq=freq, components=['tsz'], noise=False, **kwargs)
        if not os.path.exists(tsz_spectra_fname):
            missing_files[f'{freq} GHz residual tSZ spectrium'] = tsz_spectra_fname
        cib_radio_spectra_fname = fgcleanlib.get_sim_power_fname(freq=freq, components=['cib', 'radio'],
                                                                 noise=False, **kwargs)
        if not os.path.exists(cib_radio_spectra_fname):
            missing_files[f'{freq} GHz residual CIB+Radio spectrum'] = cib_radio_spectra_fname

        # power of CIB+radio on a patch before any FG cleaning:
        patch_srcs_before_fgclean_fname = fgcleanlib.get_sim_power_fname(freq=freq, **patch_kwargs)
        if not os.path.exists(patch_srcs_before_fgclean_fname):
            min_snr = simutils.round_str(np.min(fgcleanlib.sources_snr_threshold_list))
            spectra_info = f'{freq} GHz CIB+Radio spectrum on smaller patch before any FG cleaning'
            missing_files[spectra_info] = patch_srcs_before_fgclean_fname
        # power of CIB+radio on a patch after only subtracting bright sources:
        patch_sub_bright_srcs_fname = fgcleanlib.get_sim_power_fname(freq=freq, subtract_sources=True,
                                                                     sources_sub_info='bright', **patch_kwargs)
        if not os.path.exists(patch_sub_bright_srcs_fname):
            min_snr = simutils.round_str(np.min(fgcleanlib.sources_snr_threshold_list))
            spectra_info = f'{freq} GHz CIB+Radio spectrum on smaller patch after subtracting SNR >= {min_snr}'
            missing_files[spectra_info] = patch_sub_bright_srcs_fname
        # power of CIB+radio after subtracting all sources
        patch_sub_all_srcs_fname = fgcleanlib.get_sim_power_fname(freq=freq, subtract_sources=True,
                                                                  mask=True, **patch_kwargs)
        if not os.path.exists(patch_sub_all_srcs_fname):
            min_snr = simutils.round_str(np.min(fgcleanlib.sources_snr_threshold_list))
            spectra_info = f'{freq} GHz residual CIB+Radio spectrum on smaller patch after all FG cleaning'
            missing_files[spectra_info] = patch_sub_all_srcs_fname

    # print out which files are not saved:
    if verbose and (len(missing_files) > 0):
        print('\nThe following power spectrum files were not found:')
        for file_info, fname in missing_files.items():
            print(f'  {file_info:16s}: {fname}')
    return missing_files


def spectra_files_for_plots_are_saved(fgcleanlib, verbose=True):
    missing_files = _missing_spectra_files_for_plots(fgcleanlib, verbose=verbose)
    maps_are_saved = (len(missing_files) == 0)
    return maps_are_saved


def all_fgclean_files_are_saved(fgcleanlib, config_file, num_mpi_processes_fgclean=25, num_mpi_processes_spectra=2):
    maps_are_fgcleaned = fgcleanlib.all_patches_fgcleaned()
    if not maps_are_fgcleaned:
        print("The FG-cleaned maps and catalogs of detected sources and clusters have not been saved. "
              "To run the FG cleaning, follow these instructions:\n")
        print_instructions_to_reproduce_10x10(config_file, num_mpi_processes_fgclean=num_mpi_processes_fgclean, 
                                              num_mpi_processes_spectra=num_mpi_processes_spectra)
        return maps_are_fgcleaned
    else:
        matched = fgcleanlib.all_matched_patch_catalogs_saved()
        if not matched:
            match_cmd = _command_to_reproduce_10x10(config_file, take_power=False, match=True,
                                                    plots=False, num_mpi_processes=num_mpi_processes_fgclean)
            print("The output of matching the catalogs of detected sources and clusters to the true "
                  "catalogs has not been saved. To do the matching, run the following command:\n")
            print(f"    {match_cmd}")
        spectra_saved = spectra_files_for_plots_are_saved(fgcleanlib, verbose=True)
        if not spectra_saved:
            spectra_cmd = _command_to_reproduce_10x10(config_file, take_power=True, match=False,
                                                      plots=False, num_mpi_processes=num_mpi_processes_spectra)
            print("\nTo calculate and save the power spectra, run the following command:\n")
            print(f"    {spectra_cmd}")
        if matched and spectra_saved:
            print(f"All FG cleaning output has been saved.")
        return matched and spectra_saved





def _hdsims_for_noise_maps(hd_sims_dir, make_output_dirs=False):
    return hdsims.HDSims(hd_sims_dir, make_output_dirs=make_output_dirs, **fgi.noise_map_kwargs)


def _missing_s10_files_for_noise_maps(hd_sims_dir, verbose=False, make_output_dirs=False):
    """Checks for the lower-resolution S10 CAR map and catalog files
    needed to generate the higher-resolution maps of the noise used in
    the matched filter calculations; returns a dictionary of missing file names.
    """
    simlib = _hdsims_for_noise_maps(hd_sims_dir, make_output_dirs=make_output_dirs)
    missing_files = {}
    # maps:
    lowres_map_components = ['tsz', 'ksz', 'kappa']
    for component in lowres_map_components:
        freqs = simlib.freqs if (component == 'tsz') else [None]
        for freq in freqs:
            fname = simlib.get_intermediate_sim_fname(component, freq=freq)
            if not os.path.exists(fname):
                map_info = f'{freq} GHz {component} map' if (freq is not None) else f'{component} map'
                missing_files[map_info] = fname
    # catalogs:
    catalog_components = ['sz', 'cib', 'radio']
    for component in catalog_components:
        fname = simlib.get_sim_catalog_fname(component)
        if not os.path.exists(fname):
            missing_files[f'{component} catalog'] = fname
    # print out which files are not saved:
    if verbose and (len(missing_files) > 0):
        print('\n\nThe following lower-resolution simulation files were not found:')
        for file_info, fname in missing_files.items():
            print(f'  {file_info:16s}: {fname}')
    return missing_files


def _get_commands_to_download_s10_files(hd_sims_dir, verbose=False):
    simlib = _hdsims_for_noise_maps(hd_sims_dir)
    missing_files = _missing_s10_files_for_noise_maps(hd_sims_dir, verbose=verbose)
    cmd_list = []
    for map_info, fname in missing_files.items():
        cmd_list.append(hdsimsutils._github_wget_command(fname, simlib))
    return cmd_list


def save_commands_to_download_s10_files(hd_sims_dir, output_dir=None, verbose=False):
    """Save a bash file with `wget` commands to download the
    lower-resolution S10 CAR map and catalog files needed to generate the
    higher-resolution maps of the noise used in the matched filter
    calculations.
    """
    cmd_list = _get_commands_to_download_s10_files(hd_sims_dir, verbose=verbose)
    if len(cmd_list) > 0:
        commands = '\n'.join(cmd_list)
        if output_dir is None:
            simlib = _hdsims_for_noise_maps(hd_sims_dir, make_output_dirs=True)
            output_dir = simlib.sim_dir()
        else:
            output_dir = utils.mkdir(output_dir)
        fname = os.path.join(output_dir, 'download_s10_files.sh')
        with open(fname, 'w') as f:
            f.write(commands)
        if verbose:
            print(f"\nTo download the necessary lower-resolution simulation files, run the following command:\n")
            print(f"    bash {fname}")
        return fname


def _missing_files_for_noise_maps(hd_sims_dir, verbose=False, make_output_dirs=False):
    """Checks for if the maps of the noise used in the matched filter
    calculations are saved; returns a dictionary of missing file names.
    """
    simlib = _hdsims_for_noise_maps(hd_sims_dir, make_output_dirs=make_output_dirs)
    missing_files = {}
    for component in simlib.map_components:
        freqs = simlib.freqs if simutils.has_freq_dependent_component(component) else [None]
        for freq in freqs:
            fname = simlib.get_signal_sim_fname(component, freq=freq)
            map_saved = os.path.exists(fname)
            if (component == 'cmb') and (not map_saved): # check if TQU was saved
                fname_with_pol = simlib.get_signal_sim_fname(component, freq=freq, pol=True)
                map_saved = os.path.exists(fname_with_pol)
            if not map_saved:
                map_info = f'{freq} GHz {component} map' if (freq is not None) else f'{component} map'
                missing_files[map_info] = fname
    # print out which files are not saved:
    if verbose and (len(missing_files) > 0):
        print('\n\nThe following map files needed to calculate the matched filters were not found:')
        for file_info, fname in missing_files.items():
            print(f'  {file_info:18s} : {fname}')
    return missing_files


def noise_map_files_are_saved(hd_sims_dir, verbose=True):
    missing_files = _missing_files_for_noise_maps(hd_sims_dir, verbose=verbose)
    maps_are_saved = (len(missing_files) == 0)
    return maps_are_saved




def print_instructions_to_generate_noise_sims(hd_sims_dir, hdfgclean_repo_dir=None, output_dir=None):
    maps_are_saved = noise_map_files_are_saved(hd_sims_dir, verbose=False)
    if not maps_are_saved:
        if hdfgclean_repo_dir is None: # main directory of the `hdfgclean` repository
            hdfgclean_repo_dir = _get_default_hdfgclean_repo_dir()
        python_file = os.path.join(hdfgclean_repo_dir, 'generate_noise_sims.py')
        # check if the S10 CAR maps and catalogs have been saved:
        bash_file_for_s10 = save_commands_to_download_s10_files(hd_sims_dir, output_dir=output_dir)
        if bash_file_for_s10 is not None:
            print(f"\nTo generate the maps needed to calculate the matched filters, first download the necessary lower-resolution simulation files and save them in your `hd_sims_dir` by running the following command:\n")
            print(f"    bash {bash_file_for_s10}")
            print(f"\nThen, generate the higher-resolution maps by running the following command:\n")
        else:
            print(f"\nTo generate the maps needed to calculate the matched filters, run the following command:\n")
        print(f"    python {python_file} {hd_sims_dir}")
    else:
        print(f"You may proceed: all maps needed to calculate the matched filters have been saved.")


def _missing_hdsims_files(hd_sims_dir, verbose=False, make_output_dirs=False):
    """Checks for if the HD simulation files are saved; returns a
    dictionary of missing file names.
    """
    simlib = hdsims.HDSims(hd_sims_dir, freqs=fgi.freqs, pol=False, make_output_dirs=make_output_dirs)
    missing_files = {}
    for component in simlib.map_components:
        freqs = simlib.freqs if simutils.has_freq_dependent_component(component) else [None]
        for freq in freqs:
            fname = simlib.get_signal_sim_fname(component, freq=freq)
            map_saved = os.path.exists(fname)
            if (component == 'cmb') and (not map_saved): # check if TQU was saved
                fname_with_pol = simlib.get_signal_sim_fname(component, freq=freq, pol=True)
                map_saved = os.path.exists(fname_with_pol)
            if not map_saved:
                map_info = f'{freq} GHz {component} map' if (freq is not None) else f'{component} map'
                missing_files[map_info] = fname
    # catalogs:
    catalog_components = ['tsz', 'cib', 'radio']
    for component in catalog_components:
        fname = simlib.get_catalog_fname(component)
        if not os.path.exists(fname):
            missing_files[f'{component} catalog'] = fname
    # print out which files are not saved:
    if verbose and (len(missing_files) > 0):
        print('\n\nThe following simulation files were not found:')
        for file_info, fname in missing_files.items():
            print(f'  {file_info:18s} : {fname}')
    return missing_files


def hdsims_files_are_saved(hd_sims_dir, verbose=True):
    missing_files = _missing_hdsims_files(hd_sims_dir, verbose=verbose)
    maps_are_saved = (len(missing_files) == 0)
    return maps_are_saved


def print_instructions_to_download_hdsims(hd_sims_dir, hdfgclean_repo_dir=None):
    files_are_saved = hdsims_files_are_saved(hd_sims_dir, verbose=False)
    if not files_are_saved:
        if hdfgclean_repo_dir is None:
            hdfgclean_repo_dir = _get_default_hdfgclean_repo_dir()
        bash_fname = os.path.join(os.path.join(hdfgclean_repo_dir, 'reproduce_10x10'),
                                  'download_all_HDsims_data.sh')
        print("\nTo download the HD simulation files, run the following command:\n")
        print(f"    bash {bash_fname} {hd_sims_dir}")
    else:
        print(f"You may proceed: all HD simulation files have been saved.")


def compare_10x10_spectra(fgcleanlib, fdiff_tol=0.01):
    # NOTE : only intended to use in the `reproduce10x10.ipynb` notebook

    # NOTE : will not match down to machine precision, because the binned multipoles for the sim spectra are not always integers,
    # but the spectra saved to hd mock data is only interpolated to integer multipoles
    # e.g. first bin center = 101.5 ; but in hd mock data, we only have the interpolated power at ell = 101, 102

    # compare with products provided in the `HDMockData` repository:
    datalib = hd_data.HDMockData(version='v1.2')
    # for spectra:
    kwargs = {'beam': True, 'bin_dl': True, 'mask': True, 'subtract_sources': True, 'subtract_clusters': True}
    lmax = datalib.lmax
    fg_keys = {'tsz': ['tsz'], 'cib_radio': ['cib', 'radio']}
    fg_names = {'tsz': 'tSZ', 'cib_radio': 'CIB + Radio'}
    # at each freq:
    for freq in fgi.spectra_freqs:
        # residual tSZ, CIB+radio:
        hd_fg_cls = datalib.fg_spectra(freq)
        for component, components_list in fg_keys.items():
            # precomputed:
            fg_dltt = utils.cl2dl(hd_fg_cls['ells'], hd_fg_cls[component])
            # sim spectra:
            sim_spectra = fgcleanlib.get_sim_power(freq=freq, components=components_list, noise=False, **kwargs)
            lbin = sim_spectra['ells'][sim_spectra['ells'] <= lmax]
            dlbin = sim_spectra['dltt'][:len(lbin)]
            # compare:
            fdiff = utils.get_fdiff(dlbin, (fg_dltt[lbin.astype(int)] + fg_dltt[lbin.astype(int)+1])/2)
            #fdiff = utils.get_fdiff(dlbin, fg_dltt[lbin.astype(int)])
            avg_fdiff = np.mean(fdiff)
            if abs(avg_fdiff) <= fdiff_tol:
                print(f"  Success! Your {freq} GHz residual {fg_names[component]} power spectrum matches the precomputed spectrum"
                      f" (average fractional difference is {avg_fdiff:5.2f} %)")
            else:
                min_fdiff = np.min(fdiff)
                max_fdiff = np.max(fdiff)
                print(f'  Your {freq} GHz residual {fg_names[component]} power spectrum does not match the precomputed spectrum'
                      f' (average fractional difference is {avg_fdiff:5.2f} % ; min. = {min_fdiff:5.2f} %, max. = {max_fdiff:5.2f} %)')
        # total FG + noise:
        fg_noise_cls = datalib.noise_cls(freq)
        fg_noise_dltt = utils.cl2dl(fg_noise_cls['ells'], fg_noise_cls['tt'])
        sim_spectra = fgcleanlib.get_sim_power(freq=freq, components=['ksz', 'tsz', 'cib', 'radio'], noise=True, **kwargs)
        lbin = sim_spectra['ells'][sim_spectra['ells'] <= lmax]
        dlbin = sim_spectra['dltt'][:len(lbin)]
        # compare:
        fdiff = utils.get_fdiff(dlbin, (fg_noise_dltt[lbin.astype(int)] + fg_noise_dltt[lbin.astype(int)+1])/2)
        #fdiff = utils.get_fdiff(dlbin, fg_noise_dltt[lbin.astype(int)])
        avg_fdiff = np.mean(fdiff)
        if abs(avg_fdiff) <= fdiff_tol:
            print(f"  Success! Your {freq} GHz residual FG + noise power spectrum matches the precomputed spectrum"
                  f" (average fractional difference is {avg_fdiff:5.2f} %)")
        else:
            min_fdiff = np.min(fdiff)
            max_fdiff = np.max(fdiff)
            print(f'  Your {freq} GHz residual FG + noise power spectrum does not match the precomputed spectrum'
                  f' (average fractional difference is {avg_fdiff:5.2f} % ; min. = {min_fdiff:5.2f} %, max. = {max_fdiff:5.2f} %)')

    # coadded total FG + noise:
    noise_cls = datalib.cmb_noise_spectra()
    coadd_noise_dls = utils.cl2dl(noise_cls['ells'], noise_cls['tt'])
    sim_spectra = fgcleanlib.get_coadded_fgcleaned_sim_power(cl=False, dl=True)
    lbin = sim_spectra['ells'][sim_spectra['ells'] <= lmax]
    dlbin = sim_spectra['dltt'][:len(lbin)]
    # compare:
    #fdiff = utils.get_fdiff(dlbin, coadd_noise_dls[lbin.astype(int)])
    fdiff = utils.get_fdiff(dlbin, (coadd_noise_dls[lbin.astype(int)] + coadd_noise_dls[lbin.astype(int)+1])/2)
    avg_fdiff = np.mean(fdiff)
    if abs(avg_fdiff) <= fdiff_tol:
        print(f"  Success! Your coadded residual FG + noise power spectrum matches the precomputed spectrum"
              f" (average fractional difference is {avg_fdiff:5.2f} %)")
    else:
        min_fdiff = np.min(fdiff)
        max_fdiff = np.max(fdiff)
        print(f'  Your coadded residual FG + noise power spectrum does not match the precomputed spectrum'
              f' (average fractional difference is {avg_fdiff:5.2f} % ; min. = {min_fdiff:5.2f} %, max. = {max_fdiff:5.2f} %)')
