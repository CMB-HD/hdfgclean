import os
import numpy as np
from hd_mock_data import hd_data
from hdsims import hdsims, siminfo as si, utils, simutils, hdsimsutils, maps, fgcatalogs
from . import fgclean_info as fgi, fgutils, fgmaps


def number_of_patches(hdfgclean_config_file, verbose=True, safe_load=True):
    """NOTE: must be a config file for `HDFGClean` (not for `FGClean`)"""
    config = fgutils.load_yaml(hdfgclean_config_file, safe=safe_load)
    # info for patches:
    max_patch_size = config.get('max_patch_size', fgi.max_patch_size)
    patch_apod_width = config.get('patch_apod_width', fgi.patch_apod_width)
    # geometry of maps to be FG cleaned:
    ra_ctr = config.get('ra_ctr', si.ra_ctr)
    dec_ctr = config.get('dec_ctr', si.dec_ctr)
    width = config.get('width', si.width)
    height = config.get('height', si.height)
    res = config.get('res', si.hd_res)
    if 'map_apod_width' in config:
        map_apod_width = config['map_apod_width']
    else:
        map_apod_width = config.get('apod_width', si.apod_width)
    map_width = width + 2 * map_apod_width
    map_height = height + 2 * map_apod_width
    map_shape, map_wcs = maps.get_shape_wcs(res, ra_ctr, dec_ctr, map_width, height=map_height)
    
    # calculate number and size of patches:
    (patch_ra_ctrs, patch_dec_ctrs, 
     patch_width, patch_height) = fgmaps.divide_map_area_into_patches(map_shape, map_wcs, 
                                                                      map_apod_width=map_apod_width, 
                                                                      max_patch_size=max_patch_size, 
                                                                      patch_apod_width=patch_apod_width)
    num_rows = len(patch_dec_ctrs)
    num_cols = len(patch_ra_ctrs)
    num_patches = num_rows * num_cols
    if verbose:
        w = simutils.round_str(patch_width, n=3)
        h = simutils.round_str(patch_height, n=3)
        print(f"The map(s) will be broken up in to {num_patches} smaller "
              f"{w} degree x {h} degree patch(es) arranged in "
              f"{num_rows} row(s) and {num_cols} column(s)")

    return num_patches



def _run_hdfgclean_cmd_root(config_file, safe_load=True, 
                            masks=True, mask_freqs=fgi.spectra_freqs, 
                            num_mpi_processes=None):
    if num_mpi_processes is None:
        # one MPI process per patch:
        num_mpi_processes = number_of_patches(config_file, verbose=False, 
                                              safe_load=safe_load)
    cmd_parts = []
    if num_mpi_processes > 1:
        mpirun_cmd = f'mpirun -np {num_mpi_processes}'
        cmd_parts.append(mpirun_cmd)
    cmd_parts.append(f'python run_hdfgclean.py {os.path.abspath(config_file)}')
    if not safe_load:
        cmd_parts.append('--no_safe_load')
    if masks:
        cmd_parts.append('--masks')
        if set(mask_freqs) != set(fgi.spectra_freqs):
            freqs = ' '.join([str(freq) for freq in mask_freqs])
            cmd_parts.append(f'--mask_freqs {freqs}')
    cmd_root = ' '.join(cmd_parts)
    return cmd_root


def _run_hdfgclean_maps_cmd(config_file, save_maps=True, 
                            masks=True, mask_freqs=fgi.spectra_freqs, 
                            safe_load=True, num_mpi_processes=None):
    root = _run_hdfgclean_cmd_root(config_file, safe_load=safe_load, 
                                   masks=masks, mask_freqs=mask_freqs,
                                   num_mpi_processes=num_mpi_processes)
    cmd_parts = [root]
    if save_maps:
        cmd_parts.append('--savemaps')
    cmd = ' '.join(cmd_parts)
    return cmd


def _run_hdfgclean_match_cmd(config_file, match_freqs=None, 
                             masks=True, mask_freqs=fgi.spectra_freqs, 
                             safe_load=True, num_mpi_processes=None):
    root = _run_hdfgclean_cmd_root(config_file, safe_load=safe_load, 
                                   masks=masks, mask_freqs=mask_freqs,
                                   num_mpi_processes=num_mpi_processes)
    cmd_parts = [root, '--match']
    if match_freqs is not None:
        if set(match_freqs) != set(fgi.freqs):
            freqs = ' '.join([str(freq) for freq in match_freqs])
            cmd_parts.append(f'--match_freqs {freqs}')
    cmd = ' '.join(cmd_parts)
    return cmd


def _run_hdfgclean_spectra_cmd(config_file, spectra_freqs=None, plots=True,
                               masks=True, mask_freqs=fgi.spectra_freqs, 
                               safe_load=True, num_mpi_processes=2):
    root = _run_hdfgclean_cmd_root(config_file, safe_load=safe_load, 
                                   masks=masks, mask_freqs=mask_freqs,
                                   num_mpi_processes=num_mpi_processes)
    cmd_parts = [root, '--spectra']
    if spectra_freqs is not None:
        if set(spectra_freqs) != set(fgi.spectra_freqs):
            freqs = ' '.join([str(freq) for freq in spectra_freqs])
            cmd_parts.append(f'--spectra_freqs {freqs}')
    if plots:
        cmd_parts.append('--plots')
    cmd = ' '.join(cmd_parts)
    return cmd


def _run_hdfgclean_plots_cmd(config_file, match=True, match_freqs=None, 
                             spectra=True, spectra_freqs=None, 
                             masks=True, mask_freqs=fgi.spectra_freqs, 
                              safe_load=True, num_mpi_processes=1):
    root_cmd = _run_hdfgclean_cmd_root(config_file, safe_load=safe_load, 
                                       masks=masks, mask_freqs=mask_freqs,
                                       num_mpi_processes=num_mpi_processes)
    cmd_parts = [root_cmd, '--plots']
    if match:
        full_match_cmd = _run_hdfgclean_match_cmd(config_file, match_freqs=match_freqs, 
                                                  masks=masks, mask_freqs=mask_freqs,
                                                  safe_load=safe_load, 
                                                  num_mpi_processes=num_mpi_processes)
        match_cmd_info = full_match_cmd.strip(root_cmd)
        cmd_parts.append(match_cmd_info)
    if spectra:
        full_spectra_cmd = _run_hdfgclean_spectra_cmd(config_file, spectra_freqs=spectra_freqs, 
                                                    masks=masks, mask_freqs=mask_freqs,
                                                    safe_load=safe_load, 
                                                    num_mpi_processes=num_mpi_processes)
        spectra_cmd_info = full_spectra_cmd.strip(root_cmd)
        cmd_parts.append(spectra_cmd_info)
    cmd = ' '.join(cmd_parts)
    return cmd
        
    
def print_run_hdfgclean_commands(config_file, safe_load=True,
                                 save_maps=True, match=True, match_freqs=None, 
                                 spectra=True, spectra_freqs=None, plots=True,
                                 masks=True, mask_freqs=fgi.spectra_freqs, 
                                 num_mpi_processes=None,
                                 num_mpi_processes_spectra=2,
                                ):
    
    # print out number of patches:
    num_patches = number_of_patches(config_file, verbose=True, safe_load=safe_load)
    if num_mpi_processes is None: # use default = one MPI process per patch
        num_mpi_processes = num_patches
   
    # command for testing:
    root_cmd = _run_hdfgclean_cmd_root(config_file, safe_load=safe_load,
                                       masks=False, num_mpi_processes=num_mpi_processes)
    test_cmd = f'{root_cmd} --test'
    print("\n\nWe recommend that you test initializing `HDFGClean` first, by running "
          "the following command (with or without MPI; this test can typically be "
          "run on the login node of a cluster):\n")
    print(f"  {test_cmd}\n\n")

    # command to run fgcleaning:
    fgclean_cmd = _run_hdfgclean_maps_cmd(config_file, safe_load=safe_load, 
                                          save_maps=save_maps, masks=masks, 
                                          mask_freqs=mask_freqs, 
                                          num_mpi_processes=num_mpi_processes)
    print("\nTo FG-clean the maps, run the following command:\n")
    print(f"  {fgclean_cmd}\n")
    
    if match and spectra and plots: # run each separately
        match_cmd = _run_hdfgclean_match_cmd(config_file, match_freqs=match_freqs, 
                                             masks=masks, mask_freqs=mask_freqs,
                                             safe_load=safe_load, 
                                             num_mpi_processes=num_mpi_processes)
        spectra_cmd = _run_hdfgclean_spectra_cmd(config_file, spectra_freqs=spectra_freqs, 
                                               masks=masks, mask_freqs=mask_freqs,
                                               plots=plots, safe_load=safe_load, 
                                               num_mpi_processes=num_mpi_processes_spectra)
        plot_cmd = _run_hdfgclean_plots_cmd(config_file, match=match, match_freqs=match_freqs, 
                                            spectra=spectra, spectra_freqs=spectra_freqs, 
                                            masks=masks, mask_freqs=mask_freqs,
                                            safe_load=safe_load)
        print("After the command above has completed successfully, run the "
              "following two commands to calculate the power spectra and "
              "match  the detected and true catalogs, respectively; these "
              "may be run simultaneously:\n")
        print(f"  {spectra_cmd}\n")
        print(f"  {match_cmd}\n")
        print("After both of those have successfully completed, run the "
              "following command to make the plots:\n")
        print(f"  {plot_cmd}\n")
    
    elif plots and not (match or spectra): # only plots
        plot_cmd = _run_hdfgclean_plots_cmd(config_file, match=match, match_freqs=match_freqs, 
                                            spectra=spectra, spectra_freqs=spectra_freqs, 
                                            masks=masks, mask_freqs=mask_freqs,
                                            safe_load=safe_load)
        print("After the first command has completed successfully, "
              "make the plots by running the following command:\n")
        print(f"  {plot_cmd}\n")
        
    else:
        # doesn't matter what order we run the rest in:
        if match:
            match_cmd = _run_hdfgclean_match_cmd(config_file, match_freqs=match_freqs, 
                                                 masks=masks, mask_freqs=mask_freqs,
                                                 safe_load=safe_load, 
                                                 num_mpi_processes=num_mpi_processes)
            if plots:
                match_cmd = f'{match_cmd} --plots'
            print("After the first FG-cleaning command has completed "
                  "successfully, do the matching by running the "
                  "following command:\n")
            print(f"  {match_cmd}\n")
        if spectra:
            spectra_cmd = _run_hdfgclean_spectra_cmd(config_file, spectra_freqs=spectra_freqs, 
                                                   masks=masks, mask_freqs=mask_freqs,
                                                   plots=plots, safe_load=safe_load, 
                                                   num_mpi_processes=num_mpi_processes_spectra)
            print("After the first FG-cleaning command has completed "
                  "successfully, calculate the power spectra by running "
                  "the following command:\n")
            print(f"  {spectra_cmd}\n")


# --- for notebooks: ----


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


def print_instructions_to_reproduce_10x10(config_file,
                                          num_mpi_processes_fgclean=25,
                                          num_mpi_processes_spectra=2,
                                          hdfgclean_repo_dir=None):
    # check if the sims have been saved:
    hd_sims_dir = utils.load_yaml(config_file)['hd_sims_dir']
    hdsims_are_saved = hdsims_files_are_saved(hd_sims_dir)
    maps_for_noise_are_saved = noise_map_files_are_saved(hd_sims_dir)
    sim_files_saved = hdsims_are_saved and maps_for_noise_are_saved
    if not sim_files_saved:
        raise FileNotFoundError("You must save the simulation files before running FG cleaning.")
    # if they have been, print out the instructions:
    cmd_for_fg_cleaning = _command_to_reproduce_10x10(config_file, take_power=False,
                                                      match=False, plots=False,
                                                      num_mpi_processes=num_mpi_processes_fgclean,
                                                      hdfgclean_repo_dir=hdfgclean_repo_dir)
    cmd_for_matching = _command_to_reproduce_10x10(config_file, take_power=False,
                                                   match=True, plots=False,
                                                   num_mpi_processes=num_mpi_processes_fgclean,
                                                   hdfgclean_repo_dir=hdfgclean_repo_dir)
    cmd_for_spectra = _command_to_reproduce_10x10(config_file, take_power=True,
                                                  match=False, plots=False,
                                                  num_mpi_processes=num_mpi_processes_spectra,
                                                  hdfgclean_repo_dir=hdfgclean_repo_dir)
    cmd_for_all = _command_to_reproduce_10x10(config_file, take_power=True,
                                              match=True, plots=True,
                                              num_mpi_processes=num_mpi_processes_fgclean,
                                              hdfgclean_repo_dir=hdfgclean_repo_dir)
    print("There are two ways to run the FG cleaning.\n")
    print("Option 1 (recommended): run each step separately")
    print("  First, run the FG cleaning procedure on the maps to remove CIB"
          " and radio point sources and tSZ clusters:\n")
    print(f"      {cmd_for_fg_cleaning}\n")
    print("  After the first command has successfully completed, match the "
          "catalogs of detected sources/clusters to the true catalogs by running\n")
    print(f"      {cmd_for_matching}\n")
    print("  and take the power spectra of the maps after FG cleaning and "
          "save the plots by running\n")
    print(f"      {cmd_for_spectra}\n")
    print("\nOption 2: run everything with the following command:\n")
    print(f"      {cmd_for_all}\n")
    if num_mpi_processes_fgclean > num_mpi_processes_spectra:
        print("NOTE that taking the power spectra of the full maps requires "
              "much more memory than the other steps, so we recommend at least "
              "running that step separately, either by decreasing the number "
              "of MPI processes or increasing the number of compute nodes used.")
    print("\nAfter the command(s) above have completed successfully, "
          "you may proceed to the following notebook cells.")



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
    simlib = _hdsims_for_noise_maps(hd_sims_dir, make_output_dirs=True)
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


def _missing_hdsims_files(hd_sims_dir, verbose=False, make_output_dirs=False,
                          need_catalogs=True, **kwargs):
    """Checks for if the HD simulation files are saved; returns a
    dictionary of missing file names.

    `kwargs` are passed to `HDSims` (should not include `verbose` and `make_output_dirs`)
    """
    kwargs = {**kwargs, 'freqs': fgi.freqs, 'pol': False, 'make_output_dirs': make_output_dirs}
    simlib = hdsims.HDSims(hd_sims_dir, **kwargs)
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
    if need_catalogs:
        catalog_components = ['tsz', 'cib', 'radio']
        for component in catalog_components:
            fname = simlib.get_catalog_fname(component)
            if not os.path.exists(fname):
                missing_files[f'{component} catalog'] = fname
    # print out which files are not saved:
    if verbose and (len(missing_files) > 0):
        print('The following simulation files were not found:')
        for file_info, fname in missing_files.items():
            print(f'  {file_info:18s} : {fname}')
    return missing_files


def hdsims_files_are_saved(hd_sims_dir, verbose=True, need_catalogs=True, **kwargs):
    """`kwargs` are passed to `HDSims`"""
    kwargs = fgutils.dict_without_keys(kwargs, ['make_output_dirs'], copy=True)
    missing_files = _missing_hdsims_files(hd_sims_dir, verbose=verbose, 
                                          need_catalogs=need_catalogs, **kwargs)
    files_are_saved = (len(missing_files) == 0)
    return files_are_saved


def print_instructions_to_download_hdsims(hd_sims_dir, hdfgclean_repo_dir=None):
    files_are_saved = hdsims_files_are_saved(hd_sims_dir, verbose=False)
    if not files_are_saved:
        if hdfgclean_repo_dir is None:
            hdfgclean_repo_dir = _get_default_hdfgclean_repo_dir()
        bash_fname = os.path.join(hdfgclean_repo_dir,  'download_all_HDsims_data.sh')
        print("\nTo download the HD simulation files, run the following command:\n")
        print(f"    bash {bash_fname} {hd_sims_dir}")
    else:
        print(f"You may proceed: all HD simulation files have been saved.")


def _cmds_to_download_2x2hdsims(hd_sims_dir):
    git_repo_name = 'sim_files_for_example_notebooks'
    git_repo_url = f'https://github.com/CMB-HD/{git_repo_name}.git'
    # directories we need from the github repository:
    sims_dir_name = 'ra6dec6_2x2deg_hdsims'
    noise_dir_name = 'ra26dec6_2x2deg_hdsims'
    # absolute path:
    sims_dir = os.path.join(hd_sims_dir, sims_dir_name)
    noise_dir = os.path.join(hd_sims_dir, noise_dir_name)
    # relative path of files from github (relative to the `hd_sims_dir`):
    sims_dir_repo_path = os.path.join(git_repo_name, sims_dir_name)
    noise_dir_repo_path = os.path.join(git_repo_name, noise_dir_name)

    # make sure the `hd_sims_dir` exists:
    hd_sims_dir = utils.mkdir(hd_sims_dir)

    # don't overwrite any existing directories:
    if not os.path.exists(sims_dir):
        cmds_for_sim_files = [f'mv {sims_dir_repo_path} .']
    else: 
        files_to_move = os.path.join(sims_dir_repo_path, '*.*')
        destination = os.path.join(sims_dir_name, '')
        cmds_for_sim_files = [f'mv {files_to_move} {destination}']
        for dir_name in ['spectra', 'intermediate_maps']:
            # make sure the sub-directory also exists:
            utils.mkdir(os.path.join(sims_dir, dir_name))
            # get the path relative to the `hd_sims_dir`:
            dir_repo_path =  os.path.join(sims_dir_repo_path, dir_name)
            # command to move the github files into the directory:
            files_to_move = os.path.join(dir_repo_path, '*.*')
            destination = os.path.join(os.path.join(sims_dir_name, dir_name), '')
            cmds_for_sim_files.append(f'mv {files_to_move} {destination}')

    if not os.path.exists(noise_dir):
        cmd_for_noise_files = f'mv {noise_dir_repo_path} .'
    else:
        files_to_move = os.path.join(noise_dir_repo_path, '*.*')
        destination = os.path.join(noise_dir_name, '')
        cmd_for_noise_files = f'mv {files_to_move} {destination}'

    # list of commands:
    cmds = [f'cd {hd_sims_dir}', f'git clone {git_repo_url}',
            *cmds_for_sim_files, cmd_for_noise_files, 
            f'rm -rf {git_repo_name}']
    return cmds


def _save_commands_to_download_2x2hdsims(hd_sims_dir):
    commands = '\n'.join(_cmds_to_download_2x2hdsims(hd_sims_dir))
    fname = os.path.join(hd_sims_dir, 'download_hdfgclean_2x2example_files.sh')
    with open(fname, 'w') as f:
        f.write(commands)
    return fname


def hdsims_for_example_are_saved(hd_sims_dir, verbose=True):
    if verbose:
        print("Looking for the simulation files...")
    sims_to_fgclean_kwargs = {'width': 2, 'height': 2, 'apod_width': 0.2,
                              'pol': False, 'freqs': fgi.freqs}
    sims_to_fgclean_saved = hdsims_files_are_saved(hd_sims_dir, verbose=verbose,
                                                   **sims_to_fgclean_kwargs)
    if verbose:
        print("\nLooking for the maps needed to calculate the matched filters...")
    sims_for_filters_kwargs = {**fgi.noise_map_kwargs, **sims_to_fgclean_kwargs}
    sims_for_filters_saved = hdsims_files_are_saved(hd_sims_dir, verbose=verbose,
                                                    need_catalogs=False,
                                                    **sims_for_filters_kwargs)
    all_files_saved = sims_to_fgclean_saved and sims_for_filters_saved
    if verbose and all_files_saved:
        print("\nAll files are saved.")
    return all_files_saved


def print_instructions_to_download_2x2hdsims(hd_sims_dir):
    all_files_saved = hdsims_for_example_are_saved(hd_sims_dir, verbose=False)
    if all_files_saved:
        print("You may proceed: all necessary files have been saved.")
    else:
        bash_fname = _save_commands_to_download_2x2hdsims(hd_sims_dir)
        print("\nTo download the HD simulation files, run the following command:\n")
        print(f"    bash {bash_fname}")


def print_hdfgclean_example_instructions(config_file, hdfgclean_repo_dir=None):
    # check if the sims have been saved:
    hd_sims_dir = utils.load_yaml(config_file)['hd_sims_dir']
    sim_files_saved = hdsims_for_example_are_saved(hd_sims_dir, verbose=False)
    if not sim_files_saved:
        raise FileNotFoundError("You must save the simulation files before running FG cleaning.")
    # if they have been, print out the instructions:
    if hdfgclean_repo_dir is None:
        hdfgclean_repo_dir = _get_default_hdfgclean_repo_dir()
    py_file = os.path.join(hdfgclean_repo_dir, 'run_hdfgclean.py')
    print("First, test initializing `HDFGClean` by running "
          "the following command (outside of the notebook):\n")
    print(f"    python {py_file} {config_file} --test")
    print("\nThen, to run all FG cleaning steps, run the following command:\n")
    print(f"    python {py_file} {config_file} --savemaps --match --spectra --plots")
    print("\nAfter the command above has completed successfully,"
          " you may proceed to the following notebook cells.")


def print_fgclean_example_instructions(config_file, hdfgclean_repo_dir=None):
    if hdfgclean_repo_dir is None:
        hdfgclean_repo_dir = _get_default_hdfgclean_repo_dir()
    py_file = os.path.join(hdfgclean_repo_dir, 'run_fgclean.py')
    print("First, test initializing `FGClean` by running "
          "the following command (outside of the notebook):\n")
    print(f"    python {py_file} {config_file} --test")
    print("\nThen, to run the FG cleaning, run the following command:\n")
    print(f"    python {py_file} {config_file} --savemaps")
    print("\nAfter the command above has completed successfully,"
          " you may proceed to the following notebook cells.")


def all_2x2fgclean_files_are_saved(fgcleanlib, config_file, hdfgclean_repo_dir=None):
    maps_are_fgcleaned = fgcleanlib.all_patches_fgcleaned()
    matched = fgcleanlib.all_matched_patch_catalogs_saved()
    spectra_saved = spectra_files_for_plots_are_saved(fgcleanlib, verbose=False)
    all_files_saved = maps_are_fgcleaned and matched and spectra_saved
    if all_files_saved:
        print(f"All FG cleaning output has been saved.")
    else:
        if not maps_are_fgcleaned:
            print("The FG-cleaned maps and catalogs of detected sources and clusters have not been saved.")
        else:
            if not matched:
                print("The output of matching the catalogs of detected sources and clusters to the true catalogs has not been saved.")
            if not spectra_saved:
                print("The power spectra has not been saved.")
        print("To run the FG cleaning, follow these instructions:\n")
        print_hdfgclean_example_instructions(config_file, hdfgclean_repo_dir=hdfgclean_repo_dir)
    return all_files_saved



def detected_fg_catalogs_for_2x2filter():
    w = 2 # width of maps
    source_catalog_fnames = fgi.noise_map_source_catalog_files[w]
    source_catalogs = {}
    for freq, fname in source_catalog_fnames.items():
        source_catalogs[freq] = fgcatalogs.load_catalog(fname)
    cluster_catalog_fname = fgi.noise_map_cluster_catalog_files[w]
    cluster_catalog = fgcatalogs.load_catalog(cluster_catalog_fname)
    return source_catalogs, cluster_catalog



def compare_10x10_spectra(fgcleanlib, fdiff_tol=0.01):
    # NOTE : only intended to use in the `reproduce10x10.ipynb` notebook

    # NOTE : will not match down to machine precision, because the binned multipoles for the sim spectra are not always integers,
    # but the spectra saved to hd mock data is only interpolated to integer multipoles
    # e.g. first bin center = 101.5 ; but in hd mock data, we only have the interpolated power at ell = 101, 102

    # compare with products provided in the `HDMockData` repository:
    datalib = hd_data.HDMockData(version='v1.2')
    # for spectra:
    kwargs = {'beam': True, 'bin_dl': True, 'bin_cl': False, 'mask': True, 'subtract_sources': True, 'subtract_clusters': True}
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
                print(f"  Success! Your {freq} GHz residual {fg_names[component]} power spectrum matches "
                      f"the precomputed spectrum (average fractional difference is {avg_fdiff:5.2f} %)")
            else:
                min_fdiff = np.min(fdiff)
                max_fdiff = np.max(fdiff)
                print(f'  Your {freq} GHz residual {fg_names[component]} power spectrum does not match '
                      f'the precomputed spectrum (average fractional difference is {avg_fdiff:5.2f} % ; '
                      f'min. = {min_fdiff:5.2f} %, max. = {max_fdiff:5.2f} %)')
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


def compare_2x2_spectra(fgcleanlib, fdiff_tol=0.01):
    # NOTE : only intended to use in the `example_2x2.ipynb` notebook
    dir_name = 'precomputed_2x2_example_spectra'
    precomputed_spectra_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name)
    precomputed_spectra_fname = lambda fname: os.path.join(precomputed_spectra_dir, os.path.split(fname)[1])
    # for sim spectra:
    kwargs = {'beam': True, 'bin_dl': True, 'mask': True, 'subtract_sources': True, 'subtract_clusters': True}
    fg_keys = {'tsz': ['tsz'], 'cib_radio': ['cib', 'radio']}
    fg_names = {'tsz': 'tSZ', 'cib_radio': 'CIB + Radio'}
    # at each freq:
    for freq in fgi.spectra_freqs:
        # residual tSZ, CIB+radio:
        for component, components_list in fg_keys.items():
            # sim spectra:
            sim_spectra = fgcleanlib.get_sim_power(freq=freq, components=components_list, noise=False, bin_cl=False, **kwargs)
            # precomputed:
            sim_fname = fgcleanlib.get_sim_power_fname(freq=freq, components=components_list, noise=False, **kwargs)
            fname = precomputed_spectra_fname(sim_fname)
            _, precomputed_dlbin = np.loadtxt(fname, unpack=True)
            # compare:
            fdiff = utils.get_fdiff(sim_spectra['dltt'], precomputed_dlbin)
            avg_fdiff = np.mean(fdiff)
            if abs(avg_fdiff) <= fdiff_tol:
                print(f"  Success! Your {freq} GHz residual {fg_names[component]} power spectrum matches "
                      f"the precomputed spectrum (average fractional difference is {avg_fdiff:5.2f} %)")
            else:
                min_fdiff = np.min(fdiff)
                max_fdiff = np.max(fdiff)
                print(f'  Your {freq} GHz residual {fg_names[component]} power spectrum does not match '
                      f'the precomputed spectrum (average fractional difference is {avg_fdiff:5.2f} % ; '
                      f'min. = {min_fdiff:5.2f} %, max. = {max_fdiff:5.2f} %)')
        # total FG + noise:
        # sim spectra:
        sim_spectra = fgcleanlib.get_sim_power(freq=freq, components=['ksz', 'tsz', 'cib', 'radio'], noise=True, bin_cl=False, **kwargs)
        # precomputed:
        sim_fname = fgcleanlib.get_sim_power_fname(freq=freq, components=['ksz', 'tsz', 'cib', 'radio'], noise=True, **kwargs)
        fname = precomputed_spectra_fname(sim_fname)
        _, precomputed_dlbin = np.loadtxt(fname, unpack=True)
        # compare:
        fdiff = utils.get_fdiff(sim_spectra['dltt'], precomputed_dlbin)
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
    # sim spectra:
    sim_spectra = fgcleanlib.get_coadded_fgcleaned_sim_power(cl=False, dl=True)
    # precomputed:
    sim_fname = fgcleanlib.get_coadded_fgcleaned_sim_power_fname(dl=True)
    fname = precomputed_spectra_fname(sim_fname)
    _, precomputed_dlbin = np.loadtxt(fname, unpack=True)
    # compare:
    fdiff = utils.get_fdiff(sim_spectra['dltt'], precomputed_dlbin)
    avg_fdiff = np.mean(fdiff)
    if abs(avg_fdiff) <= fdiff_tol:
        print(f"  Success! Your coadded residual FG + noise power spectrum matches the precomputed spectrum"
              f" (average fractional difference is {avg_fdiff:5.2f} %)")
    else:
        min_fdiff = np.min(fdiff)
        max_fdiff = np.max(fdiff)
        print(f'  Your coadded residual FG + noise power spectrum does not match the precomputed spectrum'
              f' (average fractional difference is {avg_fdiff:5.2f} % ; min. = {min_fdiff:5.2f} %, max. = {max_fdiff:5.2f} %)')

