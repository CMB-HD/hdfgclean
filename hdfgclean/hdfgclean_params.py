"""Functions used to reproduce the parameter results of arXiv:XXXX.XXXXX (!! TODO !!)"""

import os
import numpy as np
import pandas as pd
from getdist.gaussian_mixtures import GaussianND
from getdist import plots as ps
import matplotlib.pyplot as plt
from hd_mock_data import hd_data
from hdsims import utils, simutils
from hdfisher import utils as futils, fisher, config, dataconfig
from . import fgutils, hdfgclean_utils
# for displaying tables:
try:
    from IPython.display import display
except ImportError:
    display = print


# LaTeX labels for varied parameters:
param_labels = {'logA': r'\ln \left(10^{10} A_\mathrm{s}\right)',
                'ns': r'n_\mathrm{s}', 'tau': r'\tau',
                'H0': r'H_0', 'theta': r'100\theta_\mathrm{MC}',
                'ombh2': r'\Omega_\mathrm{b} h^2', 'omch2': r'\Omega_c h^2',
                'nnu': r'N_\mathrm{eff}', 'mnu': r'\sum m_\nu',
                'HMCode_logT_AGN': r'\log T_\mathrm{AGN}',
                'A_ksz': r'A_\mathrm{kSZ}', 'n_ksz': r'n_\mathrm{kSZ}'}
# number of digits used for rounding (when printing out tables) 
# for each parameter:
ndigits_round = {'logA': 5, 'ns': 5, 'tau': 5, 'H0': 3, 'theta': 7,
                 'ombh2': 7, 'omch2': 6, 'nnu': 4, 'mnu': 4,
                 'HMCode_logT_AGN': 5, 'A_ksz': 6,  'n_ksz': 6}


def fisher_root(baryonic_feedback=False, pol_only_lensing=False):
    root = 'cdm_baryons' if baryonic_feedback else 'cdm'
    if pol_only_lensing:
        root = f'{root}_kkMVpol'
    return root


def _fisher_dir():
    dir_name = 'precomputed_hd_fisher_matrices'
    fisher_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name)
    return fisher_dir


def _precomputed_fisher_matrices_dir(baryonic_feedback=False, pol_only_lensing=False):
    all_fmats_dir = _fisher_dir()
    root = fisher_root(baryonic_feedback=baryonic_feedback, pol_only_lensing=pol_only_lensing)
    return os.path.join(all_fmats_dir, root)


def fiducial_params_file(baryonic_feedback=False):
    fname_root = 'fiducial_params'
    if baryonic_feedback:
        fname_root = f'{fname_root}_baryonic_feedback'
    return os.path.join(_fisher_dir(), f'{fname_root}.yaml')


def get_fisher_dir(fisher_output_dir=None, baryonic_feedback=False, pol_only_lensing=False):
    if fisher_output_dir is not None:
        root = fisher_root(baryonic_feedback=baryonic_feedback, pol_only_lensing=pol_only_lensing)
        fisher_dir = os.path.join(fisher_output_dir, root)
    else:
        fisher_dir = _precomputed_fisher_matrices_dir(baryonic_feedback=baryonic_feedback,
                                                      pol_only_lensing=pol_only_lensing)
    return fisher_dir


def get_fisher_params(nnu=True, mnu=True, baryonic_feedback=False, ksz=False, use_H0=False):
    # list of params, in correct order for tables and triangle plots:
    if use_H0:
        params = ['ombh2', 'omch2', 'logA', 'ns', 'tau', 'H0']
        plt_params = ['ombh2', 'omch2', 'H0', 'tau', 'logA', 'ns']
    else:
        params = ['ombh2', 'omch2', 'logA', 'ns', 'tau', 'theta']
        plt_params = ['ombh2', 'omch2', 'theta', 'tau', 'logA', 'ns']
    if nnu:
        params.append('nnu')
        plt_params.append('nnu')
    if mnu:
        params.append('mnu')
        plt_params.append('mnu')
    if baryonic_feedback:
        params.append('HMCode_logT_AGN')
        plt_params.append('HMCode_logT_AGN')
    if ksz:
        params.append('A_ksz')
        plt_params.append('A_ksz')
        params.append('n_ksz')
        plt_params.append('n_ksz')
    # label for this param set:
    params_info = 'lcdm'
    if 'nnu' in params:
        params_info = f'{params_info}_nnu'
    if 'mnu' in params:
        params_info = f'{params_info}_mnu'
    if baryonic_feedback:
        params_info = f'{params_info}_TAGN'
    if ksz:
        params_info = f'{params_info}_ksz'
    if use_H0:
        params_info = f'{params_info}_useH0'

    return params_info, params, plt_params


def fiducial_params(use_H0=False):
    _, all_varied_params, _ = get_fisher_params(nnu=True, mnu=True, baryonic_feedback=True, ksz=True, use_H0=use_H0)
    fid_params = utils.load_yaml(fiducial_params_file(baryonic_feedback=True))
    fid_params['A_ksz'] = 1
    fid_params['n_ksz'] = 0
    fid_params = fgutils.dict_with_keys(fid_params, all_varied_params)
    return fid_params


def fisher_matrix_fname(fisher_output_dir=None, nnu=True, mnu=True, use_H0=False, ksz=False,
                        baryonic_feedback=False, pol_only_lensing=False,
                        cmb_type='delensed', spectra=None, bao=True, hd_data_version='v1.2'):
    # assumes `fisher_output_dir` is a main directory w/ sub-directories for each set of fisher derivs
    fmat_dir = get_fisher_dir(fisher_output_dir=fisher_output_dir, baryonic_feedback=baryonic_feedback,
                              pol_only_lensing=pol_only_lensing)
    if fisher_output_dir is not None:
        fmat_dir = os.path.join(fmat_dir, 'fisher_matrices')
    params_label, _, _ = get_fisher_params(nnu=nnu, mnu=mnu, use_H0=use_H0,
                                           baryonic_feedback=baryonic_feedback, ksz=ksz)
    fname_info = [params_label, f'{cmb_type}CMB']
    if (spectra is not None) and (len(spectra) < 5):
        spectra_info = '_'.join([s for s in ['tt', 'te', 'ee', 'bb', 'kk'] if (s in spectra)])
        fname_info.append(spectra_info)
    if bao:
        fname_info.append('BAO')
    if hd_data_version.lower() != 'v1.2':
        fname_info.append(f'{hd_data_version.lower()}HDdata')
    fname_root = '_'.join(fname_info)
    fname = os.path.join(fmat_dir, f'{fname_root}.txt')
    return fname


def save_derivs_with_respect_to_ksz_params(derivs_dir, bao=True, use_H0=False, cmb_types=None):
    hddatalib = hd_data.HDMockData(version='v1.2')
    cmb_types = cmb_types if (cmb_types is not None) else ['lensed', 'delensed', 'unlensed']
    bao_redshifts = dataconfig.Data(hd_data_version='v1.2').desi_redshifts()
    # columns for files:
    cmb_cols = ['ells', 'tt', 'te', 'ee', 'bb', 'kk']
    bao_cols = ['z', 'rs_dv']
    # fiducial amplitude and slope of kSZ power spectrum:
    fid_ksz_params = {'A_ksz': 1, 'n_ksz': 0}
    # step sizes used to calculate numberical derivatives:
    ksz_step_sizes = {'A_ksz': 0.1, 'n_ksz': 0.01}
    # load a template kSZ power spectrum:
    ksz_ells, ksz_cls = hddatalib.cl_ksz()
    # define a function to vary its amplitude or slope:
    cl_ksz = lambda A_ksz=1, n_ksz=0: simutils.get_cl_theory_from_template(ksz_ells, ksz_cls, amp=A_ksz, slope=n_ksz)
    # vary amplitude and slope of kSZ power spectrum:
    for param, fid_val in fid_ksz_params.items():
        # calculate derivative of TT power spectrum:
        step = ksz_step_sizes[param]
        params_up = {param: fid_val + step}
        params_down = {param: fid_val - step}
        cltt_deriv = (cl_ksz(**params_up) - cl_ksz(**params_down)) / (2 * step)
        # save derivs:
        header = f'{param}: fiducial = {fid_val}, step size = {step}\n'
        # derivatives are zero for polarization spectra:
        pol_derivs = {s: np.zeros(len(cltt_deriv)) for s in cmb_cols[2:]}
        cmb_ksz_derivs = {'ells': ksz_ells, 'tt': cltt_deriv, **pol_derivs}
        for cmb_type in cmb_types:
            cmb_fname = config.fisher_cmb_deriv_fname(derivs_dir, cmb_type, param, use_H0=use_H0)
            if not os.path.exists(cmb_fname):
                futils.save_to_file(cmb_fname, cmb_ksz_derivs, keys=cmb_cols, extra_header_info=header)
            bao_fname = config.fisher_bao_deriv_fname(derivs_dir, param, use_H0=use_H0)
            if bao and (not os.path.exists(bao_fname)):
                bao_ksz_derivs = {'z': bao_redshifts, 'rs_dv': np.zeros(len(bao_redshifts))}
                futils.save_to_file(bao_fname, bao_ksz_derivs, keys=bao_cols, extra_header_info=header)


def fisher_matrices_saved(fisher_output_dir=None, verbose=False):
    if fisher_output_dir is None:
        all_saved = True
    else:
        precomputed_fisher_dir = _fisher_dir()
        missing_files = []
        for name in os.listdir(precomputed_fisher_dir):
            abs_path = os.path.join(precomputed_fisher_dir, name)
            if os.path.isdir(abs_path):
                fisher_dir = os.path.join(fisher_output_dir, name)
                fmat_dir = os.path.join(fisher_dir, 'fisher_matrices')
                for fname in os.listdir(abs_path):
                    if 'v1.1HDdata' not in fname:
                        fmat_fname = os.path.join(fmat_dir, fname)
                        if not os.path.exists(fmat_fname):
                            missing_files.append(fmat_fname)
        all_saved = (len(missing_files) == 0)
        if verbose and (not all_saved):
            print('\nThe following Fisher matrix files were not found:\n')
            for fname in missing_files:
                print(f'  {fname}')
    return all_saved


def print_fisher_instructions(fisher_output_dir, hdfgclean_repo_dir=None, num_mpi_processes_fisher=12):
    all_saved = fisher_matrices_saved(fisher_output_dir=fisher_output_dir, verbose=True)
    if fisher_output_dir is None:
        print("You may proceed: the pre-computed Fisher matrices will be used.")
    elif all_saved:
        print("You may proceed: all Fisher matrices have been saved.")
    else:
        if hdfgclean_repo_dir is None:
            hdfgclean_repo_dir = hdfgclean_utils._get_default_hdfgclean_repo_dir()
        dir_name = os.path.join(hdfgclean_repo_dir, 'reproduce_10x10')
        py_file = os.path.join(dir_name, 'calculate_fisher.py')
        mpi_cmd = f'mpirun -np {int(num_mpi_processes_fisher)}' if (num_mpi_processes_fisher > 1) else ''
        cmd = f'{mpi_cmd} python {py_file} {fisher_output_dir}'
        print("\nTo calculate and save the Fisher matrices, run the following command (outside of the notebook):\n")
        print(f"  {cmd}\n")
        print("Then, re-run this notebook cell.")


def getdist_samples(fisher_matrix, fisher_params, param_labels=param_labels):
    """Returns the `getdist.mcsamples.MCSamples` instance for the
    `fisher_matrix` (two-dimensional array of `float`), with the order
    of the parameters in the rows/columns given by the list of
    parameter names, `fisher_params`.
    """
    fid_params = fiducial_params(use_H0=('H0' in fisher_params))
    fids = [fid_params[p] for p in fisher_params]
    labels = [param_labels[p] for p in fisher_params]
    samples = GaussianND(fids, fisher_matrix, is_inv_cov=True, names=fisher_params, labels=labels)
    return samples



def load_fisher(fisher_output_dir=None, nnu=True, mnu=True, ksz=False,
                 baryonic_feedback=False, pol_only_lensing=False,
                 cmb_type='delensed', spectra=None, bao=True, use_H0=False,
                 hd_data_version='v1.2'):
    fname = fisher_matrix_fname(fisher_output_dir=fisher_output_dir, nnu=nnu, mnu=mnu, ksz=ksz,
                                baryonic_feedback=baryonic_feedback, pol_only_lensing=pol_only_lensing,
                                cmb_type=cmb_type, spectra=spectra, bao=bao, use_H0=use_H0,
                                hd_data_version=hd_data_version)
    fmat, params = fisher.load_fisher_matrix(fname)
    return fmat, params


def get_fisher_errors(fisher_output_dir=None, nnu=True, mnu=True, ksz=False,
                      baryonic_feedback=False, pol_only_lensing=False,
                      cmb_type='delensed', spectra=None, bao=True, use_H0=False,
                      hd_data_version='v1.2'):
    fmat, params = load_fisher(fisher_output_dir=fisher_output_dir, nnu=nnu, mnu=mnu, ksz=ksz,
                                baryonic_feedback=baryonic_feedback, pol_only_lensing=pol_only_lensing,
                                cmb_type=cmb_type, spectra=spectra, bao=bao, use_H0=use_H0,
                                hd_data_version=hd_data_version)
    errors = fisher.get_fisher_errors(fmat, params)
    return errors


def get_fisher_samples(fisher_output_dir=None, nnu=True, mnu=True, ksz=False,
                      baryonic_feedback=False, pol_only_lensing=False,
                      cmb_type='delensed', spectra=None, bao=True, use_H0=False,
                      param_labels=param_labels, hd_data_version='v1.2'):
    fmat, params = load_fisher(fisher_output_dir=fisher_output_dir, nnu=nnu, mnu=mnu, ksz=ksz,
                                baryonic_feedback=baryonic_feedback, pol_only_lensing=pol_only_lensing,
                                cmb_type=cmb_type, spectra=spectra, bao=bao, use_H0=use_H0,
                                hd_data_version=hd_data_version)
    samples = getdist_samples(fmat, params, param_labels=param_labels)
    return samples



def param_model_label(nnu=True, mnu=True, baryonic_feedback=False, ksz=False,
                      use_latex=True, param_labels=param_labels):
    """Label for a given set of parameters to use when printing out tables."""
    lcdm_label = r'$\Lambda$CDM' if use_latex else 'LCDM'
    label_info = [lcdm_label]
    if nnu:
        nnu_label = (r'$%s$' % param_labels['nnu']) if use_latex else 'N_eff'
        label_info.append(nnu_label)
    if mnu:
        mnu_label = (r'$%s$' % param_labels['mnu']) if use_latex else 'm_nu'
        label_info.append(mnu_label)
    if baryonic_feedback:
        feedback_label = (r'$%s$' % param_labels['HMCode_logT_AGN']) if use_latex else 'log(T_AGN)'
        label_info.append(feedback_label)
    if ksz:
        for ksz_param in ['A_ksz', 'n_ksz']:
            ksz_label = (r'$%s$' % param_labels[ksz_param]) if use_latex else ksz_param
            label_info.append(ksz_label)
    label = ' + '.join(label_info)
    return label


def lensing_label(pol_only_lensing=False, use_latex=True):
    kk_label = r'$\kappa\kappa$' if use_latex else 'kk'
    lens_label = 'pol-only' if pol_only_lensing else 'MV'
    label = r"%s %s" % (lens_label, kk_label)
    return label


def round_dict(d, ndigits=ndigits_round):
    """Given a dictionary `d` with numerical values, round each value
    to the number of digits for that key, given by the `ndigits` dict.
    """
    rounded = {}
    for key, val in d.items():
        rounded[key] = round(val, ndigits[key])
    return rounded


def dict_ratio(dict1, dict2, fill_value=None, rounded=False, ndigits_round=2):
    """Returns a dictionary with the ratio of the values in `dict1` to
    those in `dict2` for each key they share in common.

    If `rounded=True`, each ratio is rounded to `ndigits_round` digits.

    If the keys don't match, you can pass a `fill_value` as a placeholder
    for any keys not in both dictionaries; otherwise leave it as `None`
    to exclude those keys from the returned dictionary.
    """
    rdict = {}
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    all_keys = list(set(keys1 + keys2))
    for key in all_keys:
        if (key in dict1) and (key in dict2):
            ratio = dict1[key] / dict2[key]
            if rounded:
                ratio = round(ratio, ndigits_round)
            rdict[key] = ratio
        elif fill_value is not None:
            rdict[key] = fill_value
    return rdict


def print_table(list_of_dicts, list_of_labels, title=None, list_of_keys=None):
    """Display a table of keys and values in each dictionary in the list.

    Parameters
    ----------
    list_of_dicts : list of dict
        A list of dictionaries.
    list_of_labels : list of str
        A list holding a label for each dictionary in `list_of_dicts`, in
        the same order that they appear in `list_of_dicts`.
    title : str or None, default=None
        A title that is printed out above the table.
    list_of_keys : list or None, default=None
        A list of keys to use in the table. The table will only contain
        values corresponding to the keys in `list_of_keys`, and in the
        same order. If `None`, all keys are used. In either case, the
        dictionaries in `list_of_dicts` do not need to contain the same
        set of keys.
    """
    table = pd.DataFrame(list_of_dicts, index=list_of_labels, columns=list_of_keys)
    table = table.T
    if title is not None:
        print('\n', title)
    display(table)


def _param_dict_for_table(param_dict, all_params=None,
                          use_latex=True, param_labels=param_labels,
                          rounded=True, ndigits_round=ndigits_round):
    if np.isscalar(ndigits_round):
        ndigits_round = {param: ndigits_round for param in param_dict}
    if all_params is None:
        all_params = list(param_dict.keys())
    fixed_params = [p for p in all_params if (p not in param_dict)]
    odict = {}
    for param in all_params:
        param_key = r'$%s$' % param_labels[param] if use_latex else param
        if param in fixed_params:
            value = '---'
        else:
            value = param_dict[param]
            if rounded:
                n = ndigits_round[param]
                value = f'{value:.{n}f}'
        odict[param_key] = value
    return odict


def print_param_table(param_errors_list, column_labels, title=None,
                      use_latex=True, param_labels=param_labels,
                      rounded=True, ndigits_round=ndigits_round):
    baryonic_feedback = any(['HMCode_logT_AGN' in errors for errors in param_errors_list])
    ksz = any(['A_ksz' in errors for errors in param_errors_list])
    _, params_order, _ = get_fisher_params(baryonic_feedback=baryonic_feedback, ksz=ksz)
    for i, param_dict in enumerate(param_errors_list):
        param_errors_list[i] = _param_dict_for_table(param_dict, all_params=params_order,
                                                     use_latex=use_latex, param_labels=param_labels,
                                                     rounded=rounded, ndigits_round=ndigits_round)
    if use_latex:
        params_order = [r'$%s$' % param_labels[p] for p in params_order]
    print_table(param_errors_list, column_labels, title=title, list_of_keys=params_order)



def print_param_ratio_table(errors1, errors2, label1, label2, ratio_label=None, title=None,
                            use_latex=True, param_labels=param_labels,
                            rounded=True, ndigits_round=ndigits_round, ndigits_round_ratio=2):
    '''
    table with columns for errors1, errors2, errors2 / errors1
    '''
    # order of rows:
    baryonic_feedback = ('HMCode_logT_AGN' in errors1) or ('HMCode_logT_AGN' in errors2)
    ksz = ('A_ksz' in errors1) or ('A_ksz' in errors2)
    _, params_order, _ = get_fisher_params(baryonic_feedback=baryonic_feedback, ksz=ksz)
    # take the ratio:
    if rounded:
        errors1 = round_dict(errors1.copy(), ndigits=ndigits_round)
        errors2 = round_dict(errors2.copy(), ndigits=ndigits_round)
    error_ratio = dict_ratio(errors2, errors1)
    if ratio_label is None:
        ratio_label = ' / '.join([label2, label1])
    # set up table columns:
    errors1 = _param_dict_for_table(errors1.copy(), use_latex=use_latex, param_labels=param_labels,
                                    rounded=rounded, ndigits_round=ndigits_round)
    errors2 = _param_dict_for_table(errors2.copy(), use_latex=use_latex, param_labels=param_labels,
                                    rounded=rounded, ndigits_round=ndigits_round)
    error_ratio = _param_dict_for_table(error_ratio, use_latex=use_latex, param_labels=param_labels,
                                        rounded=rounded, ndigits_round=ndigits_round_ratio)
    param_errors_list = [errors1, errors2, error_ratio]
    column_labels = [label1, label2, ratio_label]
    if use_latex:
        params_order = [r'$%s$' % param_labels[p] for p in params_order]
    print_table(param_errors_list, column_labels, title=title, list_of_keys=params_order)


def compare_param_errors_to_precomputed(fisher_output_dir, use_latex=True, param_labels=param_labels,
                                        rounded=False, ndigits_round=ndigits_round, ndigits_round_ratio=2):
    # put dicts of param errors and labels for each table column into lists:
    error_ratios = []
    column_labels = [] # column labels for each set of errors
    # loop through the different parameter models:
    for baryonic_feedback in [False, True]:
        vary_ksz = [False, True] if baryonic_feedback else [False]
        for ksz in vary_ksz:
            params_label = param_model_label(baryonic_feedback=baryonic_feedback, ksz=ksz, use_latex=use_latex)
            # loop through polarization-only or MV lensing:
            for pol_only_lensing in [False, True]:
                lens_label = lensing_label(pol_only_lensing=pol_only_lensing, use_latex=use_latex)
                label = r"%s, %s" % (params_label, lens_label)
                column_labels.append(label)
                # get the parameter errors from the saved fisher matrix:
                errors = get_fisher_errors(fisher_output_dir=fisher_output_dir,
                                           baryonic_feedback=baryonic_feedback, ksz=ksz,
                                           pol_only_lensing=pol_only_lensing)
                # get the parameter errors from the precomputed fisher matrix:
                precomputed_errors = get_fisher_errors(baryonic_feedback=baryonic_feedback, ksz=ksz,
                                                       pol_only_lensing=pol_only_lensing)

                # take their ratio:
                if rounded:
                    errors = round_dict(errors, ndigits=ndigits_round)
                    precomputed_errors = round_dict(precomputed_errors, ndigits=ndigits_round)
                ratio = dict_ratio(errors, precomputed_errors)

                error_ratios.append(ratio.copy())
    # print out the table:
    title = ('Sim-based CMB-HD delensed TT, TE, EE, BB + kk with DESI BAO:\n'
             ' Ratio of your 1-sigma parameter error bars to the precomputed values')
    print_param_table(error_ratios, column_labels, use_latex=use_latex,
                      rounded=rounded, ndigits_round=ndigits_round_ratio, title=title)


def plot_fig12(fisher_output_dir=None, show=True, fname=None, dpi=500):
    # load in the Fisher matrices and get the `getdist` samples for each:
    sim_based_samples = get_fisher_samples(fisher_output_dir=fisher_output_dir, ksz=True, baryonic_feedback=True)
    prev_samples = get_fisher_samples(ksz=True, baryonic_feedback=True, hd_data_version='v1.1')
    # set up for the plot:
    samples = [sim_based_samples, prev_samples]
    labels = ['Current forecast with simulation-based\ntSZ+CIB+Radio residuals',
              'Previous forecast with estimated\ntSZ+CIB+Radio residuals']
    colors = ['#56B4E9', 'tab:red']
    lines = ['-', ':']
    lws = [1.1, 1.5]
    filled = [True, False]
    alphas = [0.75, 1.0]
    title = r'CMB-HD $TT$, $TE$, $EE$, $BB$, $\kappa\kappa$ + DESI BAO '
    _, _, params = get_fisher_params(ksz=True, baryonic_feedback=True)
    # make the plot:
    plt.rcParams['legend.title_fontsize'] = 12
    g = ps.get_subplot_plotter(width_inch=7)
    g.settings.scaling_factor = 1.1
    g.settings.legend_fontsize = 14
    g.settings.axes_fontsize = 12
    g.settings.constrained_layout = True
    g.settings.figure_legend_frame = False
    g.triangle_plot(samples, legend_labels=labels,
                    legend_loc='upper right', label_order=-1,
                    params=params, param_limits={'mnu': [0, None]},
                    filled=filled,  ls=lines, lws=lws, contour_colors=colors,
                    diag1d_kwargs={'lws': lws, 'ls': lines},
                    contour_args=[{'alpha': a} for a in alphas])
    g.legend.set_title(title)
    # manually adjust some axis ticks:
    g.subplots[-1,0].set_xticks([0.02235])
    g.subplots[-1,5].set_xticks([0.965])
    g.subplots[-1,6].set_xticks([3.044])
    # turn off the grid:
    for i in range(len(params)):
        for j in range(len(params)):
            if g.subplots[i,j] is not None:
                g.subplots[i,j].grid(visible=False)
    plt.gcf().set_dpi(dpi)
    if fname is not None:
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    # reset defaults:
    plt.rcParams['legend.title_fontsize'] = None


def plot_fig13(fisher_output_dir=None, show=True, fname=None, dpi=500):
    # load in the Fisher matrices and set up to make the plot:
    kwargs = {'fisher_output_dir': fisher_output_dir, 'baryonic_feedback': True}
    samples = [get_fisher_samples(spectra=['kk'], **kwargs),
               get_fisher_samples(pol_only_lensing=True, spectra=['ee', 'bb'], bao=False, **kwargs),
               get_fisher_samples(spectra=['tt'], bao=False, **kwargs),
               get_fisher_samples(pol_only_lensing=True, spectra=['te'], bao=False, **kwargs),
               get_fisher_samples(pol_only_lensing=True, spectra=['te', 'ee', 'bb'], bao=False, **kwargs),
               get_fisher_samples(**kwargs)]
    labels = [r'Only $\kappa\kappa$ + BAO',
              r'Only $EE$ and $BB$',
              r'Only $TT$',
              r'Only $TE$',
              r'Only $TE$, $EE$ and $BB$',
              r'$TT$, $TE$, $EE$, $BB$, $\kappa\kappa$ + BAO']
    colors = ['lightgray', '#56B4E9', 'tab:olive', '#009E73', 'tab:purple', 'tab:red']
    lines = ['-']*len(samples)
    lws = [1.25]*len(samples)
    filled = [True]*len(samples)
    params = ['ombh2', 'omch2', 'theta', 'ns', 'nnu']

    # make the plot:
    plt.rcParams['legend.title_fontsize'] = 12
    g = ps.get_subplot_plotter(width_inch=5)
    g.settings.scaling_factor = 1.1
    g.settings.legend_fontsize = 14
    g.settings.axes_fontsize = 12
    g.settings.constrained_layout = True
    g.settings.figure_legend_frame = False
    g.triangle_plot(samples, legend_labels=labels, label_order=-1,
                    legend_loc='upper right', bbox_to_anchor=(1, 1.05),
                    params=params, param_limits={'mnu': [0, None]},
                    filled=filled, lws=lws, contour_colors=colors,
                    diag1d_kwargs={'lws': lws, 'ls': lines})
    # set axis limits (since kk+BAO contours are much larger than others):
    nsigma = 4 # axis limits are fiducial value +/- `nsigma` * largest error bar
    fid_params = fiducial_params()
    errors = [get_fisher_errors(pol_only_lensing=True, spectra=['ee', 'bb'], bao=False, **kwargs),
              get_fisher_errors(spectra=['tt'], bao=False, **kwargs),
              get_fisher_errors(pol_only_lensing=True, spectra=['te'], bao=False, **kwargs),
              get_fisher_errors(pol_only_lensing=True, spectra=['te', 'ee', 'bb'], bao=False, **kwargs),
              get_fisher_errors(**kwargs)]
    for i, param_row in enumerate(params):
        for j, param_col in enumerate(params):
            if g.subplots[i,j] is not None:
                g.subplots[i,j].grid(visible=False)
                max_err_for_row_param = max([err_dict[param_row] for err_dict in errors])
                max_err_for_col_param = max([err_dict[param_col] for err_dict in errors])
                xmax = fid_params[param_col] + nsigma * max_err_for_col_param
                xmin = fid_params[param_col] - nsigma * max_err_for_col_param
                ymax = fid_params[param_row] + nsigma * max_err_for_row_param
                ymin = fid_params[param_row] - nsigma * max_err_for_row_param
                g.subplots[i,j].set_xlim([xmin, xmax])
                if j != i: # don't change y-axis limits for 1d along diags
                    g.subplots[i,j].set_ylim([ymin, ymax])
    plt.gcf().set_dpi(dpi)

    if fname is not None:
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    # reset defaults:
    plt.rcParams['legend.title_fontsize'] = None


