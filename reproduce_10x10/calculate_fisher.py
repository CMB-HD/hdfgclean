import os
import argparse
from hdfisher import fisher, utils, mpi
from hdfgclean import hdfgclean_params


# command-line argument for the output directory:
description = ("Calculate and save Fisher matrices with the simulation-based CMB-HD"
               " mock data (version 'v1.2' in `hdMockData`) from MacInnis et. al. (2026).")
arg_help_text = ("Path to an output directory where the Fisher matrices, and the"
                 " numerical derivatives used to calculate them, will be saved."
                " If no path is provided, the current directory will be used by default.")
formatter_class = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter_class, description=description)
parser.add_argument('fisher_output_dir', nargs='?', default=os.getcwd(), help=arg_help_text)
args = parser.parse_args()

fisher_output_dir = args.fisher_output_dir
if mpi.rank == 0:
    utils.set_dir(fisher_output_dir)
mpi.comm.barrier()
log = utils.get_logger(name='fisher', fmt="{message:s}") # use logging to print out messages

# sim-based HD mock data:
hd_data_version = 'v1.2'
cmb_type = 'delensed'
# need to calculate four sets of numerical derivatives in total for
# two different CAMB models (without and with baryonic feedback):
models_with_baryons = {'cdm': False, 'cdm_baryons': True}
# and two different lensing noise curves
# (MV from all estimators or polarization-only estimators),
# used for delensing and for the HD covariance matrices:
pol_only_kk = {'mv': False, 'pol': True}


# calculate numerical derivatives and fisher matrices:
fisherlibs = {}
for model, baryonic_feedback in models_with_baryons.items():
    fisherlibs[model] = {}
    for kk_type, pol_only_lensing in pol_only_kk.items():
        fisher_dir = hdfgclean_params.get_fisher_dir(fisher_output_dir=fisher_output_dir,
                                                     baryonic_feedback=baryonic_feedback,
                                                     pol_only_lensing=pol_only_lensing)
        fisherlib = fisher.Fisher(fisher_dir,
                                  feedback=baryonic_feedback,
                                  hd_data_version=hd_data_version,
                                  pol_only_lensing=pol_only_lensing)
        fisherlibs[model][kk_type] = fisherlib

        # calculate derivatives:
        if mpi.rank == 0:
            model_info = 'for a model with baryonic feedback ' if baryonic_feedback else ''
            lensing_info = 'using polarization-only lensing' if pol_only_lensing else ''
            log.info(f'calculating numerical derivatives {model_info}{lensing_info}')
        # calculate numerical derivates of theory with respect to CAMB cosmo. params:
        fisherlib.calculate_fisher_derivs()
        # vary amplitude and slope of kSZ power spectrum
        # (not done in `hdfisher` since these are not CAMB/CLASS params):
        if baryonic_feedback and (mpi.rank == 0):
            hdfgclean_params.save_derivs_with_respect_to_ksz_params(fisherlib.derivs_dir)
        mpi.comm.barrier()

        # calculate and save fisher matrices using all HD power spectra + DESI BAO:
        if mpi.rank == 0:
            log.info('calculating Fisher matrices')
            # for models with baryonic feedback, calculate
            # Fisher matrices with kSZ parameters fixed or free:
            vary_ksz = [True, False] if baryonic_feedback else [False]
            for ksz in vary_ksz:
                _, params, _ = hdfgclean_params.get_fisher_params(baryonic_feedback=baryonic_feedback, ksz=ksz)
                fmat_fname = hdfgclean_params.fisher_matrix_fname(fisher_output_dir=fisher_output_dir,
                                                                  baryonic_feedback=baryonic_feedback,
                                                                  pol_only_lensing=pol_only_lensing,
                                                                  ksz=ksz, cmb_type=cmb_type)
                if not os.path.exists(fmat_fname):
                    fisherlib.get_fisher(cmb_type=cmb_type, params=params, priors=True,
                                         with_desi=True, save=True, fname=fmat_fname)
                    log.info(f'saved {fmat_fname}')
        mpi.comm.barrier()


# save Fisher matrices for different combinations of CMB/lensing spectra and BAO
# for the 9-parameter LCDM + Neff + m_nu + log(T_AGN)  (baryonic feedback) model:
model = 'cdm_baryons'
baryonic_feedback = True
ksz = False
_, params, _ = hdfgclean_params.get_fisher_params(baryonic_feedback=baryonic_feedback, ksz=ksz)
priors = fisherlibs[model]['mv'].default_priors(params=params)
kk_bao_priors = {'ombh2': 0.00036, 'ns': 0.02} # for kk-only w/o other CMB
fname_kwargs = {'fisher_output_dir': fisher_output_dir, 'cmb_type': cmb_type,
                'baryonic_feedback': baryonic_feedback, 'ksz': ksz}
# list of different combinations of data for MV or pol-only lensing:
fmat_info = {'mv': [{'spectra': ['tt'], 'bao': False, 'priors': priors},
                    {'spectra': ['kk'], 'bao': True, 'priors': {**priors, **kk_bao_priors}}],
             'pol': [{'spectra': ['te', 'ee', 'bb', 'kk'], 'bao': True, 'priors': priors},
                     {'spectra': ['te', 'ee', 'bb'], 'bao': False, 'priors': priors},
                     {'spectra': ['te'], 'bao': False, 'priors': priors},
                     {'spectra': ['ee', 'bb'], 'bao': False, 'priors': priors}]}
if mpi.rank == 0:
    for kk_type, pol_only_lensing in pol_only_kk.items():
        fisherlib = fisherlibs[model][kk_type]
        for fmat_dict in fmat_info[kk_type]:
            fmat_spectra = fmat_dict['spectra']
            fmat_priors = fmat_dict['priors']
            bao = fmat_dict['bao']
            fmat_fname = hdfgclean_params.fisher_matrix_fname(pol_only_lensing=pol_only_lensing,
                                                              spectra=fmat_spectra,
                                                              bao=bao, **fname_kwargs)
            if not os.path.exists(fmat_fname):
                fisherlib.get_fisher(cmb_type=cmb_type, params=params,
                                     priors=fmat_priors, spectra=fmat_spectra,
                                     with_desi=bao, save=True, fname=fmat_fname)
                log.info(f'saved {fmat_fname}')
mpi.comm.barrier()

