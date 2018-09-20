import numpy as np
from PME_aux_functions import ready_set_go

"""Parameterization and unpacking functions for the background terms. Currently
no parameterizations are used for these parameters. Background amplitudes and
the white noise term are in log-scale."""

# #============================================================================
# Harvey exponent parameterization and unpacking functions.
# #============================================================================
def unpack_harvey_exponents(packed_vals, n, fit_dict, **kwargs):
    nplus = n+fit_dict['fit']['harvey_exponents']['npars']
    vals = packed_vals[n:nplus]
    return vals,nplus
def harvey_exponents_parameterization(fit_dict = None, settings = None, plist = False):
    key = 'harvey_exponents'
    parameterization_list = ['independent']
    if plist:return parameterization_list

    if settings.bkg_exp == 'independent':
        	fit_dict['fit'][key]['unpack_func'] = unpack_harvey_exponents
        	spread, zero_check, relative  = settings.mcmc_sprd, True, False

    fit_dict['fit'][key]['npars']       = len(fit_dict['fit'][key]['init_guess'])
    pos, fit_dict['fit'][key]['llim'], fit_dict['fit'][key]['ulim'] = ready_set_go(fit_dict['fit'][key]['init_guess'],
                                                                                   [settings.bkg_exp_l_lims, settings.bkg_exp_u_lims],
                                                                                   fit_dict['fit'][key]['npars'],
                                                                                   settings.mcmc_wlkrs,
										   spread,
                                                                                   zero_check,
                                                                                   relative)
    return pos

# #============================================================================
# Harvey frequency parameterization and unpacking functions.
# #============================================================================
def unpack_harvey_frequencies(packed_vals, n, fit_dict, **kwargs):
    nplus = n+fit_dict['fit']['harvey_frequencies']['npars']
    vals = packed_vals[n:nplus]
    return vals, nplus
def harvey_frequencies_parameterization(fit_dict = None, settings = None, plist = False):
    key = 'harvey_frequencies'
    parameterization_list = ['independent']
    if plist:return parameterization_list

    if settings.bkg_freqs == 'independent':
        fit_dict['fit'][key]['unpack_func'] = unpack_harvey_frequencies
        spread, zero_check, relative  = settings.mcmc_sprd, True, False

    fit_dict['fit'][key]['npars']       = len(fit_dict['fit'][key]['init_guess'])
    pos, fit_dict['fit'][key]['llim'], fit_dict['fit'][key]['ulim'] = ready_set_go(fit_dict['fit'][key]['init_guess'],
                                                                                   [settings.bkg_freq_l_lims, settings.bkg_freq_u_lims],
                                                                                   fit_dict['fit'][key]['npars'],
                                                                                   settings.mcmc_wlkrs,
										   spread,
                                                                                   zero_check,
                                                                                   relative)
    return pos

#==============================================================================
# Harvey power parameterization and unpacking functions.
#==============================================================================
def unpack_harvey_powers(packed_vals, n, fit_dict, **kwargs):
    nplus = n+fit_dict['fit']['harvey_powers']['npars']
    vals = packed_vals[n:nplus]
    vals = np.exp(vals)
    return vals,nplus
def harvey_powers_parameterization(fit_dict = None, settings = None, plist = False):
    key = 'harvey_powers'
    parameterization_list = ['independent']
    if plist:return parameterization_list

    if settings.bkg_power == 'independent':
        fit_dict['fit'][key]['unpack_func'] = unpack_harvey_powers
        spread, zero_check, relative  = settings.mcmc_sprd, False, False

    fit_dict['fit'][key]['npars']       = len(fit_dict['fit'][key]['init_guess'])
    pos, fit_dict['fit'][key]['llim'], fit_dict['fit'][key]['ulim'] = ready_set_go(fit_dict['fit'][key]['init_guess'],
                                                                                   [np.log(settings.bkg_pow_l_lims), np.log(settings.bkg_pow_u_lims)],
                                                                                   fit_dict['fit'][key]['npars'],
                                                                                   settings.mcmc_wlkrs,
										   spread,
                                                                                   zero_check,
                                                                                   relative)
    return pos

#==============================================================================
# White noise parameterization and unpacking functions.
#==============================================================================
def unpack_white_noise(packed_vals, n, fit_dict, **kwargs):
    nplus = n+fit_dict['fit']['harvey_powers']['npars']
    vals = packed_vals[n:nplus]
    vals = np.exp(vals)
    return vals,nplus
def white_noise_parameterization(fit_dict = None, settings = None, plist = False):
    key = 'white_noise'
    parameterization_list = ['independent']
    if plist:return parameterization_list

    if settings.bkg_wn == 'independent':
        fit_dict['fit'][key]['unpack_func']  = unpack_white_noise
        spread, zero_check, relative  = settings.mcmc_sprd, False, False

    fit_dict['fit'][key]['npars']        = len(fit_dict['fit'][key]['init_guess'])
    pos, fit_dict['fit'][key]['llim'], fit_dict['fit'][key]['ulim'] = ready_set_go(fit_dict['fit'][key]['init_guess'],
                                                                                   [np.log(settings.bkg_WN_l_lims), np.log(settings.bkg_WN_u_lims)],
                                                                                   fit_dict['fit'][key]['npars'],
                                                                                   settings.mcmc_wlkrs,
										   spread,
                                                                                   zero_check,
                                                                                   relative)
    return pos
