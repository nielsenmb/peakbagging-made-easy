import numpy as np
from PME_aux_functions import expand, ready_set_go

"""Mode frequency parameterization and unpacking functions. Currently no
parameterization is used for this parameter."""

def unpack_mode_frequencies(packed_vals, n, fit_dict, **kwargs):
    # Expanding array such that all m for given nl have same frequency
    # Is adjusted later by the splitting! Such that each emm gets a
    # different frequency nu_nlm.
    nplus = n+fit_dict['fit']['mode_freqs']['npars']
    vals = packed_vals[n:nplus]
    freqs       = expand(fit_dict['fit']['ells'], vals)
    return freqs, nplus

def mode_frequency_parameterization(fit_dict = None, settings = None, plist = False):
    """Is called during setup_parameters. Selects the parameterization
    of the mode frequencies."""
    key = 'mode_freqs'
    parameterization_list = ['independent']
    if plist:return parameterization_list

    if settings.md_freqs == 'independent':
        fit_dict['fit'][key]['unpack_func'] = unpack_mode_frequencies
        spread, zero_check, relative  = settings.mcmc_sprd, False, True

    fit_dict['fit'][key]['npars']       = len(fit_dict['fit'][key]['init_guess'])
    pos, fit_dict['fit'][key]['llim'], fit_dict['fit'][key]['ulim'] = ready_set_go(fit_dict['fit'][key]['init_guess'],
                                                                                   [settings.freqs_l_lims, settings.freqs_u_lims],
                                                                                   fit_dict['fit'][key]['npars'],
                                                                                   settings.mcmc_wlkrs,
					                                            				  spread,
                                                                                   zero_check,
                                                                                   relative)
    return pos
