import numpy as np
from PME_aux_functions import expand,ready_set_go

"""Mode frequency parameterization and unpacking functions. Currently no
parameterization is used for this parameter. Mode heights are computed in
log-scale"""

def unpack_mode_heights(packed_vals, n, fit_dict, **kwargs):
    # Expanding array such that all m for given nl have same height
    # Total mode power is later modulated by the inclination and mode width
    nplus = n+fit_dict['fit']['mode_heights']['npars']
    vals = packed_vals[n:nplus]
    ht = expand(fit_dict['fit']['ells'], vals)

    # Converting to linear scale
    heights = np.exp(ht)
    return heights,nplus

def mode_height_parameterization(fit_dict = None, settings = None, plist = False):
    """Is called during setup_parameters. Selects the parameterization
    of the mode frequencies."""
    key = 'mode_heights'
    parameterization_list = ['independent']
    if plist:return parameterization_list

    if settings.md_hghts == 'independent':
        	fit_dict['fit'][key]['unpack_func'] = unpack_mode_heights
        	# Heights are converted to log scale. In the *initial_guesses.txt file the amplitudes are left as linear, so they are easier to edit.
        	fit_dict['fit'][key]['init_guess']  = np.log(fit_dict['fit'][key]['init_guess'])
        	spread, zero_check, relative  = settings.mcmc_sprd, False, False

    fit_dict['fit'][key]['npars'] = len(fit_dict['fit'][key]['init_guess'])
    pos, fit_dict['fit'][key]['llim'], fit_dict['fit'][key]['ulim'] = ready_set_go(fit_dict['fit'][key]['init_guess'],
                                                                                   [np.log(settings.heights_l_lims), np.log(settings.heights_u_lims)],
                                                                                   fit_dict['fit'][key]['npars'],
                                                                                   settings.mcmc_wlkrs,
										   spread,
                                                                                   zero_check,
                                                                                   relative)
    return pos
