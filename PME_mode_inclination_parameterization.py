import numpy as np
from PME_aux_functions import ready_set_go

"""
Mode width parameterization and unpacking functions.

To add a parameterization of this variable:
1. Add a shorthand name as one of the a choices for the -incls option in the
'identify_input' function in PME_FUNCTIONS.py.
2. Add an if statement to mode_inclination_parameterization, pointing to the
option and the shorthand name you specified in identify_input.
3. This if statement must include a parameterization of the initial guesses
and specify the unpacking function.
4. Define a function that unpacks the parameterization such that each mode
has its own value of inclination etc. (including each emms!). This function
must take packed_vals, and fit_dict as arguments. packed_vals is a list of
the fit parameters, where the relevant parameter(s) for this particular unpack
function is sliced off the start of the list. The reduced list is then returned
along with the unpacked values.

The details of the parameterization and unpacking functions for the share_all
option of the mode inclinations are provided as an example. Other
parameterizations for this and other parameters should follow a similar form.
"""


#==============================================================================
# 1. Unpacking functions for the mode inclinations.
#==============================================================================
def unpack_inclination_share_all(packed_vals, n, fit_dict, **kwargs):
    """Default setting for the inclination parameterization. This
    function takes a list of parameterized (packed) values and the
    fit dictionary. Note that some other parameters may require
    additional inputs which are given through the kwargs."""

    # Using this parameterization the packed_vals array contains a
    # single value of i, which is expanded into an array of length equal
    # to the number of modes in the fit (including emms!).
    nplus = n+fit_dict['fit']['mode_incls']['npars']
    vals = packed_vals[n:nplus]
    incls_out = np.zeros_like(fit_dict['fit']['x_enns']) + vals
    return incls_out,nplus


def unpack_inclination_share_n(packed_vals, n, fit_dict, **kwargs):
    """Alternate setting for the inclination parameterization. A
    different value of i is given to each radial order. This has not
    been tested and is mostly just for fun. Not sure its physically
    meaningful. This setting should probably be removed."""
    nplus = n+fit_dict['fit']['mode_incls']['npars']
    vals = packed_vals[n:nplus]
    incls_out = np.zeros_like(fit_dict['fit']['x_enns'])
    for i, n in enumerate(range(fit_dict['fit']['mode_incls']['npars'])):
        incls_out[fit_dict['fit']['x_enns'] == n] = vals[i]
    return incls_out,nplus



#==============================================================================
# 2. Setup for the parameterization functions for the mode inclinations.
#==============================================================================
def mode_inclination_parameterization(fit_dict = None, settings = None, plist = False):
    """Is called during setup_parameters. Selects the parameterization
    of the mode inclinations. """

    # Setting parameter key
    key = 'mode_incls'

    parameterization_list = ['share_n',
			     'share_all']
    if plist:return parameterization_list

    # Option used to assign the same inclination to all modes
    if settings.incl == 'share_all':
        # The initial guesses are recast into the parameterized form. These
        # are later used to define the inital positions of the walkers.
        fit_dict['fit'][key]['init_guess'] = np.array([np.mean(fit_dict['fit'][key]['init_guess'])])

        # Pointing the unpack keyword in fit dictionary to the correct unpacking function
        fit_dict['fit'][key]['unpack_func'] = unpack_inclination_share_all

        spread, zero_check, relative  = 1, True, False

    # Option used to assign the same inclination to modes of same radial order
    if settings.incl == 'share_n':
        	# Assigning the same inclination value to modes of same enn
        ti         = np.zeros(max(fit_dict['fit']['enns'])+1)
        for i, n in enumerate(np.arange(int(max(fit_dict['fit']['enns']))+1)):
            ti[i] =  fit_dict['fit'][key]['init_guess'][np.where(fit_dict['fit']['enns'] == n)[0][0]]
        fit_dict['fit'][key]['init_guess'] = ti

        # Pointing the unpack keyword in fit dictionary to the correct unpacking function
        fit_dict['fit'][key]['unpack_func'] = unpack_inclination_share_n

        spread, zero_check, relative  = 1, True, False

    # Given the parameterization selected above, the number of parameters are
    # determined.
    fit_dict['fit'][key]['npars'] = len(fit_dict['fit'][key]['init_guess'])

    # Based on the initial guesses the starting positions of the walkers are
    # determined by ready_set_go in PME_FUNCTIONS.py. The lower and upper limits
    # of the parameter are also specified, and the corresponding keywords in the
    # fit dictionary are updated.

    pos, fit_dict['fit'][key]['llim'], fit_dict['fit'][key]['ulim'] = ready_set_go(fit_dict['fit'][key]['init_guess'],
                                                                                   [settings.incls_l_lims, settings.incls_u_lims],
                                                                                   fit_dict['fit'][key]['npars'],
                                                                                   settings.mcmc_wlkrs,
										                                        spread,
                                                                                   zero_check,
                                                                                   relative)
    # The initial walker positions are return, to be used to make the
    # master walker position list in setup_parameters which is eventually
    # passed to EMCEE.
    return pos

