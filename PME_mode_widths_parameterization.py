import numpy as np
from PME_aux_functions import expand, ready_set_go

"""Mode width parameterization and unpacking functions"""

def unpack_mode_width_independent(packed_vals, n, fit_dict, **kwargs):
    """Each mode has an independent width (note: this probably makes the fit
    very unstable, and the correlation with the splittings will be high)."""
    nplus = n + fit_dict['fit']['mode_widths']['npars']
    vals = packed_vals[n:nplus]
    # Expanding array such that all m for given nl have same width
    wt = expand(fit_dict['fit']['ells'], vals)

    # Converting to linear scale
    widths      = np.exp(wt)

    packed_vals = packed_vals[fit_dict['fit']['mode_widths']['npars']:]

    return widths, nplus

def unpack_mode_width_polynomial(packed_vals, n, fit_dict, **kwargs):
    """mode widths are parameterized in terms of a polynomial as a function
    of frequency, centered on nu_max. The polynomial order is set by
    settings.width_poly (5th by default)"""
    nplus = n + fit_dict['fit']['mode_widths']['npars']
    vals = packed_vals[n:nplus]

    # Evaluating the polynomial at mode frequencies nu_nlm-nu_max
    wt = np.polyval(vals, kwargs['freqs']-kwargs['nu_max'])

    # Converting to linear scale
    widths      = np.exp(wt)

    packed_vals = packed_vals[fit_dict['fit']['mode_widths']['npars']:]

    return widths, nplus


def unpack_mode_width_aporchux(packed_vals, n, fit_dict, **kwargs):
    """
    Appourchaux 2014 (corrigendum 2016) parameterization:
    from Eq. (1) of Appourchaux et al. 2016, A&A 595, C2

    p[0] -> \alpha
    p[1] -> \ln\Gamma_\alpha
    p[2] -> \ln\Delta\Gamma_{dip}
    p[3] -> \nu_{dip}
    p[4] -> \ln W_{dip}

    """
    nplus = n + fit_dict['fit']['mode_widths']['npars']

    p = packed_vals[n:nplus]

    lnx = np.log(kwargs['freqs'])

    ln_numax = np.log(kwargs['nu_max'])

    lnGamma = p[0]*(lnx - ln_numax) + p[1] - p[2] / ( 1. + 4*((lnx - np.log(p[3]))/(p[4] - ln_numax))**2)
    #lnGamma = p[0]*(lnx - ln_numax) + p[1] - p[2] / ( 1. + 4*((lnx - p[3])/(p[4] - ln_numax))**2)

    widths = np.exp(lnGamma)

    return widths, nplus

def mode_width_parameterization(fit_dict = None, settings = None, plist = False):
    """Is called during setup_parameters. Selects the parameterization
    of the mode widths. """

    key = 'mode_widths'
    parameterization_list = ['independent',
			     'polynomial',
			     'aporchucks']
    if plist: return parameterization_list

    if settings.widths == 'independent':
        fit_dict['fit'][key]['unpack_func'] = unpack_mode_width_independent

        # Adjusting default limits in case they are not manually set
        if settings.widths_u_lims == [1e5]: settings.widths_u_lims = [10]
        if settings.widths_l_lims == [1e5]: settings.widths_l_lims = [2e-19]

        spread, zero_check, relative  = settings.mcmc_sprd, False, True

    if settings.widths == 'polynomial':
        # Parameterizing the initial guesses for the mode widths. The polynomial is
        # centered on nu_max and is fit to the log(widths). Initial guess for
        # the coefficients are based on the ell = 0 modes.
        fs = fit_dict['fit']['mode_freqs']['init_guess']
        ls = fit_dict['fit']['ells']
        ws = fit_dict['fit'][key]['init_guess']
        fit_dict['fit'][key]['init_guess'] = np.polyfit(fs[ls==0] - fit_dict['star']['nu_max'], np.log(ws[ls==0]), settings.width_poly)

        fit_dict['fit'][key]['unpack_func'] = unpack_mode_width_polynomial

        # Expanding list of default limits in case they are not manually set
        if settings.widths_l_lims == [1e5]: settings.widths_l_lims = [1e5]*(settings.width_poly+1)
        if settings.widths_u_lims == [1e5]: settings.widths_u_lims = [1e5]*(settings.width_poly+1)

        spread, zero_check, relative  = settings.mcmc_sprd, False, True

    if settings.widths == 'aporchucks':
        # This follows the parameterization of the widths with frequency
        # according to Appourchaux et al. 2014. Note that some of the
        # variables have been changed to log-scale for convenience in the
        # fit.
        fit_dict['fit'][key]['init_guess'] = [4.  ,np.log(5)   ,np.log(3.)  , fit_dict['star']['nu_max'], np.log((fit_dict['star']['nu_max'] + fit_dict['fit']['mode_freqs']['init_guess'][-1])/2.)]

        fit_dict['fit'][key]['unpack_func'] = unpack_mode_width_aporchux

        nu_max = fit_dict['star']['nu_max']
        # Expanding list of default limits in case they are not manually set
        if settings.widths_l_lims == [1e5]: settings.widths_l_lims = [ 2e-19,np.log(2e-19),np.log(2e-19),fit_dict['fit']['mode_freqs']['init_guess'][0] ,np.log(nu_max+2e-19)]
        if settings.widths_u_lims == [1e5]: settings.widths_u_lims = [ 10.  ,np.log(60.)  ,np.log(60.)  ,fit_dict['fit']['mode_freqs']['init_guess'][-1],np.log(1e4)  ]


        spread, zero_check, relative  = settings.mcmc_sprd, False, False

    fit_dict['fit'][key]['npars'] = len(fit_dict['fit'][key]['init_guess'])


    pos, fit_dict['fit'][key]['llim'], fit_dict['fit'][key]['ulim'] = ready_set_go(fit_dict['fit'][key]['init_guess'],
                                                                                   [settings.widths_l_lims, settings.widths_u_lims],
                                                                                   fit_dict['fit'][key]['npars'],
                                                                                   settings.mcmc_wlkrs,
										   spread,
                                                                                   zero_check,
                                                                                   relative)




    return pos
