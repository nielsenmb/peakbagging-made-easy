import numpy as np
import pickle, os
from PME_aux_functions import expand, match_freqs_to_enns,ready_set_go

"""
Mode splitting parameterization and unpacking functions.

Note for the radial_step_profile:
You must provide a pre-computed structure models with. These must be arranged
in a Python dictionary placed in star directory and named starID_stellar_model_dict.
The dictionary must contain the following arrays of variables with assigned keywords:
- 'r': array of radii from 0 to R on which the model is calculated
- 'eps_r': radial displacements: xi_r(r)
- 'eps_h': horizontal displacements xi_h(r)
- 'rho': mass density rho(r)
- 'R_cz': radius R_cz of the base of the convection zone
- 'freqs': surface corrected mode frequencies (will be matched with fit frequencies)
- 'enns': list of enns for each mode frequency
- 'ells': list of ells for each mode frequency

This dictionary is added to the fit dictionary. Only information regarding the
modes included in the fit is copied.
"""



def integrate_kernels(D):
    """This function is used when the radial_step_profile option is used for
    the splittings. It computes the rotational sensitivity kernels based on
    the radial and horizontal displacements and the mass density as a function
    of radius (only!). The kernels for each mode n,l are then integrated in
    two parts, from r=0 to the base of the convection zone, and then onward to
    the surface of the star. The kernels and the two integrals for each are
    stored in the dictionary."""
    r, rho = D['r']/max(D['r']), D['rho']

    e_r, e_h = D['eps_r'], D['eps_h']

    Lsqr = D['ells']*(D['ells']+1)

    Knl = ((e_r**2.0 + (Lsqr-1)*e_h**2.0 - 2.0*e_r*e_h).T*r**2.0*rho).T / np.trapz((e_r**2.0 + Lsqr*e_h**2.0).T*r**2.0*rho, r, axis = 1)

    D['Knl'] = {'kernels'          : Knl,
                'core_integral'    : np.trapz(Knl[r <  D['Rcz']], r[r <  D['Rcz']], axis = 0),
                'envelope_integral': np.trapz(Knl[r >= D['Rcz']], r[r >= D['Rcz']], axis = 0)}


def unpack_mode_splitting_independent(packed_vals, n, fit_dict, **kwargs):
    """All enns,ells have different splittings"""
    nplus = n + fit_dict['fit']['mode_splits']['npars']

    vals = packed_vals[n:nplus]

    freqs = kwargs['freqs']

    freqs += fit_dict['fit']['x_emms']*expand(fit_dict['fit']['ells'], vals)

    return freqs, nplus


def unpack_mode_splitting_share_all(packed_vals, n, fit_dict, **kwargs):
    """All modes in the fit have the same splitting"""
    nplus = n + fit_dict['fit']['mode_splits']['npars']

    vals = packed_vals[n:nplus]

    freqs = kwargs['freqs']

    freqs += fit_dict['fit']['x_emms']*vals

    return freqs, nplus


def unpack_mode_splitting_share_n(packed_vals, n, fit_dict, **kwargs):
    """Modes of same enn have the same splitting"""
    nplus = n + fit_dict['fit']['mode_splits']['npars']

    vals = packed_vals[n:nplus]

    freqs = kwargs['freqs']

    for i, n in enumerate(np.arange(fit_dict['fit']['mode_splits']['npars'])):

        freqs[fit_dict['fit']['x_enns'] == n] += fit_dict['fit']['x_emms'][fit_dict['fit']['x_enns'] == n]*vals[i]

    return freqs, nplus


def unpack_mode_splitting_share_l(packed_vals, n, fit_dict, **kwargs):
    """Modes of same ell have the same splitting"""
    nplus = n + fit_dict['fit']['mode_splits']['npars']

    vals  = packed_vals[n:nplus]

    freqs = kwargs['freqs']

    for i, l in enumerate(np.arange(fit_dict['fit']['mode_splits']['npars'])):

        freqs[fit_dict['fit']['x_ells'] == l] += fit_dict['fit']['x_emms'][fit_dict['fit']['x_ells'] == l]*vals[i]

    return freqs, nplus


def unpack_mode_splitting_a_coeff(packed_vals, n, fit_dict, **kwargs):
    """Splittings are parameterized in terms of a-coefficents a1 and a3
    where a1 and a3 are identical for all modes"""
    nplus = n + fit_dict['fit']['mode_splits']['npars']

    a_coeffs = packed_vals[n:nplus]

    freqs = kwargs['freqs']

    for l in range(int(max(fit_dict['fit']['x_ells']))+1):
        for m in range(2*l+1):
            idx = np.invert((fit_dict['fit']['x_ells'] != l)+(fit_dict['fit']['x_emms'] != m-l))

            p = fit_dict['fit']['a_coeff_pols'][l][m]

            freqs[idx] += a_coeffs[0]*p[0]+a_coeffs[1]*p[1]

    return freqs, nplus


def unpack_mode_splitting_a_coeff_all(packed_vals, n, fit_dict, **kwargs):
    """Splittings are parameterized in terms of a-coefficents a1, a2, and
    a3 where the a-coefficients are identical for all modes"""
    nplus = n + fit_dict['fit']['mode_splits']['npars']

    a_coeffs = packed_vals[n:nplus]

    freqs = kwargs['freqs']

    for l in range(int(max(fit_dict['fit']['x_ells']))+1):
        for m in range(2*l+1):
            idx = np.invert((fit_dict['fit']['x_ells'] != l)+(fit_dict['fit']['x_emms'] != m-l))

            p = fit_dict['fit']['a_coeff_pols'][l][m]

            freqs[idx] += a_coeffs[0]*p[0]+a_coeffs[1]*p[1]+a_coeffs[2]*p[2]

    return freqs, nplus


def unpack_mode_splitting_a_coeff_all_poly(packed_vals, n, fit_dict, **kwargs):
    """Splittings are parameterized in terms of a-coefficents a1, a2, and
    a3 where the a1 and a3 are identical for all modes. a2 is a linear
    function of frequency with a constant offset"""
    nplus = n + fit_dict['fit']['mode_splits']['npars']

    a_coeffs = packed_vals[n:nplus]

    freqs = kwargs['freqs']

    for l in range(int(max(fit_dict['fit']['x_ells']))+1):
        for m in range(2*l+1):
            idx = np.invert((fit_dict['fit']['x_ells'] != l)+(fit_dict['fit']['x_emms'] != m-l))

            p = fit_dict['fit']['a_coeff_pols'][l][m]

            freqs[idx] += a_coeffs[0]*p[0]+a_coeffs[1]*p[1]+a_coeffs[2]*p[2] + a_coeffs[3]*freqs[idx]*p[1]

    return freqs, nplus


def unpack_mode_splitting_a_coeff_all_a2lin(packed_vals, n, fit_dict, **kwargs):
    """Splittings are parameterized in terms of a-coefficents a1, a2, and
    a3 where the a1 and a3 are identical for all modes. a2 is a linear
    function of frequency without a constant offset"""
    nplus = n + fit_dict['fit']['mode_splits']['npars']

    a_coeffs = packed_vals[n:nplus]

    freqs = kwargs['freqs']

    for l in range(int(max(fit_dict['fit']['x_ells']))+1):
        for m in range(2*l+1):
            idx = np.invert((fit_dict['fit']['x_ells'] != l)+(fit_dict['fit']['x_emms'] != m-l))

            p = fit_dict['fit']['a_coeff_pols'][l][m]

            freqs[idx] += a_coeffs[0]*p[0]+a_coeffs[1]*p[1]*freqs[idx] + a_coeffs[2]*p[2]

    return freqs, nplus


def unpack_mode_splitting_radial_step_profile(packed_vals, n, fit_dict, **kwargs):
    """Splittings are parameterized in terms of radial step profile. Where the
    radiative interior and convective envelope have different rotation rates."""
    nplus = n + fit_dict['fit']['mode_splits']['npars']

    O1, O2 = packed_vals[n:nplus]

    freqs = kwargs['freqs']

    s = O1*fit_dict['star']['model']['Knl']['core_integral']+O2*fit_dict['star']['model']['Knl']['envelope_integral']

    s = expand(fit_dict['fit']['ells'],s)

    freqs += fit_dict['fit']['x_emms']*s

    return freqs, nplus


def mode_splitting_parameterization(fit_dict = None, settings = None, plist = False):
    """Is called during setup_parameters. Selects the parameterization
    of the mode splittings. """

    key = 'mode_splits'
    parameterization_list = ['independent',
			     'share_all',
			     'share_n',
			     'share_l',
			     'a_coeff',
			     'a_coeff_all',
			     'a_coeff_poly',
			     'a_coeff_a2lin',
			     'radial_step_profile']
    if plist: return parameterization_list

    if settings.split == 'independent':
	# Setting initial guesses such that all multiplets nl have a different splitting
        fit_dict['fit'][key]['unpack_func'] = unpack_mode_splitting_independent

        spread, zero_check, relative  = 9999, True, False

    if settings.split == 'share_all':
	# Setting initial guess such that all modes have the same splittings
        fit_dict['fit'][key]['init_guess'] = np.array([np.mean(fit_dict['fit'][key]['init_guess'])])

        fit_dict['fit'][key]['unpack_func'] = unpack_mode_splitting_share_all

        spread, zero_check, relative  = 9999, True, False

    if settings.split == 'share_n':
	# Setting initial guesses such that modes with same enn have the same splittings
        ts = np.zeros(max(fit_dict['fit']['enns'])+1)

        for i, n in enumerate(range(max(fit_dict['fit']['enns'])+1)):
            ts[i] = fit_dict['fit'][key]['init_guess'][np.where(fit_dict['fit']['enns'] == n)[0][len(np.where(fit_dict['fit']['enns'] == n)[0])/2]]

        fit_dict['fit'][key]['init_guess'] = np.array(ts)

        fit_dict['fit'][key]['unpack_func'] = unpack_mode_splitting_share_n

 	spread, zero_check, relative  = 9999, True, False

    if settings.split == 'share_l':
	# Setting initial guesses such that modes with same ell have the same splittings
        ts = np.zeros(max(fit_dict['fit']['ells'])+1)

        for i, l in enumerate(np.arange(int(max(fit_dict['fit']['ells'])+1))):
            ts[i] = fit_dict['fit'][key]['init_guess'][np.where(fit_dict['fit']['ells'] == l)[0][len(np.where(fit_dict['fit']['ells'] == l)[0])/2]]
        fit_dict['fit'][key]['init_guess'] = np.array(ts)

	fit_dict['fit'][key]['unpack_func'] = unpack_mode_splitting_share_l

        spread, zero_check, relative  = 9999, True, False

    if settings.split == 'a_coeff':
	# Defining orthogonal polynomials for the a-coefficients
        fit_dict['fit']['a_coeff_pols'] = np.array([[                           [0, 0]                        ],
                                                    [                  [-1, 0], [0, 0], [1, 1]                ], #WRONG
                                                    [         [-2,-2], [-1, 4], [0, 0], [1,-4], [2, 2]        ],
                                                    [[-3,-3], [-2,3],  [-1, 3], [0, 0], [1,-3], [2,-3], [3, 3]]])

	# Setting initial guess for the a-coefficients
        fit_dict['fit'][key]['init_guess'] = np.array([np.mean(fit_dict['fit'][key]['init_guess']), 1e-3])

        fit_dict['fit'][key]['unpack_func'] = unpack_mode_splitting_a_coeff

	if settings.splits_u_lims == [5]: settings.splits_u_lims = [ 10.0,  10.0]
   	if settings.splits_l_lims == [0]: settings.splits_l_lims = [-10.0, -10.0]

        spread, zero_check, relative  = settings.mcmc_sprd, False, False

    if settings.split == 'a_coeff_all':
	# Defining orthogonal polynomials for the a-coefficients
        fit_dict['fit']['a_coeff_pols'] = np.array([[                                        [0,    0, 0]                                  ],
                                                    [                          [-1,   1, 0], [0,   -2, 0], [1,   1, 0]                     ],
                                                    [             [-2, 2,-2],  [-1,  -1, 4], [0,   -2, 0], [1,  -1,-4], [2, 2, 2]          ],
                                                    [[-3, 3, -3], [-2, 0, 3],  [-1,-1.8, 3], [0, -2.4, 0], [1,-1.8,-3], [2, 0,-3], [3, 3, 3]]])

	# Setting initial guess for the a-coefficients
        fit_dict['fit'][key]['init_guess'] = np.array([np.mean(fit_dict['fit'][key]['init_guess']),1e-3, 1e-3])

        fit_dict['fit'][key]['unpack_func'] = unpack_mode_splitting_a_coeff_all

   	if settings.splits_u_lims == [5]: settings.splits_u_lims = [ 10.0,  10.0, 10.0]
        if settings.splits_l_lims == [0]: settings.splits_l_lims = [-10.0, -10.0,-10.0]

        spread, zero_check, relative  = settings.mcmc_sprd, False, False

    if settings.split == 'a_coeff_poly':
	# Defining orthogonal polynomials for the a-coefficients
        fit_dict['fit']['a_coeff_pols'] = np.array([[                                        [0,    0, 0]                                  ],
                                                    [                          [-1,   1, 0], [0,   -2, 0], [1,   1, 0]                     ],
                                                    [             [-2, 2,-2],  [-1,  -1, 4], [0,   -2, 0], [1,  -1,-4], [2, 2, 2]          ],
                                                    [[-3, 3, -3], [-2, 0, 3],  [-1,-1.8, 3], [0, -2.4, 0], [1,-1.8,-3], [2, 0,-3], [3, 3, 3]]])

	# Setting initial guess for the a-coefficients
        fit_dict['fit'][key]['init_guess'] = np.array([np.mean(fit_dict['fit'][key]['init_guess']),1e-3, 1e-3,1e-5])

        fit_dict['fit'][key]['unpack_func'] = unpack_mode_splitting_a_coeff_all_poly

        if settings.splits_u_lims == [5]: settings.splits_u_lims = [10.0 ,  10.0, 10.0, 0.1]
        if settings.splits_l_lims == [0]: settings.splits_l_lims = [-10.0, -10.0,-10.0,-0.1]

        spread, zero_check, relative  = settings.mcmc_sprd, False, False

    if settings.split == 'a_coeff_a2lin':
	# Defining orthogonal polynomials for the a-coefficients
        fit_dict['fit']['a_coeff_pols'] = np.array([[                                        [0,    0, 0]                                  ],
                                                    [                          [-1,   1, 0], [0,   -2, 0], [1,   1, 0]                     ],
                                                    [             [-2, 2,-2],  [-1,  -1, 4], [0,   -2, 0], [1,  -1,-4], [2, 2, 2]          ],
                                                    [[-3, 3, -3], [-2, 0, 3],  [-1,-1.8, 3], [0, -2.4, 0], [1,-1.8,-3], [2, 0,-3], [3, 3, 3]]])

	# Setting initial guess for the a-coefficients
        fit_dict['fit'][key]['init_guess'] = np.array([np.mean(fit_dict['fit'][key]['init_guess']),-1e-6, 1e-3])

        fit_dict['fit'][key]['unpack_func'] = unpack_mode_splitting_a_coeff_all_a2lin

        if settings.splits_u_lims == [5]: settings.splits_u_lims = [ 10.0,  2.e-4, 0.1] #upper limit for a1a2a3 at 1mHz: 10,.1, and .1 .uHz
        if settings.splits_l_lims == [0]: settings.splits_l_lims = [-10.0, -2.e-4,-0.1] #lower limit at -10,-0.1, and .1 uHz

        spread, zero_check, relative = [0.2,2.0,2.0], False, False

    if settings.split == 'radial_step_profile':
	# Loading the dictionary containing the stellar structure model
	# This is used to compute the rotation kernels
        fit_dict['star']['model'] = pickle.load(open(os.path.join(settings.directory, settings.star+"_stellar_model.dict"), "rb" ))

	# Attempting to match the model frequencies to those in fit list
        match_freqs_to_enns(fit_dict)

	# Computing and integrating rotation kernels
        integrate_kernels(fit_dict['star']['model'])

    	# Setting initial guesses for the interior and envelope rotation rates
        fit_dict['fit'][key]['init_guess'] = np.array([np.mean(fit_dict['fit'][key]['init_guess']), np.mean(fit_dict['fit'][key]['init_guess'])])

        fit_dict['fit'][key]['unpack_func'] = unpack_mode_splitting_radial_step_profile

    	if settings.splits_u_lims == [5]: settings.splits_u_lims = [15 ,  15]
    	if settings.splits_l_lims == [0]: settings.splits_l_lims = [-15, -15]

        spread, zero_check, relative  = settings.mcmc_sprd, False, False

    fit_dict['fit'][key]['npars']       = len(fit_dict['fit'][key]['init_guess'])

    pos, fit_dict['fit'][key]['llim'], fit_dict['fit'][key]['ulim'] = ready_set_go(fit_dict['fit'][key]['init_guess'],
                                                                                   [settings.splits_l_lims, settings.splits_u_lims],
                                                                                   fit_dict['fit'][key]['npars'],
                                                                                   settings.mcmc_wlkrs,
										   spread,
                                                                                   zero_check,
                                                                                   relative)
    return pos



### ANCILLARY FUNCTIONS ###

def get_a_coeffs(fit_dict):
    """
        this function calculates/extract the a coefficients depending on the
        mode splitting parameterization chosen.
        
        Though it could be done via dictionaries of functions,
        the used parameterization is selected via IF statements, 
        to make more clear and simple how to implement new parameterizations

        if no parameterization is present, a WARNING message is printed and 
        a np.float(0) is returned
        
        WARNING:
        If you add a new parameterization please add the corresponding part here
        to extract the a_coefficients.
        It may also be done via a least-squares fit of the output frequencies.

    """

    unpack_func = fit_dict['fit']['mode_splits']['unpack_func']
    fd = fit_dict['fit']['mode_splits'] 
    freqs =fit_dict['fit']['mode_freqs'] 
    a_coeffs = None
    if unpack_func.func_name == 'unpack_mode_splitting_independent':
        pass
    elif unpack_func.func_name == 'unpack_mode_splitting_share_all':
        pass
    elif unpack_func.func_name == 'unpack_mode_splitting_share_n':
        pass
    elif unpack_func.func_name == 'unpack_mode_splitting_share_l':
        pass
    elif unpack_func.func_name == 'unpack_mode_splitting_a_coeff':
        pass
        
    elif unpack_func.func_name == 'unpack_mode_splitting_a_coeff_all':
        pass
    
    elif unpack_func.func_name == 'unpack_mode_splitting_a_coeff_all_poly':
        pass
    
    elif unpack_func.func_name == 'unpack_mode_splitting_a_coeff_all_a2lin':
        

        a_coeffs={}
        a_coeffs['a1']=[fd['best_fit'][0],\
                        fd['84th'][0]-fd['best_fit'][0],\
                        fd['16th'][0]-fd['best_fit'][0]]
        a_coeffs['a2']=[fd['best_fit'][1]*freqs['best_fit'],\
                        fd['84th'][1]*freqs['84th']-fd['best_fit'][1]*freqs['best_fit'],\
                        fd['16th'][1]*freqs['16th']-fd['best_fit'][1]*freqs['best_fit']]
        a_coeffs['a3']=[fd['best_fit'][2],\
                        fd['84th'][2]-fd['best_fit'][2],\
                        fd['16th'][2]-fd['best_fit'][2]]
        a_coeffs['freqs']=freqs['best_fit']

    elif unpack_func.func_name == 'unpack_mode_splitting_radial_step_profile':
        pass

    return a_coeffs


