import numpy as np

"""This is a repository of minor functions used when importing the parameterization
functions. These functions must be imported first, otherwise the parameterization
functions will complain."""

def match_freqs_to_enns(fit_dict):
    """This function is called when the radial_step_profile is used for the
    splittings. It attempts to match the initial guess frequencies from
    autoguess (or the user) with those from previously computed stellar
    models. Correct enn values are also assigned, instead of the pseudo-enn
    values used in the *initial_guess.txt file."""
    model_freqs = fit_dict['star']['model']['freqs']

    guess_freqs = fit_dict['fit']['mode_freqs']['init_guess']

    modes_in_fit = np.zeros_like(model_freqs)

    for l in range(int(max(fit_dict['fit']['ells']))+1):

        idx_l = np.where(fit_dict['fit']['ells'] == l)[0]

        mod_idx_l = fit_dict['star']['model']['ells'] == l

        for idx in idx_l:

            min_idx = np.argmin(abs(guess_freqs[idx]-model_freqs[mod_idx_l]))

            fit_dict['fit']['enns'][idx] = fit_dict['star']['model']['enns'][mod_idx_l][min_idx]

            idx_m = np.where(fit_dict['star']['model']['freqs'] == model_freqs[mod_idx_l][min_idx])

            modes_in_fit[idx_m] = 1

    modes_in_fit = modes_in_fit.astype(bool)

    sort_idx = np.argsort(fit_dict['star']['model']['freqs'][modes_in_fit])

    fit_dict['star']['model']['freqs'] = fit_dict['star']['model']['freqs'][modes_in_fit][sort_idx]
    fit_dict['star']['model']['eps_r'] = fit_dict['star']['model']['eps_r'][:, modes_in_fit][:, sort_idx]
    fit_dict['star']['model']['eps_h'] = fit_dict['star']['model']['eps_h'][:, modes_in_fit][:, sort_idx]
    fit_dict['star']['model']['enns']  = fit_dict['star']['model']['enns'][modes_in_fit][sort_idx]
    fit_dict['star']['model']['ells']  = fit_dict['star']['model']['ells'][modes_in_fit][sort_idx]


def expand(d, p):
    """Expands an array of values for each l into an array of 2*l+1 values for each l."""
    q = np.hstack(np.array([[x for x in np.ones(2*d[i]+1)*f] for i, f in enumerate(p)]))
    return q


def ready_set_go(dict_initvals, set_lims, Npars, Nwalkers, scatter, zero_check = False, relative = True):
    """This function is used to generate a spread of initial positions for the MCMC walkers. These initial
    positions are randomly drawn from a uniform distribution corresponding to +/- 100*scatter% of the input guess
    from autoguess. This spread is truncated by the upper and lower limits of the variables.

    The initial positions are returned in the form of a list of lists, one for each variable and the walker positions
    along that parameter space axis. The lower and upper limits are also return in the form of a 1D list."""

    if ((len(set_lims[0]) != Npars) and (len(set_lims[0]) != 1)) or ((len(set_lims[1]) != Npars) and (len(set_lims[1]) != 1)):
        print('Error: The number of provided limits should be either 1 or equal to the size of the parameter group in question.')
        print('Exiting...')
        sys.exit()

    if relative == True:

        u = dict_initvals + np.array(set_lims[1])

        if set_lims[0] == 2e-19:
            l = np.zeros(Npars) + np.array(set_lims[0])
        else:
            l = dict_initvals - np.array(set_lims[0])

        if zero_check == True:
            if any(l < 0): print("""With the provided limits, some of the limits were found to go below zero, this is being rectified.""")
            l[l < 0] = 2e-19
    else:
        l = np.zeros(Npars) + np.array(set_lims[0])
        u = np.zeros(Npars) + np.array(set_lims[1])

    if scatter is None:
	# If scatter is None, then upper and lower limits are used.
        pos = np.array([[np.random.uniform(l[i],u[i]) for i in range(Npars)] for i in range(Nwalkers)])

    elif isinstance(scatter, (list, tuple, np.ndarray)):
        pos = np.array([[np.random.uniform(max(l[i], dict_initvals[i]*(1.0-scatter[i])),
                                           min(u[i], dict_initvals[i]*(1.0+scatter[i]))) for i in range(Npars)] for i in range(Nwalkers)])
    elif isinstance(scatter, (float, int)):
        pos = np.array([[np.random.uniform(max(l[i], dict_initvals[i]*(1.0-scatter)),
                                           min(u[i], dict_initvals[i]*(1.0+scatter))) for i in range(Npars)] for i in range(Nwalkers)])
    else:
        print("Sure you want the scatter to be type %s? Doesn't seem to work." % (type(scatter)))
        print("If you want different scatter percentages for different parameters")

    return pos, l, u
