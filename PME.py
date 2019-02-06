import sys, os
import numpy as np
from PME_FUNCTIONS import *
import matplotlib.pyplot as plt

settings = identify_input()

if settings.autoguess:
    """Autoguess will start by computing the initial guesses for the background
    terms. A short initial fit is performed using n harvey-like background terms,
    a gaussian p-mode envelope, and a white noise term. This is done in order to
    provide a realistic initial guess for the background for the following
    automated oscillation mode detection and identification.

    You can select a number of harvey like noise terms between 1-3. Note
    that 2 terms is usually a good option for a Sun-like star. Using a single
    term is only advisable if the spectrum is truncated at a frequency so that
    it does not include the very low frequency red noise. Adding a third term
    may be necessary in some cases, and PME is currently set up such that the
    third term is located between the p-mode envelope and the granulation bump.
    Note that some stars (more evolved?) show only very weak power in this
    additional noise term, and so adding the third term is not necessary.

    The background fit is followed by the mode detection and identification.
    Modes are initially simply identified as peaks in the spectrum, by
    identifying the highest peak in an interval around nu_max. This peak is
    then divided out by fitting a lorentzian profile to the peak. This is done
    iteratively until the change in likelihood of dividing out more peaks
    reaches a specified limit (set by the -min_sign option). Note that usually
    only the central l=0, l=1 modes and a few l=2 modes are picked up by this
    detection scheme.  Adjusting the -min_sign option to a low value may help
    to find more peaks, but typically manual interaction is necessary.

    Mode identification is also attempted, and is based on finding frequency
    differences similar to the small separation. That is, if no l=0 and l=2
    pairs are found in the list of detected peaks, the mode identification will
    fail. However, so long as one pair is found, the remaining modes should be
    identified. Note, this only works for SOLAR-LIKE OSCILLATORS with no mode
    mixing, and will not work for l=3 modes.

    Autoguess can be run with the -check or -refit options. The former will
    plot initial guesses for the p-mode frequencies. The latter is used when
    additional modes are manually added in the command line or in the
    *initial_guesses.txt file. -refit will go through the mode frequencies and
    refit the lorentzian profiles to the spectrum in order to determine the
    mode heights and widths. Note that the fits to low power modes may not
    converge, typically yielding either an absurd mode height or width, this
    must be manually corrected in the *initial_guesses.txt file following -refit.
    """
    print('Starting PME in %s mode on %s' % ('autoguess', settings.star))

    if settings.check:
        check_initial_guesses(settings)
        sys.exit()

    #==============================================================================
    #  Get initial guesses for background terms
    #==============================================================================
    if not settings.refit:
        freq, power = get_powerspectrum(settings)

        # Sets up the fit dictionary, this will contain all the relevant data
        # for the fit and the star. The only other data will be in the
        # *initial_guesses.txt file.
        fit_dict = setup_fit_dict(freq, power, settings)

        # Calculates nu_max and the large separation. The background estimation
        # uses a Gaussian envelope to account for the presence of the oscillation
        # modes, and so estimate of nu_max must be computed first. The large
        # separation is used later in the mode indentification scheme.
        get_dnu_numax(fit_dict, settings)

        print("""Finding initial guesses for the background terms""")
        diag_figs = background_fit(fit_dict, settings)

        print("""Saving background results to %s""" % (os.path.join(settings.directory,settings.star+'_output.npz')))
        np.savez(os.path.join(settings.directory,settings.star+'_output.npz'), fit_dict = fit_dict)

    else:
        oput_fname = get_me_this_file(settings, name = settings.star+'_output.npz', ext = '*_output.npz')
        fit_dict = np.load(oput_fname)['fit_dict'].item()


    #==============================================================================
    # Identify peaks in p-mode envelope
    #==============================================================================
    print("""Starting mode identification... wish me luck!""")

    check_for_added_peaks(fit_dict,settings)

    write_init_file(fit_dict,settings) # Creating the initial_guesses.txt file in case mode ID fails.

    idx = define_freq_interval(fit_dict, settings)

    nlplot = add_peaks(idx, fit_dict, settings)

    diag_figs.append(nlplot)

    try:
        assign_ells(fit_dict, settings)
    except:
        enns = np.zeros_like(fit_dict['fit']['mode_freqs']['init_guess'], dtype = int)-1
        ells = np.zeros_like(fit_dict['fit']['mode_freqs']['init_guess'], dtype = int)-1
        print('Allocation of ells failed...')

    nlax = diag_figs[-1].gca()

    for i in range(len(fit_dict['fit']['ells'])):
        nlax.text(fit_dict['fit']['mode_freqs']['init_guess'][i]+0.1,nlax.get_ylim()[1]-0.2, str(fit_dict['fit']['enns'][i]))
        nlax.text(fit_dict['fit']['mode_freqs']['init_guess'][i]+0.1,nlax.get_ylim()[1]-0.4, str(fit_dict['fit']['ells'][i]))

    write_init_file(fit_dict, settings, True)

    print("""Saving diagnostics figures for the background fit.""")
    for i,fig in enumerate(diag_figs):
        fig.savefig(os.path.join(settings.directory,settings.star+'_diag_%i.png' %(i)))


    print("""Autoguess is done. Have a nice day""")


elif settings.peakbagging:
    print('Starting PME in %s mode on %s' % ('peakbagging', settings.star))

    fit_dict = get_fit_dict(settings)

    if settings.mcmc_burnt:
        settings_ = fit_dict['settings']
        settings_.mcmc_thrds = settings.mcmc_thrds
        settings_.mcmc_burnt = True
        settings_.samples = settings.samples
        settings_.mcmc_stps = settings.mcmc_stps
        settings_.mcmc_substps = settings.mcmc_substps
        settings_.mcmc_wlkrs = settings.mcmc_wlkrs
        settings_.autoguess = False
        settings_.peakbagging = True
        settings_.output = False
        settings = settings_
 
    pos, llim, ulim = setup_parameters(fit_dict,settings)
  
    if not settings.no_trim and not settings.mcmc_burnt:
    	fcut = define_freq_interval(fit_dict, settings)

    	idx_fcut = fit_dict['spectrum']['freq'] <= fcut 

    	if len(fit_dict['spectrum']['freq']) > len(fit_dict['spectrum']['freq'][idx_fcut]):

    	    print('Trimming upper end of power spectrum at %.2f' % (fcut))

    	    fit_dict['spectrum']['freq'] = fit_dict['spectrum']['freq'][idx_fcut]  
    	    fit_dict['spectrum']['power'] = fit_dict['spectrum']['power'][idx_fcut] 
    	    fit_dict['spectrum']['model']['bkg'] = fit_dict['spectrum']['model']['bkg'][idx_fcut]           
	    #fit_dict['spectrum']['model']['peakbagging'] = fit_dict['spectrum']['model']['peakbagging'][idx_fcut]           

    if settings.ask_prior:
        print("""To add a prior, add '-prior myprior.prior' to the -peakbagging call. 'myprior' must be one of the following paramter names:""")
        for key in fit_dict['parameter_keys']['mode_fit_keys']+fit_dict['parameter_keys']['bkg_keys']: 
            print(key+'.prior')
        print("""Example: python /my/directory tabbysstar -peakbagging -prior mode_splittings.prior""")
        sys.exit()

    priors = add_priors(fit_dict,settings)

    peakbagging(pos, llim, ulim, priors, fit_dict, settings)


elif settings.plot:
    print('Starting PME in %s mode on %s' % ('plotting', settings.star))

    print("""Trying to get model spectrum parameters from %s""" % (settings.star))
    fit_dict = get_fit_dict(settings)

    plot_best_fit(fit_dict, fit_dict['settings'])

    #try:
    plot_lnprobabilities(fit_dict, fit_dict['settings'])
    #except:
    #    print """WARNING: Something went wrong when trying to plot the likelihoods. This may mean that the fit is invalid."""

    plot_percentiles(fit_dict, fit_dict['settings'])

    plot_eschelle(fit_dict, fit_dict['settings'])

    plot_unpacked_all(fit_dict,fit_dict['settings'])
    plt.show()

elif settings.output:
    fit_dict = get_fit_dict(settings)
        
    settings_ = fit_dict['settings']
    settings_.mcmc_thrds = settings.mcmc_thrds
    settings_.samples = settings.samples
    settings_.mcmc_stps = settings.samples
    settings_.mcmc_substps = 0
    settings_.mcmc_burnt = True
    settings_.output = settings.output
    settings = settings_

    pos, llim, ulim = setup_parameters(fit_dict,settings)

    priors = add_priors(fit_dict,settings)

    peakbagging(pos, llim, ulim, priors, fit_dict, settings)
    
else:
    print("""No mode for PME selected. Please select either -autoguess, -peakbagging, -plot, or -output.""")
    print("""Exiting""")
    sys.exit()

