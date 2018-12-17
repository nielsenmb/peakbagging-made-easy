import matplotlib.pyplot as plt
import sys, os, emcee, argparse, flops, glob
import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter1d
from scipy.special import gamma as gamma_func
from scipy.interpolate import interp1d
from scipy import signal
import time as tm
from itertools import cycle
from PME_aux_functions import *
from PME_background_parameterization import *
from PME_mode_frequencies_parameterization import *
from PME_mode_heights_parameterization import *
from PME_mode_widths_parameterization import *
from PME_mode_splittings_parameterization import *
from PME_mode_inclination_parameterization import *
                                     

def get_me_this_file(settings, name = '', ext = ''):

   # Is name a full path name?
   if os.path.isfile(name):
       return name

   elif not os.path.isdir(settings.directory):
      print """%s does not appear to be a directory. Kind of hard to find the correct file if you don't give me the right directory.""" % (settings.directory)
      print """Exiting..."""
      sys.exit()

   elif os.path.isdir(settings.directory) and (len(name) > 0):
       # Is it a basename?
       if os.path.isfile(os.path.join(settings.directory, name)):
           return os.path.join(settings.directory, name)
       # Does it contain a wildcard?
       elif '*' in name:
	   fname =  glob.glob(os.path.join(settings.directory, name))
	   if len(fname) == 0:
               print("""Couldn't find any files with the wildcard %s in %s.""") % (name, settings.directory)
               print('Exiting...')
               sys.exit()
	   elif len(fname) > 1:
               fname =  glob.glob(os.path.join(settings.directory, settings.star+ext))
	       if len(fname) > 1: print """WARNING: multiple files with wildcard %s%s were found in %s.""" % (name,ext,settings.directory)
	       return fname[0]
	   else:
	       return fname[0]
       else:
           print """The provided input: %s, does not appear to exist in %s as either a filename or wildcard""" % (name,settings.directory)

   #Otherwise, search the directory for anything with the extension.
   elif os.path.isdir(settings.directory) and len(ext) > 0:
       if '*' not in ext: ext = '*'+ext
       fname =  glob.glob(os.path.join(settings.directory, ext))
       if len(fname) == 0:
           print("""Couldn't find any files with extension %s in %s.""") % (ext, settings.directory)
           print('Exiting...')
           sys.exit()
       elif len(fname) > 1:
           fname =  glob.glob(os.path.join(settings.directory, settings.star+ext))
	   if len(fname) > 1: print """WARNING: multiple files with extension %s were found.""" (ext)
	   return fname[0]
       else:
	   return fname[0]
   else:
       print """The provided input name: %s, does not appear to exist in %s as either a filename or wildcard""" % (name,settings.directory)
       print """Exiting..."""
       sys.exit()


def add_priors(fit_dict, settings):
    """Checks for the -prior option in the input line, and tries to find the
    *.prior corresponding to the input. Will take multiple priors. These are
    added to the dictionary entry for the corresponding variable. If an entry
    contains multiple variables, e.g., the mode frequency entry, each variable
    in the entry will have its own prior pdf. Each variable entry will receive
    an updated 'prior' entry, containing either None for no prior or an array
    with the independent variable x and the prior p(x). A combined list of the
    priors is returned for used in the lnprior() function."""

    # Resets the prior entries for each parameter entry, and adds a a list
    # containing a number 'npars' lists to each variable only containing None.
    # Later, if priors are added, they will replace the None entry. When the
    # prior lists are read, lists still containing None will be skipped.
    for key in fit_dict['parameter_keys']['mode_fit_keys']+fit_dict['parameter_keys']['bkg_keys']:
        fit_dict['fit'][key]['prior'] = []
        for k in range(fit_dict['fit'][key]['npars']):
            fit_dict['fit'][key]['prior'].append([None])

    # If prior requests were found in the input line, the request is first
    # compared to the fit dictionary to see if the variable exists.
    if settings.prior is not None:
        print """Comparing requested prior to list of parameter names in %s_output.npz""" % (settings.star)
        for line in settings.prior:
            if '.prior' in line:
                line = line[:-6]

            # The requested prior name must match the dictionary entry key.
            if line not in fit_dict['parameter_keys']['mode_fit_keys'] + fit_dict['parameter_keys']['bkg_keys']:
                print """WARNING: %s was not found in the *_output.npz dictionary, perhaps you misspelled it.""" % (line)
                print """Here are the names of the parameters for which a prior can be applied:"""
                for key in fit_dict['parameter_keys']['mode_fit_keys']+fit_dict['parameter_keys']['bkg_keys']: print key
                print """Exiting..."""
                sys.exit()

            # If the variables in the request matches a variable in the
            # dicationary the corresponding *.prior file will be searched for.
            fname = get_me_this_file(settings, name = '*%s.prior' % (line), ext = '*.prior')

            # If the prior file is found, PME will try to read it. This part
            # needs more stringent checks on the format of the prior. Correct
            # format is an independent variable x and a pdf p(x) for each
            # variable in the entry, i.e., for multi-variable entries like
            # mode_freq, there would be 1 column with the x variable and 1
            # for each mode frequency pdf. pdfs that are 0 for all values of x
            # are skipped later on. This is used when you only want to apply a
            # prior to a single variable in the entry. The requested prior name
            # must match the dictionary entry key, and needs to report any
            # errors too.
            print """Found %s, lets see if we can read it...""" % (os.path.basename(fname[0]))
            try:
                P = np.genfromtxt(fname[0])
                x, pofx = P[:, 0::2], P[:, 1::2]
            except:
                print """Hmmm...something went wrong when reading the priors
                file for %s. Make sure the file contains a column for the axis
                on which the prior is defined, and one or more columns with
                probabilities of the relevant prior.""" % (line)
                print """Exiting..."""
                sys.exit()
            print """Success!!"""

            # Check to see if the number of columns in the prior file match the
            # number 'npars' of the variables in the entry
            if np.shape(pofx)[1] != fit_dict['fit'][line]['npars']:
                print """The parameterization of %s has n=%s variables, but the
                provided prior array is not of the same size. The provided
                prior array must have n+1 columns (1st column is the axis on
                which the prior is defined). See the -help for information
                about how the prior of each parameter should be defined. For
                variables without a prior, simply replace the corresponding
                column with zeros."""  % (line, str(fit_dict['fit'][line]['npars']))
                print """Exiting..."""
                sys.exit()

            # If all checks are passed the prior is interpolated over on a
            # finer grid, the integral is normalized to unit, and the prior pdf
            # is converted to log-scale. Note the independent variable x is
            # still linear scale.
            fit_dict['fit'][line]['prior'] = []

            for k in range(fit_dict['fit'][line]['npars']):
                if all(pofx[:, k] == 0):
                    # If a column contains only zeros, the prior entry in the
                    # dictionary is set to None. I guess this could just be
                    # replaced with a continue.
                    fit_dict['fit'][line]['prior'].append([None])

                else:
                    f = interp1d(x[:,k], pofx[:,k])

                    xn = np.linspace(min(x[:,k]), max(x[:,k]), 10000)

                    prr = f(xn)

                    prr /= np.trapz(prr, xn)

                    prr = np.log(prr)

                    fit_dict['fit'][line]['prior'].append([np.vstack((xn, prr))])

        print 'Done adding priors'

    # The contents of the 'prior' key in the list of variable entries is copied
    # to a list outside the dictionary. This is what is actually passed to the
    # lnprior() function when the priors are used to computed the likelihoods.
    # There might be a better way of doing it but I couldn't think of one.
    priors = []
    for key in fit_dict['parameter_keys']['mode_fit_keys']+fit_dict['parameter_keys']['bkg_keys']:
        for k in range(fit_dict['fit'][key]['npars']):
            priors.append(fit_dict['fit'][key]['prior'][k])

    return priors


def savefit(sampler, substeps, fit_dict, settings):
    """This function is called to save the progress of the fit. Note that only
    the last 500 steps in the MCMC are stored at any one time (still need to
    implement this as a tunable paramter). The lnprobabilities and the
    percentiles values of the chain ARE however stored for all steps. This is
    implemented to save storage space, otherwise the MCMC chain will require
    several gigs of storage space per star."""

    percentages = [50-95/2, 50 - 68.27/2, 50., 50 + 68.27/2, 50+95/2]

    if settings.peakbagging:

        chunk = sampler.chain.copy()
        
        lnprobs = sampler.lnprobability.copy()

        # At each save point, the last positions of the walkers are stored. When
        # the fit is restarted this will be the starting point for the walkers.
        fit_dict['fit']['last_chain_state'] = sampler.chain[:, -1, :]

        # Step number is recorded
        if np.size(fit_dict['fit']['steps']) == 0:
            fit_dict['fit']['steps'] = np.array([substeps])
        else:
            fit_dict['fit']['steps'] = np.append(fit_dict['fit']['steps'], fit_dict['fit']['steps'][-1]+substeps)

        # The lnprobabilities for each walkers are stored as percentiles for each substep.
        try:
            L = np.percentile(lnprobs,percentages)
        except:
            print "Inf/nan values in lnprob"
            L = np.ones(5)*1e-20

        if np.size(fit_dict['fit']['lnprobabilities']) == 0:
            fit_dict['fit']['lnprobabilities'] = L
        else:
            fit_dict['fit']['lnprobabilities'] = np.vstack((fit_dict['fit']['lnprobabilities'], L))

        # Here the percentile values of the walker distribution is stored for each
        # step in the chain.
        for key in fit_dict['parameter_keys']['mode_fit_keys']+fit_dict['parameter_keys']['bkg_keys']:
            C = chunk[:, -substeps:, :fit_dict['fit'][key]['npars']] # (nwalkers, substeps, npars)

            Cflat = C.reshape(-1,fit_dict['fit'][key]['npars']) # (nwalkers*substeps, npars)

            Cflat_percs =  np.percentile(Cflat, percentages, axis = 0)[:,np.newaxis,:] # (nwalkers*substeps, 0, npars)

            if np.size(fit_dict['fit'][key]['chain']) == 0:
                fit_dict['fit'][key]['chain'] = C[:, -settings.samples:, :]

                fit_dict['fit'][key]['chain_percentiles'] = Cflat_percs
                
            else:
                fit_dict['fit'][key]['chain']             = np.hstack((fit_dict['fit'][key]['chain'], C))[:, -settings.samples:, :]

                fit_dict['fit'][key]['chain_percentiles'] = np.concatenate((fit_dict['fit'][key]['chain_percentiles'], Cflat_percs), axis = 1)

            fit_dict['fit'][key]['16th'], fit_dict['fit'][key]['best_fit'], fit_dict['fit'][key]['84th'] = np.percentile(Cflat, percentages[1:4], axis = 0) #fit_dict['fit'][key]['chain_percentiles'][1:4,-1,:]

            chunk = chunk[:, :, fit_dict['fit'][key]['npars']:]

        # Here a model is generated using the median value of each chain. This can
        # be used for diagnosing the fit.
        all_chains_flat = sampler.chain.reshape((-1, fit_dict['fit']['ndim']))
        
        pf = np.percentile(all_chains_flat, 50., axis = 0)
        
        enns, ells, emms, freqs, heights, widths, incls, h_exponent, h_frequency, h_power, w_noise = unpack_parameters(pf, fit_dict, settings)

        fit_dict['spectrum']['model']['peakbagging'] = flops.spectral_model(fit_dict['spectrum']['freq'],
                                                                            ells,
                                                                            abs(emms),
                                                                            freqs,
                                                                            heights,
                                                                            widths,
                                                                            incls*np.pi/180.0,
                                                                            h_exponent,
                                                                            h_frequency,
                                                                            h_power,
                                                                            w_noise)
        # The dictionary is saved
        np.savez(os.path.join(settings.directory, settings.star+'_output.npz'), fit_dict = fit_dict)

    # Output ascii file with mode parameters

    if settings.output:
        
        n_samps = len(all_chains_flat[:,0])

        n_modes = fit_dict['fit']['mode_freqs']['npars']

        mode_pars = np.zeros((3,n_modes,n_samps))
            
        for i in range(n_samps):

            pars = all_chains_flat[i,:]
                
            enns, ells, emms, freqs, heights, widths, incls, h_exponent, h_frequency, h_power, w_noise = unpack_parameters(pars, fit_dict, settings)

            mode_pars[:,:,i] = freqs[emms == 0], heights[emms==0], widths[emms == 0]

        mode_pars_percs = np.percentile(mode_pars,percentages[1:4], axis = 2) # perc, par, mode

        out_arr = np.array([enns[emms == 0],
			    ells[emms == 0],
			    mode_pars_percs[1,0,:],
                            mode_pars_percs[2,0,:]-mode_pars_percs[1,0,:],
                            mode_pars_percs[1,0,:]-mode_pars_percs[0,0,:],
                            mode_pars_percs[1,1,:],
                            mode_pars_percs[2,1,:]-mode_pars_percs[1,1,:],
                            mode_pars_percs[1,1,:]-mode_pars_percs[0,1,:],
                            mode_pars_percs[1,2,:],
                            mode_pars_percs[2,2,:]-mode_pars_percs[1,2,:],
                            mode_pars_percs[1,2,:]-mode_pars_percs[0,2,:]]).T

        fname = os.path.join(settings.directory, settings.star+'_bestfit_modes_pars.csv')


        hdr = 'enn, ell, Mode frequency [muHz], Mode frequency +error [muHz], Mode frequency -error [muHz], Mode height [ppm^2/muHz], Mode height +error [ppm^2/muHz], Mode height -error [ppm^2/muHz], Mode width [muHz], Mode width +error [muHz], Mode width -error [muHz]'
        
        np.savetxt(fname,out_arr, header = hdr, delimiter=',', fmt = '%i,%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f')
            

    return fit_dict['fit']['last_chain_state']


def unpack_parameters(packed_vals, fit_dict, settings):
    """This unpacks the parameterized fit values into a simple form, i.e., a
    single frequency, width, height, splitting, inclination for each mode.
    Example: if the share_all flag is set for the inclination, the MCMC sampler
    will only iterate 1 parameter, and so this 1 value is assigned to all the N
    modes individually.

    This uses the 'unpack_func' stored in the dictionary for each parameter
    entry during the fit setup.

    Note that the packed values are in a single ordered array, and must be
    unpacked in the correct order. So don't go changing the order of the
    variables in the return statement below

    EMCEE cannot iterate through variables in dictionary entries, and requires
    the parameters as a list. This are contained in the 'packed_vals' list.
    Each unpacking function the removes corresponding values from the
    packed_vals list, which is returned and sent to the next unpacking function
    etc. When the white_noise unpack function has run the packed_vals list
    should be empty. (maybe another source of the memory leak?)"""

    n = 0 # current position of the unpacking sequence in the packed_vals list
    freqs_out , n   = fit_dict['fit']['mode_freqs']['unpack_func'](        packed_vals, n, fit_dict)
    heights_out , n = fit_dict['fit']['mode_heights']['unpack_func'](      packed_vals, n, fit_dict, freqs = freqs_out)
    widths_out , n  = fit_dict['fit']['mode_widths']['unpack_func'](       packed_vals, n, fit_dict, freqs = freqs_out, nu_max = fit_dict['star']['nu_max'])
    freqs_out , n   = fit_dict['fit']['mode_splits']['unpack_func'](       packed_vals, n, fit_dict, freqs = freqs_out)
    incls_out , n   = fit_dict['fit']['mode_incls']['unpack_func'](        packed_vals, n, fit_dict)
    hexp_out , n    = fit_dict['fit']['harvey_exponents']['unpack_func'](  packed_vals, n, fit_dict)
    hfre_out , n    = fit_dict['fit']['harvey_frequencies']['unpack_func'](packed_vals, n, fit_dict)
    hpow_out , n    = fit_dict['fit']['harvey_powers']['unpack_func'](     packed_vals, n, fit_dict)
    wno_out , n     = fit_dict['fit']['white_noise']['unpack_func'](       packed_vals, n, fit_dict)

    return fit_dict['fit']['x_enns'], fit_dict['fit']['x_ells'], fit_dict['fit']['x_emms'], freqs_out, heights_out, widths_out, incls_out, hexp_out, hfre_out, hpow_out, wno_out

def get_fit_dict(settings):
    """Checking and opening fit dictionary"""

    fname = get_me_this_file(settings, name = settings.star+'_output.npz', ext = '*output.npz')

    print("Trying to read %s." % (fname))
    try:
        fit_dict = np.load(fname)['fit_dict'].item()
    except:
        print("Couldn't load %s. Try something else" % (fname))
        print('Exiting...')
        sys.exit()
    print('Success!')

    return fit_dict


def setup_parameters(fit_dict, settings):
    """Extracts the initial guesses for each parameter and casts them in a
    parameterized form. These parameterized values are fed to the MCMC sampler
    which starts walking through the available parameter space. At each step the
    values are then unpacked into a simple form again, using the
    unpack_parameters function which is then used to compute the expectation value.

    Will use the specified file if a full path name is given. If a basename is
    given it will check if the file exists and then use that. Otherwise it will
    search for a *_initial_guesses.txt file in the main directory which should
    have been specified in the function call arguments."""



    """It is possible to request the use of previously calculated chains. If the
    current dimensions of the parameter space and the number of walkers match the
    those of the last run, the final positions of the walkers will be used as the
    initial starting points for the current run."""
    if settings.mcmc_burnt:

        print('Trying to load last chain position...')
        pos = fit_dict['fit']['last_chain_state']
        llim, ulim = np.array([]), np.array([])
        for key in fit_dict['parameter_keys']['mode_fit_keys']+fit_dict['parameter_keys']['bkg_keys']:
            llim, ulim = np.append(llim, fit_dict['fit'][key]['llim']), np.append(ulim, fit_dict['fit'][key]['ulim'])
        if np.shape(pos) != (settings.mcmc_wlkrs, fit_dict['fit']['ndim']):
            print("Number of walkers and/or fit dimensions don't match previous chain state.")
            print("Exiting...")
            sys.exit()
        print('Success!')
        return pos, llim, ulim

    else:
        #    ==============================================================================
        #     The following only runs if -mcmc_burnt is False
        #    ==============================================================================
       	fname = get_me_this_file(settings, name = settings.star+'_initial_guesses.txt', ext = '*_initial_guesses.txt')

        for i, k in enumerate(fit_dict['parameter_keys']['mode_fit_keys']):
            fit_dict['fit'][k]['init_guess'] = np.genfromtxt(fname, usecols = range(2, 7)).T[i]

        for i, k in enumerate(['enns', 'ells']):
            fit_dict['fit'][k] = np.genfromtxt(fname, usecols = range(2)).T[i].astype(int)

        fit_dict['fit']['x_ells'] = expand(fit_dict['fit']['ells'],fit_dict['fit']['ells']).astype(int)
        fit_dict['fit']['x_enns'] = expand(fit_dict['fit']['ells'],fit_dict['fit']['enns']).astype(int)
        fit_dict['fit']['x_emms'] = np.hstack(np.array([[x for x in np.arange(2*l+1)-l] for l in fit_dict['fit']['ells']])).astype(int)

        if settings.mcmc_wlkrs%2 != 0:
            print """WARNING: Number of walkers must be even."""
            print """The current number of walkers is %i, adding 1 walker to compensate.""" % (settings.mcmc_walkers)
            settings.mcmc_wlkrs += 1

	# These functions compute the initial random positions of the walkers.
	# Also updates the llim and ulim keys in the fit_dict.
	param_funcs = [mode_frequency_parameterization,
	 	           mode_height_parameterization,
	 	           mode_width_parameterization,
		           mode_splitting_parameterization,
		           mode_inclination_parameterization,
		           harvey_exponents_parameterization,
		           harvey_frequencies_parameterization,
		           harvey_powers_parameterization,
		           white_noise_parameterization]

	# Calls each function in param_funcs, thereby updating the llim and ulim
	# keys in the fit_dict and combines all the fit parameters into a single list
	pos = np.concatenate(tuple(pfunc(fit_dict,settings) for pfunc in param_funcs), axis = 1)

        llim, ulim = np.array([]), np.array([])

	fit_dict['fit']['ndim'] = 0

        for key in fit_dict['parameter_keys']['mode_fit_keys']+fit_dict['parameter_keys']['bkg_keys']:

            llim, ulim = np.append(llim, fit_dict['fit'][key]['llim']), np.append(ulim, fit_dict['fit'][key]['ulim'])

	    # Counting up the number of dimensions of the fit, based on the parameterized variables
            fit_dict['fit']['ndim'] += fit_dict['fit'][key]['npars']

            # reset chain, chain percentiles
            fit_dict['fit'][key]['chain'] = np.array([])
            fit_dict['fit'][key]['chain_percentiles'] = np.array([])

        # reset likelihoods
        fit_dict['fit']['lnprobabilities'] = np.array([])
	fit_dict['fit']['steps'] = np.array([])

        if settings.mcmc_wlkrs < 2*fit_dict['fit']['ndim']:
            print """WARNING: EMCEE requires at least twice the number of walkers as fit parameters, and that the number of walkers is even."""
            print """The current number of parameters in the fit is: %i""" % (fit_dict['fit']['ndim'])
            print """Adjust the -mcmc_walkers setting to at least %i and run PME again in -peakbagging mode.""" % (fit_dict['fit']['ndim']*2)
            print """Exiting..."""
            sys.exit()

    return pos, llim, ulim


def peakbagging(pos, llim, ulim, priors, fit_dict, settings):
    """This is the fun part. """

    x, y = fit_dict['spectrum']['freq'], fit_dict['spectrum']['power']

    """Initialize the sampler that will accept the initial positions of the walkers"""

    sampler = emcee.EnsembleSampler(settings.mcmc_wlkrs,
                                    fit_dict['fit']['ndim'],
                                    pb_lnprob,
                                    args=(x, y, llim, ulim, priors, fit_dict, settings),
                                    threads = settings.mcmc_thrds)

    if not settings.mcmc_burnt:
	print "Saving dictionary with initial setup"
	np.savez(os.path.join(settings.directory, settings.star+'_output.npz'), fit_dict = fit_dict)

    """Office of Silly Steps, department of Ministry of Silly Walks"""
    if settings.mcmc_substps == 0:
        print("Running sampler in ALL-OR-NOTHIN' mode! Here goes!")
        steps, substeps = settings.mcmc_stps, settings.mcmc_stps
        if settings.samples > settings.mcmc_stps:
           print("sample_posterior number must be <= than total number of steps.")
           print("Exiting...")
           sys.exit()
        
    elif settings.mcmc_substps > 0:
        print('Running sampler in chunks')
        steps, substeps = settings.mcmc_stps, settings.mcmc_substps
    	if settings.mcmc_stps < settings.mcmc_substps:
           print("Substep number must be <= than total number of steps.")
           print("Exiting...")
           sys.exit()
        if settings.samples >= settings.mcmc_stps:
           print("sample_posterior number must be <= than total number of steps.")
           print("Exiting...")
           sys.exit()
        elif settings.samples >= settings.mcmc_substps:
           print("sample_posterior number must be <= total number of substeps.")
           print("Exiting...")
           sys.exit()
    else:
	print "You requested negative substeps?!? This is unpossible! Goodbye..."
        sys.exit()

    t1 = tm.time()

    s = 1

    while steps > 0:

        ts = tm.time()
        sampler.run_mcmc(pos, substeps)
        t2 = tm.time()
        print('Step %i done. Time taken: %.4f sec, avg. time per eval. %.5f sec.' % (s, (t2-ts), (t2-ts)/(substeps*settings.mcmc_wlkrs)))

        pos = savefit(sampler, substeps, fit_dict, settings)

        steps -= substeps

        if steps < substeps: substeps = steps

        s += 1

        if settings.mcmc_thrds > 1:
            # when using multiprocessing the workers don't release the used memory
            # at the end of a run, so running multiple loops gradually eats up available memory.
            # This terminates all the workers in the pool, freeing the used memory,
            # downside is you have to reset the sampler every time. bit messy in my opinion
            # May become redundant if this is ever fixed in future versions of emcee.
            sampler.pool.terminate()

        sampler = emcee.EnsembleSampler(settings.mcmc_wlkrs,
                                        fit_dict['fit']['ndim'],
                                        pb_lnprob,
                                        args=(x, y, llim, ulim, priors, fit_dict, settings),
                                        threads = settings.mcmc_thrds)

    print('Sampler done. Time taken: %.3f hrs , avg. time per step %.4f sec, avg. time per eval. %.5f sec.' % ((t2-t1)/60/60, (t2-t1)/settings.mcmc_stps, (t2-t1)/(settings.mcmc_stps*settings.mcmc_wlkrs)))

def pb_lnprob(fitvalues, x, y, llim, ulim, priors, fit_dict, settings):

    lp = lnprior(fitvalues, llim, ulim, priors)

    if np.isinf(lp) or np.isnan(lp): return -np.inf

    enns, ells, emms, freqs, heights, widths, incls, h_exponent, h_frequency, h_power, w_noise = unpack_parameters(fitvalues, fit_dict, settings)

    expt = np.array(flops.spectral_model(x,
                                         ells, abs(emms), freqs, heights, widths,
                                         incls*np.pi/180.0, h_exponent, h_frequency, h_power, w_noise))

    lnlikelihood = lnlike(expt, y)

    if np.isinf(lnlikelihood) or np.isnan(lnlikelihood): return -np.inf

    return lp + lnlikelihood


def lnprior(arr, llim, ulim, priors = None):
    """Evaluates any priors, uniform or otherwise and adds it to the likelihood.
    """

    """The default setting for all parameters is a uniform lnprior of 0"""
    lnprior = 0

    """If EMCEE attempts to evaluate the likelihood beyond the limits of the
    variable the lnprior is set to -inf, telling EMCEE to skip this step.
    Note that the limits should encompass the prior, you may need to adjust the
    llim/ulim settings for the relevant parameter when PME is called.
    """
    if any(arr < llim) or any(arr > ulim):
        return -np.inf

    """If the priors array is not empty the contents will be searched for the
    parameter with a non-zero prior

    If the entry p in the prior array is None, meaning no prior is requested for
    this parameter, it will be skipped.

    Else if a prior (p[0]!=None) is requested for a given parameter arr[i],
    the value of that parameter is checked to see if it lies outside the region
    where the prior is defined. If this is the case a constant is assumed as a
    prior. This constant is set to the value of the prior at the edge of its
    range, which avoids discontinueties for the walkers.

    Else the prior will be evaluated at the location arr[i] at which EMCEE is
    trying to evaluate the likelihood.

    """
    if priors != None: # are any priors requested?
        for i, p in enumerate(priors): # searching the list of priors

            if p[0] is None:
                continue

            elif (arr[i] < min(p[0][0, :])) or (arr[i] > max(p[0][0, :])):
                lnprior += p[0][1, -1]
                continue

            else:
                idx = np.argmin(abs(arr[i]-p[0][0, :]))
                lnprior += p[0][1, idx]

    return lnprior


def lnlike(expt, y, s = 1):
    """returns the likelihood of the powerspectrum being produced by a given model
    (expt). If not binning is done, flops.lnlikelihood is used (primarily in
    -peakbagging), otherwise likelihood function is adjusted to acount for smoothing."""
    if any(np.isnan(expt)): return -np.inf
    elif any(np.isinf(expt)): return -np.inf
    elif any(expt <= 0.0): return -np.inf
    else:
        if s == 1:
            return flops.lnlikelihood(expt, y) #
        else:
            return np.sum((s-1.0)*np.log(s)-np.log(gamma_func(s))+(s-1.0)*np.log(y)-s*np.log(expt) - s*y/expt)



def identify_input():
    """Parses the inputs provided by the user"""

    p = argparse.ArgumentParser(description = """
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Some intilligent and up to date help info should go here
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        """,
        epilog = """
        Some additional useful info should go here.
        """)

    #==============================================================================
    #  Options for the general control of the program
    #==============================================================================
    p.add_argument('directory'                  , help="""directory containing data file"""                       , type = str)
    p.add_argument('star'                       , help="""abreviation for the star in question"""                 , type = str)
    p.add_argument('-autoguess'                 , help="""Set the operation mode to autoguess."""                 , action = 'store_true')
    p.add_argument('-peakbagging'               , help="""Set the operation mode to peakbagging. Must have
                                                       run -autoguess before using this option"""                 , action = 'store_true')
    p.add_argument('-plot'                      , help="""Set the operation mode to plot. Plots the
                                                       results of the peakbagging run. Must have run with the
                                                       -peakbagging option first duh! """                         , action = 'store_true')
    p.add_argument('-output'                    , help ="""Write output to ascii file"""                          , action = 'store_true')

    #==============================================================================
    #  Options for the general control of the program
    #==============================================================================
    p.add_argument('-sample_posterior'          , help="""Number of samples to draw from posterior distribution.
                                                       Use with -output to sample the posterior and generate
                                                       reliable estimates of the percentiles of the marginalized
                                                       parameters"""                                              , type = int, dest = 'samples', default = 2)
    p.add_argument('-ps_file'                , help="""For -autoguess and -peakbagging this input may be
                                                       used to specify the file name of the power spectrum.
                                                       If a base filename or wildcard is provided, the file is
                                                       assumed to be in 'directory'. If no input is
                                                       provided 'directory' will be searched for a file
                                                       containing the default wildcard '*.pow'."""                , type = str , dest = 'input', default = '')

    #==============================================================================
    # Options mostly specific to the -autoguess module.
    #==============================================================================
    p.add_argument('-nu_max_0'                  , help="""Value of solar nu_max in muHz, used in calculating
                                                       initial guesses for background fit. (AG)"""                , type = float, dest = 'nu_max_0', default = 3150)
    p.add_argument('-nu_gran_0'                 , help="""Frequency of the solar granulation term, try
                                                       adjusting this if the background fit fails, but keep it
                                                       between 550-650.(AG)"""                                    , type = float, dest = 'nu_gran_0', default = 620)
    p.add_argument('-n_harvey'                  , help="""Number of Harvey-like terms to include in the fit.
                                                       Note: does not currently work well if the spectrum is
                                                       truncated below the granulation timescale, and you also
                                                       want to fit 2 harvey laws. Only needs to be set for
                                                       -autoguess, -peakbagging should detect it automatically.
                                                        (AG)"""                                                  , type = int, dest = 'n_harvey', default = 2)
    p.add_argument('-bins'                      , help="""Binning factor for background fit, keep it
                                                       large for speed. (AG)"""                                  , type = int, dest = 'bins', default = 100)
    p.add_argument('-nu_max'                    , help="""Stellar nu_max, you should only need to specify this
                                                        if the automatic detection fails. (AG)"""                , type = float, dest = 'nu_max', default = -9999)
    p.add_argument('-l_sep:'                    , help="""Stellar large frequency separation. (AG)"""            , type = float, dest = 'l_sep', default = -9999)
    p.add_argument('-refit','-r'                , help="""Refit the l=0 modes to get a better initial guess
                                                       for the mode line widths. This is recommended if many
                                                       peaks have been added manually to the output file.
                                                       (AG)"""                                                   , dest = 'refit', action = 'store_true')
    p.add_argument('-add_peaks'                 , help="""To manually add peaks to the initial guess, specify
                                                       the frequency and ell value (if known) in a list. Valid
                                                       formats are alternating frequency and ell, i.e.,
                                                       "-add_peaks l,nu,l,nu..." (or vice versa) or two
                                                       consecutive series "-add_peaks l,l,l,nu,nu,nu"
                                                       (or vice versa). Seperators must be commas.
                                                       Modes may be added to *initial_guesses.txt
                                                       file once it has been created, in which case the -refit
                                                       option should be used (AG)"""                             , type = str, dest = 'add_peaks', default = '')
    p.add_argument('-intrvl_llim'               , help="""Number of half radial orders below nu_max to look
                                                       at and identify. Note: Can also be a point (in muHz)
                                                       along the frequency axis. (AG)"""                         , type = float, dest = 'intrvl_llim', default = 10)
    p.add_argument('-intrvl_ulim'               , help="""Number of half radial orders above nu_max to look
                                                       at and identify. Note: Can also be a point (in muHz)
                                                       along the frequency axis. (AG)"""                         , type = float, dest = 'intrvl_ulim', default = 10)
    p.add_argument('-min_sign'                  , help="""Minimum significance level for selecting new peaks,
                                                       lower values finds more peaks, but risk picking up
                                                       noise. Try lowering if only a few l=2 modes are found.
                                                       (AG)"""                                                   , type = float, dest = 'min_sign', default = 50)
    p.add_argument('-check','-c'                , help="""Use this to make a quick check to see if the
                                                       initial guesses make sense. If autoguess has not
                                                       been run, this will simply show a plot of the power
                                                       spectrum, with a smoothed spectrum overlayed. (AG)"""     , action = 'store_true')


    #==============================================================================
    # Options mostly specific to the -peakbagging module.
    #==============================================================================
    p.add_argument('-mcmc_walkers'              , help="""Number of walkers to use for the MCMC chain
                                                       (must be even), and at least 2x number of
                                                       of variables in the fit. (AG,PB)"""                        , type = int, dest = 'mcmc_wlkrs', default = 100)
    p.add_argument('-mcmc_threads'              , help="""Number of threads to use for computing
                                                       the MCMC chains. (AG,PB)"""                                , type = int, dest = 'mcmc_thrds', default = 8)
    p.add_argument('-mcmc_burnt','-b'           , help="""Flag to use previously computed
                                                       walker positions. Used for splitting up
                                                       very long runs if a maximum computing time
                                                       is enforced by the system."""                              , action = 'store_true')
    p.add_argument('-mcmc_start_spread'         , help="""The initial scatter of the walker positions around
                                                       the initial guess, must be between 0 and 1. Note that if
                                                       the scatter is unacceptable compared to the limits of the
                                                       fit, the limit will be enforced. If this is an issue an
                                                       option is to change the -*l_lims or -*u_lims options for
                                                       the parameter of interest."""                              , dest = 'mcmc_sprd', default = 0.1)
    p.add_argument('-mcmc_substeps'             , help="""Number of steps after which walker
                                                       progress is saved. Must be smaller than mcmc_steps"""
									                                          , type = int, dest = 'mcmc_substps', default = 0)
    p.add_argument('-mcmc_steps'                , help="""Total number of steps for the walkers
                                                       to take in the MCMC chain."""                              , type = int, dest = 'mcmc_stps', default = 20)
    p.add_argument('-prior'                     , help="""To add a prior for a paramter simply place a file with
                                                       myprior.prior filename in the star directory, and add
                                                       '-prior myprior.prior' to the -peakbagging call in the
                                                       commandline. For a list of parameter names that a valid as
                                                       file names, use the -show_prior_list option. Multiple
                                                       prior filenames can be added in succession."""             , type = str, nargs = '+')
    p.add_argument('-show_prior_list'           , help="""Prints a list of valid filenames for the priors"""      , action = 'store_true', dest = 'ask_prior')
    p.add_argument('-no_trim'                   , help="""Disable trimming of the upper part of the spectrum, 
						       this will increase the evaluation time, but provide a 
						       better estimate of the white noise level """               , action = 'store_true', dest = 'no_trim')
    p.add_argument('-bkg_exponent'              , help="""Parameterizations for background terms are described in
                                                       PME_background_parameterization.py"""                      , type = str, dest = 'bkg_exp'  , choices = harvey_exponents_parameterization(plist = True)  , default = 'independent')
    p.add_argument('-bkg_power'                 , help="""Parameterizations for background terms are described in
                                                       PME_background_parameterization.py"""                      , type = str, dest = 'bkg_power', choices = harvey_powers_parameterization(plist = True)     , default = 'independent')
    p.add_argument('-bkg_frequencies'           , help="""Parameterizations for background terms are described in
                                                       PME_background_parameterization.py"""                      , type = str, dest = 'bkg_freqs', choices = harvey_frequencies_parameterization(plist = True), default = 'independent')
    p.add_argument('-bkg_whitenoise'            , help="""Parameterizations for background terms are described in
                                                       PME_background_parameterization.py"""                      , type = str, dest = 'bkg_wn'   , choices = white_noise_parameterization(plist = True)       , default = 'independent')
    p.add_argument('-mode_frequencies'          , help="""Parameterizations for mode frequencies are described in
                                                       PME_mode_frequencies_parameterization.py"""                , type = str, dest = 'md_freqs' , choices = mode_frequency_parameterization(plist = True)    , default = 'independent')
    p.add_argument('-mode_heights'              , help="""Parameterizations for mode heights are described in
                                                       PME_mode_heights_parameterization.py"""                    , type = str, dest = 'md_hghts' , choices = mode_height_parameterization(plist = True)       , default = 'independent')
    p.add_argument('-widths'                    , help="""Parameterizations for mode widths are described in
                                                       PME_mode_widths_parameterization.py"""                     , type = str, dest = 'widths'   , choices = mode_width_parameterization(plist = True)        , default = 'polynomial')
    p.add_argument('-width_poly'                , help="""If -widths is set to 'polynomial' this sets the
                                                       polynomial order for width as a function of frequency"""   , type = int, dest = 'width_poly', default = 5)
    p.add_argument('-splittings'                , help="""Parameterizations for mode splittings are described in
                                                       PME_mode_splittings_parameterization.py"""                 , type = str, dest = 'split'    , choices = mode_splitting_parameterization(plist = True)    , default = 'share_all')
    p.add_argument('-inclinations'              , help="""Parameterizations for mode inclinations are described in
                                                       PME_mode_inclinanations_parameterization.py"""             , type = str, dest = 'incl'     , choices = mode_inclination_parameterization(plist = True)  , default = 'share_all')


    #==============================================================================
    # Options for the -peakbagging module which deal with setting lower and upper
    # limits for each parameter. Note that the lower and upper limits are given separately.
    #==============================================================================

    p.add_argument('-freq_u_lims','-ful'        , help="""Relative(!) limits for ALL mode frequencies. This should
                                                      be less than the small separation minus 2x the splitting
                                                      (roughly). Otherwise the walkers will start to confuse
                                                      mode peaks in the l=2,0 pair.
                                                      """                                                         , nargs='+', type = float, dest = 'freqs_u_lims', default = [3])
    p.add_argument('-freq_l_lims','-fll'        , help="""Relative(!) limits for ALL mode frequencies. This should
                                                       be less than the small separation minus 2x the splitting
                                                       (roughly). Otherwise the walkers will start to confuse
                                                       mode peaks in the l=2,0 pair.
                                                       """                                                        , nargs='+', type = float, dest = 'freqs_l_lims', default = [3])
    p.add_argument('-height_u_lims','-hul'      , help="""Relative(!) limits for ALL mode heights.
                                                       """                                                        , nargs='+', type = float, dest = 'heights_u_lims' , default = [1e3])
    p.add_argument('-height_l_lims','-hll'      , help="""Relative(!) limits for ALL mode heights.
                                                       """                                                        , nargs='+', type = float, dest = 'heights_l_lims' , default = [2e-19])
    p.add_argument('-width_u_lims','-wul'       , help="""Relative(!) limits for ALL mode widths. If -widths is
                                                       set simultaneously to 'independent' the lower limit may
                                                       be automatically modified so that the absolute lower limit
                                                       is > 0. If -widths is set to polynomial it is probably
                                                       best to allow a very large range since this limit applies
                                                       to all the polynomial coefficients.
                                                       """                                                        , nargs='+', type = float, dest = 'widths_u_lims'  , default = [1e5])
    p.add_argument('-width_l_lims','-wll'       , help="""Relative(!) limits for ALL mode widths. If -widths is
                                                       set simultaneously to 'independent' the lower limit may
                                                       be automatically modified so that the absolute lower limit
                                                       is > 0. If -widths is set to polynomial it is probably
                                                       best to allow a very large range since this limit applies
                                                       to all the polynomial coefficients.
                                                       """                                                        , nargs='+', type = float, dest = 'widths_l_lims'  , default = [1e5])
    p.add_argument('-split_u_lims','-sul'       , help="""Relative(!) limits for ALL mode splittings.
                                                       """                                                        , nargs='+', type = float, dest = 'splits_u_lims'  , default = [5])
    p.add_argument('-split_l_lims','-sll'       , help="""Relative(!) limits for ALL mode splittings.
                                                       """                                                        , nargs='+', type = float, dest = 'splits_l_lims'  , default = [0])
    p.add_argument('-incl_u_lims','-iul'        , help="""Absolute(!) limits for the stellar inclination.
                                                       """                                                        , nargs='+', type = float, dest = 'incls_u_lims'   , default = [90])
    p.add_argument('-incl_l_lims','-ill'        , help="""Absolute(!) limits for the stellar inclination.
                                                       """                                                        , nargs='+', type = float, dest = 'incls_l_lims'   , default = [0])
    p.add_argument('-bkg_exp_u_lims','-beul'    , help="""Absolute(!) limits for exponents of the Harvey laws.
                                                       Lower limit should be > 0.
                                                       """                                                        , nargs='+', type = float, dest = 'bkg_exp_u_lims' , default = [10])
    p.add_argument('-bkg_exp_l_lims','-bell'    , help="""Absolute(!) limits for exponents of the Harvey laws.
                                                       Lower limit should be > 0.
                                                       """                                                        , nargs='+', type = float, dest = 'bkg_exp_l_lims' , default = [1e-5])
    p.add_argument('-bkg_freq_u_lims','-bful'   , help="""Absolute(!) limits for frequencies (widths) of the
                                                       Harvey laws. The lower limit should be >=0, the upper
                                                       limit should be < the Nyquist frequency. """               , nargs='+', type = float, dest = 'bkg_freq_u_lims', default = [8300])
    p.add_argument('-bkg_freq_l_lims','-bfll'   , help="""Absolute(!) limits for frequencies (widths) of the
                                                       Harvey laws. The lower limit should be >=0, the upper
                                                       limit should be < the Nyquist frequency. """               , nargs='+', type = float, dest = 'bkg_freq_l_lims', default = [2e-19])
    p.add_argument('-bkg_pow_u_lims','-bpul'    , help="""Absolute(!) limits for the power of the Harvey laws.
                                                       Lower limit should be > 0. """                             , nargs='+', type = float, dest = 'bkg_pow_u_lims' , default = [1e20])
    p.add_argument('-bkg_pow_l_lims','-bpll'    , help="""Absolute(!) limits for the power of the Harvey laws.
                                                       Lower limit should be > 0. """                             , nargs='+', type = float, dest = 'bkg_pow_l_lims' , default = [2e-19])
    p.add_argument('-bkg_WN_u_lims','-bwnul'    , help="""Absolute(!) limits for the white noise level.
                                                       The lower limit should be > 0, the upper limit is pretty
                                                       arbitrary as this quantity is very well constrained by
                                                       the data. """                                              , nargs='+', type = float, dest = 'bkg_WN_u_lims'  , default = [100])
    p.add_argument('-bkg_WN_l_lims','-bwnll'    , help="""Absolute(!) limits for the white noise level.
                                                       The lower limit should be > 0, the upper limit is pretty
                                                       arbitrary as this quantity is very well constrained by
                                                       the data. """                                              , nargs='+', type = float, dest = 'bkg_WN_l_lims'  , default = [2e-19])


    if len(sys.argv) == 1:
        print('No input provided, exiting...')
        sys.exit()

    S = p.parse_args()

    return S


def test_objective(p, f, x, y, current_model):
    return -lnlike(simple_lorentzian(p, f, x)+current_model, y)

def simple_lorentzian(p, f, x):
    p0,p1 = np.exp(p)
    return p0 / p1 / (1.0 + (2.0*(x-f)/p1)**2.0)

def divide_out_peak(x, y, model, smth, f, significance):
    """Function for fitting the peaks in the power spectrum individually, and
    dividing it out. Also returns the difference in the likelihoods."""
    logL_old = lnlike(model, y)

    freq, height, width = x[np.argmin(abs(x-f))], np.log(smth[np.argmin(abs(x-f))]), np.log(1.0)

    fidx = between(x, freq-0.75, freq+3.0)

    out = optimize.minimize(test_objective, [height, width], method = 'Nelder-Mead', args = (freq, x[fidx], y[fidx], model[fidx]))

    model += simple_lorentzian(out.x, freq,x)

    logL_new = lnlike(model, y)

    significance = logL_new - logL_old

    return out, model, significance


def add_peaks(idx, fit_dict, settings):
    """This function identifies peaks in the power spectrum, but does not assign
    ells to them. It functions by identifying the highest peak in the smoothed
    spectrum (lorentzian smoothing kernel), fitting a lorentzian to the peak and
    dividing it out. The significance of removing the peak is the evaluated by
    comparing the likelihood of the model before and after. Once the change in
    likelihood decreases to a set level min_sign, the loop will stop and no more
    peaks are added. This list is then passed to the function that assigns ells.

    The function first runs through modes that have already been added to dictionary.
    If this is the first time -autoguess is being run, this list is of course 0.
    This ensures that modes from the *initial_guesses.txt or modes added in the
    command line are taken into account before the the automatic peak detection
    takes place.
    """
    f, p = fit_dict['spectrum']['freq'],fit_dict['spectrum']['power']

    significance, N_smooth = np.inf, 5000

    dx, M = np.median(np.diff(f)), min(len(f), N_smooth)

    window = simple_lorentzian([1.0, 0.2], f[M/2], f[:M])

    model = fit_dict['spectrum']['model']['bkg'].copy()

    init_spec = np.convolve(p / model, window*dx, mode='same')

    for i, nu in enumerate(fit_dict['fit']['mode_freqs']['init_guess']):

        if fit_dict['fit']['ells'][i] == 3: continue

        smth = np.convolve(p / model, window*dx, mode='same')

        nu = f[smth == max(smth[between(f, nu-1.5, nu+1.5)])]

        out, model, significance = divide_out_peak(f, p, model, smth, nu, significance)

        fit_dict['fit']['mode_freqs']['init_guess'][i]   = nu
        fit_dict['fit']['mode_heights']['init_guess'][i] = np.exp(out.x[0])
        fit_dict['fit']['mode_widths']['init_guess'][i]  = np.exp(out.x[1])

    if not settings.refit:

	print """Finding the peaks in the spectrum..."""

        significance = np.inf

        x, y, m = f[idx], p[idx], model[idx]

        dx, M = np.median(np.diff(x)), min(len(x), N_smooth)

        window = simple_lorentzian([1.0, 0.2], x[M/2], x[:M])

        while significance >= settings.min_sign:

            smth = np.convolve(y/m, window*dx, mode='same')

            out, m, significance = divide_out_peak(x, y, m, smth, x[np.argmax(smth)], significance)

            if significance >= settings.min_sign:
                fit_dict['fit']['mode_freqs']['init_guess']   = np.append(fit_dict['fit']['mode_freqs']['init_guess']  , x[np.argmax(smth)])
                fit_dict['fit']['mode_heights']['init_guess'] = np.append(fit_dict['fit']['mode_heights']['init_guess'], np.exp(out.x[0]))
                fit_dict['fit']['mode_widths']['init_guess']  = np.append(fit_dict['fit']['mode_widths']['init_guess'] , np.exp(out.x[1]))

                for k in ['enns', 'ells', 'emms']: fit_dict['fit'][k] = np.append(fit_dict['fit'][k], -1)

    #==============================================================================
    # sorting with frequency
    #==============================================================================
    sidx = np.argsort(fit_dict['fit']['mode_freqs']['init_guess'])
    for k in ['enns', 'ells', 'emms']: fit_dict['fit'][k] = fit_dict['fit'][k][sidx]
    for k in ['mode_freqs', 'mode_heights', 'mode_widths']: fit_dict['fit'][k]['init_guess'] = fit_dict['fit'][k]['init_guess'][sidx]

    print """Found a total of %i peaks""" % (len(fit_dict['fit']['mode_freqs']['init_guess']))

    #==============================================================================
    # if running in refit mode, PME re-assigns enns and writes the contents of
    # fit_dict to the *output.npz and *initial_guesses.txt file, and then exits.
    #==============================================================================
    if settings.refit:
        assign_enns(fit_dict)
        write_init_file(fit_dict, settings, True)
        sys.exit()


    peakfig = plt.figure(figsize = (20, 15))
    pax = peakfig.add_subplot(111)
    for k in fit_dict['fit']['mode_freqs']['init_guess']: pax.axvline(k, color = 'k')
    pax.fill_between(f, np.zeros_like(f), init_spec, color = 'b')
    pax.fill_between(x, np.zeros_like(x), smth, color = 'g')
    pax.set_ylim(-0.1, pax.get_ylim()[1])
    pax.set_xlim(x[0]*0.90, x[-1]*1.10)
    pax.set_ylabel('Power [ppm$^2$/$\mu$Hz]')
    pax.set_ylabel('Frequency [$\mu$Hz]')

    return peakfig


def assign_ells(fit_dict, settings):
    """This is Warricks magical ell identification algorithm, the knowledge of
    how it works has been lost to the passages of time."""

    idx = fit_dict['fit']['ells'] < 3

    flist = fit_dict['fit']['mode_freqs']['init_guess'][idx]
    ells  = fit_dict['fit']['ells'][idx]

    df = np.diff(flist)

    # guess whether separation is a small separation
    half_seps = np.floor((df + fit_dict['star']['Del_nu']/4.0)/(fit_dict['star']['Del_nu']/2.0)).astype(int)

    if not any(half_seps == 0):
        print("Couldn't find any small separations.\n" + \
              "Automatic classification won't work.  Aborting.")
        sys.exit()

    # start at the first small separation
    print("Assigning first small separation as l=2,0 pair.")
    i1 = np.arange(len(df))[half_seps == 0][0]

    ells[i1], ells[i1+1] =  2, 0

    print("Working upwards in frequency...")

    M = len(flist)
    for i in range(i1+2, M-1):

        if ells[i] > -1:
            continue

        elif half_seps[i] == 0:
            # what if there are more small-separated modes?
            small_seps = 0
            while half_seps[i+small_seps] == 0:
                small_seps += 1

            if small_seps > 1: print("WARNING: Found " + str(small_seps) + " in a row at index " + str(i) + ".")

            if half_seps[i-1] % 2 == 1:
                ells[i]   = 3 - ells[i-1]
                ells[i+1] = 1 - ells[i-1]

            elif half_seps[i-1] % 2 == 0:
                ells[i]   = 2 + ells[i-1]
                ells[i+1] =     ells[i-1]

        elif half_seps[i] == 1:
            if half_seps[i-1]   % 2 == 1:
                ells[i]   = 1 - ells[i-1]
            elif half_seps[i-1] % 2 == 0:
                ells[i]   =     ells[i-1]
        else:
            print("WARNING:  Reached an unimplemented case.  Classification unreliable.")

    print("Identifying highest-frequency mode...")
    # this needs to be fixed
    # if it's a pair, it should have already been assigned
    if half_seps[-1] == 0:
        ells[M-2] = 2
        ells[M-1] = 0
    elif half_seps[-1] == 1:
        ells[M-1] = 1-ells[M-2]
    else:
        print("WARNING:  Reached an unimplemented case.  Classification unreliable.")

    print("Working downwards in frequency...")
    for i in range(i1-1, 0, -1):
        if ells[i] > -1:
            continue

        elif half_seps[i-1] == 0:
            if   half_seps[i] % 2 == 1:
                ells[i-1] = 3 - ells[i+1] % 2
                ells[i]   = 1 - ells[i+1] % 2
            elif half_seps[i] % 2 == 0:
                ells[i-1] = 2 + ells[i+1] % 2
                ells[i]   =     ells[i+1] % 2

        elif half_seps[i-1] == 1:
            if   half_seps[i] % 2 == 1:
                ells[i]   = 1 - ells[i+1] % 2
            elif half_seps[i] % 2 == 0:
                ells[i]   =     ells[i+1] % 2
        else:
            print("WARNING:  Reached an unimplemented case.  Classification unreliable.")

    print("Identifying lowest-frequency mode...")
    # fix this too
    if half_seps[0] == 0:
        ells[0] = 2
        ells[1] = 0
    elif half_seps[0] == 1:
        ells[0] = abs(1 - ells[1])
    else:
        print("WARNING:  Reached an unimplemented case.  Classification unreliable.")
    fit_dict['fit']['ells'][idx] = ells
    print("Assigning n values.")
    assign_enns(fit_dict)


def check_for_added_peaks(fit_dict, settings):
    """Checks to see if -add_peaks or -refit has been used. Will exit if both are used
    at the same time."""
    if (len(settings.add_peaks) > 0) and (settings.refit == False):
        f_n_l = settings.add_peaks.split(',')

        try:
            f_n_l = np.array(f_n_l).astype(float)
        except:
            print("Couldn't convert string to float, check the list of manual peaks and follow the directions in the help.")
            sys.exit()

        for b in f_n_l:
            if b < 10: fit_dict['fit']['ells']                = np.append(fit_dict['fit']['ells'], int(b))
            else: fit_dict['fit']['mode_freqs']['init_guess'] = np.append(fit_dict['fit']['mode_freqs']['init_guess'], b)

        if len(fit_dict['fit']['mode_freqs']['init_guess']) != len(fit_dict['fit']['ells']):
            fit_dict['fit']['ells'] = np.zeros_like(fit_dict['fit']['mode_freqs']['init_guess'])-1
            print("""The number of freqs and ells don't match, no ells assigned to manually added peaks.
                     Try correcting the input line, or leave it until the *initial_guesses.txt file has been generated,
                     edit the ells and then re-run with the -refit option.""")

        fit_dict['fit']['enns'] = np.zeros_like(fit_dict['fit']['ells'], dtype = int)

    elif (len(settings.add_peaks) == 0) and (settings.refit == True):
        print 'Loading previous *initial_guesses.txt'
	fname = get_me_this_file(settings, name = settings.star+'_initial_guesses.txt', ext = '*_initial_guesses.txt')
        fit_dict['fit']['enns'], fit_dict['fit']['ells'], fit_dict['fit']['mode_freqs']['init_guess'] = np.genfromtxt(fname, usecols = [0, 1, 2]).T
	fit_dict['fit']['enns'], fit_dict['fit']['ells'] = fit_dict['fit']['enns'].astype(int), fit_dict['fit']['ells'].astype(int)

    elif (len(settings.add_peaks) > 0) and (settings.refit == True):
        print """Cannot use -refit and -add_peaks at the same time. If you are adding peaks you can either add
        them in the input line as l,nu,l,nu or vice versa, or add them to the *initial_guesses.txt as -1 l nu 1 1 1 1,
        with one peak per line. If you added the peaks to *initial_guesses.txt you must re-run with the -refit option ."""
        print """Exiting..."""
        sys.exit()

    fit_dict['fit']['emms']                       = np.zeros_like(fit_dict['fit']['mode_freqs']['init_guess'],dtype = int)
    fit_dict['fit']['mode_heights']['init_guess'] = np.zeros_like(fit_dict['fit']['mode_freqs']['init_guess'])
    fit_dict['fit']['mode_widths']['init_guess']  = np.zeros_like(fit_dict['fit']['mode_freqs']['init_guess'])

    for k in ['mode_heights', 'mode_widths']:
        for i, l in enumerate(fit_dict['fit']['ells']):
            if l == 3.:
                if fit_dict['fit'][k]['init_guess'][i] <= 0:
                    fit_dict['fit'][k]['init_guess'][i] = 0.75


def write_init_file(fit_dict, settings, auto_done = False):

    if settings.refit :
        fname = get_me_this_file(settings, name = settings.star+'_initial_guesses.txt', ext = '*_initial_guesses.txt')
    else :
        fname = os.path.join(settings.directory, settings.star+'_initial_guesses.txt')
    outputfile = open(fname, 'w')
    outputfile.write(('# 1. n, 2. l, 3. Freq. [muHz], 4. Height ppm^2/[muHz], 5. width [muHz], 6. split. [muHz], 7. incl [deg]\n'))
    if len(fit_dict['fit']['mode_freqs']['init_guess']) > 0:
        for i in range(len(fit_dict['fit']['ells'])):
       	    outputfile.write(('%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n') % (fit_dict['fit']['enns'][i],
         	                                                             fit_dict['fit']['ells'][i],
                	                                                     fit_dict['fit']['mode_freqs']['init_guess'][i],
                	                                                     fit_dict['fit']['mode_heights']['init_guess'][i],
                	                                                     fit_dict['fit']['mode_widths']['init_guess'][i],
                	                                                     1,
                	                                                     45))

    outputfile.close()

    np.savez(os.path.join(settings.directory, settings.star+'_output.npz'), fit_dict = fit_dict)

def check_initial_guesses(settings):
    """This function is called with the -autoguess -check option, which is used
    to check the initial guesses for the modes. This will produce a figure with
    the smoothed spectrum and markers for each of the identified modes. Unidentified
    modes are labeled as l=-1 and are colored black. An eschelle diagram is also
    plotted with the same color coding."""
    oput_fname = get_me_this_file(settings, name = settings.star+'_output.npz', ext = '*_output.npz')
    fit_dict = np.load(oput_fname)['fit_dict'].item()
    x, y = fit_dict['spectrum']['freq'],fit_dict['spectrum']['power']

    checkfig = plt.figure(figsize = (16, 9))
    cax = checkfig.add_subplot(111)
    smth = smooth(y, window_len=0.5/np.median(np.diff(x)))
    cax.plot(x, smth, 'k', label = 'Smoothed')
    cax.set_xlabel('Frequency [$\mu$Hz]')
    cax.set_ylabel('Power [ppm$^2$/muHz]')
    cax.legend(numpoints = 1)

    print("""Generating figures based on the mode IDs in  %s""" % (settings.star+'_initial_guesses.txt'))
    ig_fname = get_me_this_file(settings, name = settings.star+'_initial_guesses.txt', ext = '*_initial_guesses.txt')
    enns, ells, freqs, heights= np.genfromtxt(ig_fname, usecols = [0, 1, 2, 3]).T
    enns, ells = enns.astype(int),ells.astype(int)

    labels = ['l = -1', 'l = 0', 'l = 1', 'l = 2', 'l = 3']

    ls = [-1, 0, 1, 2, 3]

    cols = ['k', 'b', 'g', 'r', 'm']

    for i in range(int(max(ells)+2)):
        if len(freqs[ells == ls[i]]) == 0: continue
        cax.plot(freqs[ells == ls[i]], heights[ells == ls[i]] + noiselevel(y), marker = 'o', ms = 15, color =cols[i], label = labels[i])

    cax.legend(numpoints = 1)

    checkfig.savefig(settings.directory +'/'+settings.star+'_checkfig.png', bbox_inches = 'tight')

    cax.set_xlim(min(freqs)-400, max(freqs)+400)
    cax.set_ylim(0, max(heights)+10)



    plot_eschelle(fit_dict, settings, enns, ells, freqs)

    plt.show()
    sys.exit()


def define_freq_interval(fit_dict, settings):
    """Returns a list of indices for the requested frequency interval.
    intrvl_llim/intrvl_ulim can be either an integer number of radial orders, or
    frequencies limits. The number of radial orders are estimated based on the
    local minima of the smoothed spectrum, in between the l=2,0 pair and the l=1
    mode."""

    print """Attempting to define the fit interval..."""

    freq, power = fit_dict['spectrum']['freq'],fit_dict['spectrum']['power']

    if (settings.intrvl_llim > 100.0) and (settings.intrvl_ulim > 100.0):
        # If the number of requested radial orders is >100 it is assumed that it is instead a frequency limit

        if settings.intrvl_ulim < settings.intrvl_llim:
            # Check to see if input is in the correct order
            print("Upper limit of requested frequency interval is < lower limit. Adjust and try again. Exiting..." )
            sys.exit()

        pllim, pulim = settings.intrvl_llim, settings.intrvl_ulim

    elif settings.refit:
        # If the -refit option is used the limits are reset to be +/- half a large separation from the lowest and highest peak frequencies
        new_freqs = sorted(fit_dict['fit']['mode_freqs']['init_guess'])

        pllim, pulim = new_freqs[0]-fit_dict['star']['Del_nu']/2.0, new_freqs[-1]+fit_dict['star']['Del_nu']/2.0

    elif settings.peakbagging:

        new_freqs = sorted(fit_dict['fit']['mode_freqs']['init_guess'])

        pllim, pulim = new_freqs[0]-fit_dict['star']['Del_nu']/2.0, new_freqs[-1]+fit_dict['star']['Del_nu']/2.0
	
	idx_cr = fit_dict['spectrum']['model']['bkg']-fit_dict['spectrum']['model']['bkg'][-1] > fit_dict['spectrum']['model']['bkg'][-1]

	fcut = max([max(new_freqs),fit_dict['spectrum']['freq'][idx_cr][-1]])+2000
	
	return fcut

    else:
        # If the number of requested radial orders is <=100 it is assumed that they are in fact radial orders and not frequency limits
        # The spectrum is strongly smoothed and the minima between l=0 and l=1 are found. The counting of the radial orders starts from the
        # minimum closests to nu_max and progresses outward to the respective lower and upper limits, in steps of the large separation.
        pitidx = pitfinder(smooth(power, window_len=5.0/np.median(np.diff(freq))))

        pitfreqs = freq[pitidx]

        center_pit_idx = np.argmin(abs(pitfreqs-fit_dict['extra_params']['pmode_env_freq']['init_guess']))

        pllim, pulim = pitfreqs[center_pit_idx]-settings.intrvl_llim*fit_dict['star']['Del_nu'], pitfreqs[center_pit_idx]+settings.intrvl_ulim*fit_dict['star']['Del_nu']

    idx = between(freq, pllim, pulim)

    print """Mode search interval: %.0fmuHz to %.0fmuHz""" % (round(freq[idx][0]),round(freq[idx][-1]))

    return idx


def smooth(x, t = None, window_len=9):
    """Smoothing function using the scipy gaussian_filter1d function."""
    # Various checks for the input vector
    if type(x) == type([]):
        x = np.array(x)

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    x_smth = gaussian_filter1d(x, window_len)

    return x_smth


def pitfinder(y):
    """Identifies the indices of the local minima in a 1D array. Used to automatically
    detect the frequency range to be searched for modes."""
    p4 = np.append(np.append(False, ((np.diff(y[0:-1])*np.diff(y[1:])) < 0.0)), False)

    p5 = np.append(np.append(False, (np.diff(np.diff(y)) > 0.0)), False)

    p6 = p4*p5 > 0.0

    return p6


def peakfinder(y):
    """Identifies the indices of the local maxima in a 1D array. Used when detecting
    modes for the initial guesses."""
    p1 = np.append(np.append(False, ((np.diff(y[0:-1])*np.diff(y[1:])) < 0)), False)

    p2 = np.append(np.append(False, (np.diff(np.diff(y)) < 0)), False)

    p3 = p1*p2 > 0

    return p3


def assign_enns(fit_dict):
    """Assigns pseudo-enn values starting at 0 for the first frequency in the list,
    the following ells in a series of l=0,1,2,3 are then assigned the same enn value."""
    fit_dict['fit']['enns'] = np.zeros_like(fit_dict['fit']['ells'], dtype = int)
    fit_dict['fit']['enns'][0] = 10

    for i in range(1, len(fit_dict['fit']['enns'])):

        if fit_dict['fit']['ells'][i] == 0:
            fit_dict['fit']['enns'][i] = fit_dict['fit']['enns'][i-1] + 1

        elif fit_dict['fit']['ells'][i] == 1:
            if fit_dict['fit']['ells'][i-1] >= 1:
                fit_dict['fit']['enns'][i] = fit_dict['fit']['enns'][i-1] + 1
            else: fit_dict['fit']['enns'][i] = fit_dict['fit']['enns'][i-1]

        elif fit_dict['fit']['ells'][i] == 2:
            if fit_dict['fit']['ells'][i-1] >= 2:
                fit_dict['fit']['enns'][i] = fit_dict['fit']['enns'][i-1] + 1
            else: fit_dict['fit']['enns'][i] = fit_dict['fit']['enns'][i-1]

        elif fit_dict['fit']['ells'][i] == 3:
            if fit_dict['fit']['ells'][i-1] >= 3:
                fit_dict['fit']['enns'][i] = fit_dict['fit']['enns'][i-1]
            else: fit_dict['fit']['enns'][i] = fit_dict['fit']['enns'][i-1] - 1
        else:
            fit_dict['fit']['enns'][i] = -1

    fit_dict['fit']['enns'] -= fit_dict['fit']['enns'][0]




def get_powerspectrum(settings):
    """Extract the data from power spectrum file. Will use the specified file if
    a full path name is given. If a basename is given it will check if the file
    exists and then use that. Otherwise it will search for a *.pow file in the
    specified directory.

    *** Important ***
    *** Format must be frequency in 1st column, power in 2nd ***
    *** Any header lines must be preceded by # **
    *** File may contain additional columns, but these will be ignored ***
    """

    fname = get_me_this_file(settings, name = settings.input, ext = '*.pow')

    print("Trying to read %s." % (fname))
    try:
        freq, power = np.genfromtxt(fname, dtype = (float,float), usecols = [0, 1]).T
        print('Success!')
        return freq[power > 0.0], power[power > 0.0]
    except:
        print("Couldn't load %s. Try something else" % (fname))
        print('Exiting...')
        sys.exit()



def setup_fit_dict(f, p, settings):
    """Setups up or resets the fit dictionary."""
    fit_dict = {}
    fit_dict['settings'] = settings
    fit_dict['spectrum'] = {}
    fit_dict['spectrum']['freq'], fit_dict['spectrum']['power'], fit_dict['spectrum']['model'] = f, p, {}
    fit_dict['star'] = {}
    fit_dict['star']['model'], fit_dict['star']['ID'] = {}, settings.star
    fit_dict['fit'], fit_dict['extra_params'] = {}, {}

    fit_dict['parameter_keys'] = {'mode_fit_keys':  ['mode_freqs', 'mode_heights', 'mode_widths', 'mode_splits', 'mode_incls'],
                                  'mode_no_keys':   ['enns', 'ells', 'emms'],
                                  'bkg_keys':       ['harvey_exponents', 'harvey_frequencies', 'harvey_powers', 'white_noise'],
                                  'extra_keys':     ['pmode_env_power', 'pmode_env_freq', 'pmode_env_width']}

    for k in fit_dict['parameter_keys']['extra_keys']:
        fit_dict['extra_params'][k] = {x: np.array([]) for x in ['init_guess', 'best_fit', '16th', '84th', 'chain', 'llim', 'ulim', 'npars', 'prior', 'unpack_func']}

    for k in fit_dict['parameter_keys']['mode_fit_keys']+fit_dict['parameter_keys']['bkg_keys']:
        fit_dict['fit'][k]                      = {x: np.array([]) for x in ['init_guess', 'best_fit', '16th', '84th', 'chain', 'chain_percentiles', 'llim', 'ulim', 'npars', 'prior']}

    for k in fit_dict['parameter_keys']['mode_no_keys']:
        fit_dict['fit'][k]   = np.array([])

    return fit_dict


def get_dnu_numax(fit_dict, settings):
    """A function for computing the large separation and nu_max for a solar-like
    oscillator. Can't remember exactly how it works, but it takes a section
    the spectrum and correlates it with the rest of the spectrum"""
    print """Attempting to find nu_max and the large separation"""

    f, p = fit_dict['spectrum']['freq'],fit_dict['spectrum']['power']

    if (settings.nu_max == -9999.) or (settings.l_sep == -9999.):

        idx = between(f, 0.,5000.)

        f, p = f[idx], p[idx]

        n = 12

        f, p = f[::n], p[::n]

        df = np.median(np.diff(f))

        S, B = np.zeros_like(f), np.zeros_like(f)

        delta_nu = lambda x: 0.263*x**0.772
        numax = lambda x: ((1./0.263)*x)**(1./0.772)
        g = lambda p, x: p[0]*np.exp(-(x-p[1])**2 / (2*p[2]**2))
        obj_func = lambda p, y, x: np.sum((y-g(p[:3], x)+p[3]*x+p[4])**2)

        for i, c in enumerate(f):

            minnu, maxnu=max([f[0], c-delta_nu(c)*1.1]), min([f[-1], c+delta_nu(c)*1.1])

            idx = between(f, minnu, maxnu)

            a = signal.fftconvolve(p[idx]-np.mean(p[idx]), p[idx][::-1]-np.mean(p[idx]), mode='full')

            a = a[int(len(a)/2):]

            S[i] = np.sum(a)

            idx1 = between(f[idx]-minnu, 0.7*delta_nu(c), 1.3*delta_nu(c))

            a[np.invert(idx1)] = 0

            B[:len(a)] += a

        idx = peakfinder(smooth(B, window_len = 5./df))

        dnu = f[idx][np.argmax(B[idx])]

        numax_guess = numax(dnu)

        idx = between(f, numax_guess*0.5, numax_guess*1.5)

        out = optimize.minimize(obj_func, [S[np.argmin(abs(f-numax_guess))], numax_guess,200,0,np.median(S[idx])], args = (S[idx], f[idx]), method = 'Nelder-Mead')

    if settings.nu_max == -9999.:
        fit_dict['star']['nu_max'] = out.x[1]
    
    else :
        fit_dict['star']['nu_max'] = settings.nu_max

    if settings.l_sep == -9999.:
        fit_dict['star']['Del_nu'] = dnu
    
    else:
        fit_dict['star']['Del_nu'] = settings.l_sep

    print 'Guessed nu_max = %.2f muHz and large separation = %.2f muHz' % (round(fit_dict['star']['nu_max'],2),round(fit_dict['star']['Del_nu'],2))
    return fit_dict['star']['Del_nu'], fit_dict['star']['nu_max']


def background_fit(fit_dict, settings):
    """Fits the background terms in the spectrum for the -autoguess module.
    Initial guesses for this fit are made based on some prior expectations about
    what a typical solar-like power spectrum looks like, i.e., 2-3 harvey-like
    noise terms at progressively lower and lower frequency, and with higher and
    higher power. This fit incorporates a gaussian to account for the power excess
    of the p-mode envelope, and a constant for the white noise level.
    The n_harvey option needs to updated to be a bit smarter.

    A short initial fit is performed to ensure that the background is reasonably
    well accounted for, before the mode detection and identification is attempted.
    """
    f, p = fit_dict['spectrum']['freq'] ,fit_dict['spectrum']['power']

    dicts = [fit_dict['fit'][i] for i in sorted(fit_dict['parameter_keys']['bkg_keys'])] + [fit_dict['extra_params'][i] for i in sorted(fit_dict['parameter_keys']['extra_keys'])]

    # The spectrum is binned for speed in the fitting process, since high
    # resolution is not required.
    F, P = bin_the_spectrum(f, p, settings.bins)

    #==============================================================================
    # Initial positions for tuning fit
    #==============================================================================
    # Indices of the frequency axis around numax, used later for computing the
    # amplutide of the p-mode envelope
    idx_p = between(F,
                    fit_dict['star']['nu_max']-fit_dict['star']['nu_max']/4.0,
                    fit_dict['star']['nu_max']+fit_dict['star']['nu_max']/4.0)

    # Indices for the low frequency part of the spectrum, used to compute the
    # first Harvey component
    idx_h1 = between(F,
                     0.,
                     fit_dict['star']['nu_max'] * settings.nu_gran_0 / settings.nu_max_0)

    # Function to minimize in order to find the first Harvey components, calls
    # 'flops.harvey' which is specified in flops.f90
    bkg_objective = lambda D, x, y, s: -lnlike(y, flops.harvey(x, D[0], D[1], D[2])+D[3], int(s))

    # For each of the conditions n_harvey 1,2 or 3, the initial guesses are determined.
    # This can probably be done in a smarter way. If n_harvey = 1, it is assumed
    # that this refers to the granulation noise and that the spectrum is truncated
    # at some suitable frequency. This will not incorporate any contribution from
    # the very low frequency noise from e.g. instrumental effects or activity.
    if settings.n_harvey == 1:
        fit_dict['fit']['harvey_exponents']['init_guess']    = np.array([2.0])
        fit_dict['fit']['harvey_frequencies']['init_guess']  = np.array([fit_dict['star']['nu_max'] * settings.nu_gran_0 / settings.nu_max_0])
        fit_dict['fit']['harvey_powers']['init_guess']       = np.array([fit_dict['star']['nu_max'] * settings.nu_gran_0 / settings.nu_max_0 * np.mean(P[idx_h1][-5:])])
    # n_harvey = 2 assumes that the noise level arises from the very low frequency noise and the granulation
    if settings.n_harvey == 2:
        out = optimize.minimize(bkg_objective, [2.0, F[idx_h1][0], P[idx_h1][0], np.mean(P[idx_h1][-50:])], method = 'Nelder-Mead', args = (F[idx_h1], P[idx_h1], settings.bins))
        fit_dict['fit']['harvey_exponents']['init_guess']    = np.array([out.x[0], 2.0])
        fit_dict['fit']['harvey_frequencies']['init_guess']  = np.array([out.x[1], fit_dict['star']['nu_max'] * settings.nu_gran_0 / settings.nu_max_0])
        fit_dict['fit']['harvey_powers']['init_guess']       = np.array([out.x[2], fit_dict['star']['nu_max'] * settings.nu_gran_0 / settings.nu_max_0 * np.mean(P[idx_h1][-5:])])
    # n_harvey = 3 assumes that the noise level arises from the very low frequency noise, the granulation, as well as an additional variation between the granulation bump and p-mode envelope
    if settings.n_harvey == 3:
        out = optimize.minimize(bkg_objective, [2.0, F[idx_h1][0], P[idx_h1][0], np.mean(P[idx_h1][-50:])], method = 'Nelder-Mead', args = (F[idx_h1], P[idx_h1], settings.bins))
        fit_dict['fit']['harvey_exponents']['init_guess']    = np.array([out.x[0], 4.0, 4.0])
        fit_dict['fit']['harvey_frequencies']['init_guess']  = np.array([out.x[1], fit_dict['star']['nu_max'] * settings.nu_gran_0 / settings.nu_max_0, fit_dict['star']['nu_max'] / 2.0 * (1.0 + settings.nu_gran_0 / settings.nu_max_0)])
        fit_dict['fit']['harvey_powers']['init_guess']       = np.array([out.x[2], fit_dict['star']['nu_max'] * settings.nu_gran_0 / settings.nu_max_0 * np.mean(P[idx_h1][-5:]), fit_dict['star']['nu_max'] * settings.nu_gran_0 / settings.nu_max_0 * np.mean(P[idx_h1][-5:])])

    # The white noise level is estimated by the mean of the last few hundred indices of the power spectrum, presuming they are sorted in frequency.
    fit_dict['fit']['white_noise']['init_guess']               = np.array([noiselevel(P, verbose = True)])

    # The 'extra_params' keyword here refers to the parameters describing the
    # gaussian p-mode envelope. These are not included in the peakbagging fit
    # later on, and so are not grouped with the other parameters with the 'fit'
    # keyword.
    fit_dict['extra_params']['pmode_env_freq']['init_guess']   = np.array([fit_dict['star']['nu_max']])
    fit_dict['extra_params']['pmode_env_width']['init_guess']  = np.array([fit_dict['star']['nu_max']/(4.0*np.sqrt(2.0*np.log(2.0)))])
    fit_dict['extra_params']['pmode_env_power']['init_guess']  = np.array([max(P[idx_p])/4.0])

    # The lower limits of the parameters in the background fit are set here.
    # Values that in principle can have a lower limit of 0 are set to 2e-19 (arbitrarily chosen)
    fit_dict['fit']['harvey_exponents']['llim']          = np.array([2e-19 for x in range(len(fit_dict['fit']['harvey_exponents']['init_guess']))])
    fit_dict['fit']['harvey_frequencies']['llim']        = np.array([2e-19 for x in range(len(fit_dict['fit']['harvey_frequencies']['init_guess']))])
    fit_dict['fit']['harvey_powers']['llim']             = np.array([2e-19 for x in range(len(fit_dict['fit']['harvey_powers']['init_guess']))])
    fit_dict['fit']['white_noise']['llim']               = np.array([2e-19])
    fit_dict['extra_params']['pmode_env_freq']['llim']   = np.array([100])
    fit_dict['extra_params']['pmode_env_width']['llim']  = np.array([10])
    fit_dict['extra_params']['pmode_env_power']['llim']  = np.array([2e-19])

    # The upper limits of the parameters in the background fit are set here.
    fit_dict['fit']['harvey_exponents']['ulim']          = np.array([10 for x in range(len(fit_dict['fit']['harvey_exponents']['init_guess']))])
    fit_dict['fit']['harvey_frequencies']['ulim']        = np.array([F[-1] for x in range(len(fit_dict['fit']['harvey_frequencies']['init_guess']))])
    fit_dict['fit']['harvey_powers']['ulim']             = np.array([1e20 for x in range(len(fit_dict['fit']['harvey_powers']['init_guess']))])
    fit_dict['fit']['white_noise']['ulim']               = np.array([100])
    fit_dict['extra_params']['pmode_env_freq']['ulim']   = np.array([F[-1]])
    fit_dict['extra_params']['pmode_env_width']['ulim']  = np.array([1e3])
    fit_dict['extra_params']['pmode_env_power']['ulim']  = np.array([1e20])

    # The number of parameters in each keyword are counted here. This is simply
    # to ease calculations later.
    for d in dicts: d['npars'] = len(d['init_guess'])

    #==============================================================================
    # Tuning initial background parameters
    #==============================================================================
    # setup for the short initial MCMC fit to the background
    params, llim, ulim = [np.concatenate([d[k][0:] for d in dicts]).ravel() for k in ['init_guess', 'llim', 'ulim']]
    for j in range(len(params)):
        if params[j] < llim[j]: params[j] = llim[j]
        if params[j] > ulim[j]: params[j] = ulim[j]
    pos = np.array([[np.random.uniform(max(llim[i], params[i]*(1.0-settings.mcmc_sprd)),
                                       min(ulim[i], params[i]*(1.0+settings.mcmc_sprd))) for i in range(len(params))] for k in range(settings.mcmc_wlkrs)])

    # Sets up the EMCEE sampler for fitting the background to the binned spectrum.
    # The function bkg_lnprob is used for this.
    sampler = emcee.EnsembleSampler(settings.mcmc_wlkrs, len(llim), bkg_lnprob, args=(llim, ulim, F, P, settings), threads = settings.mcmc_thrds)

    # Number of steps for the EMCEE sampler to run
    r = 1000

    # Running the sampler
    sampler.run_mcmc(pos, r)

    # Fraction of the total run to discard, considered the burn-in phase
    burn = int(r*0.8)

    # Median values of the resulting MCMC chain
    m = np.median(sampler.chain[:, burn:,:].reshape((-1, len(llim))), axis = 0)

    # Assigning the median values to the 'init_guess' keywords in the fit_dict
    # dictionary. These are pointed to by the entries in the dicts list.
    for d in dicts:

        d['init_guess'] = m[:d['npars']]

        m = m[d['npars']:]

    # A model using the initial guesses for the background without the p-mode
    # envelope is computed and saved to the fit dictionary.
    fit_dict['spectrum']['model']['bkg'] = HGN(f,  np.concatenate([d['init_guess'][0:] for d in dicts]).ravel()[:-3], {'gaussian': False, 'noise': True, 'n_harvey': settings.n_harvey})

    # The above is repeated but with the p-mode envelope included. This is
    # intended for the diagnostics figures produced below.
    model = HGN(F, np.concatenate([d['init_guess'][0:] for d in dicts]).ravel(), {'gaussian': True, 'noise': True, 'n_harvey': settings.n_harvey})

    # Converting amplitude parameters in the fit to log scale.
    for key1 in ['harvey_powers', 'white_noise']:
        for key2 in ['init_guess', 'llim', 'ulim']:
            fit_dict['fit'][key1][key2] = np.log(fit_dict['fit'][key1][key2])

    #==============================================================================
    # Plotting
    #==============================================================================
    probfig = plt.figure(figsize = (15, 15))
    ax_prob = probfig.add_subplot(111)
    sampler.lnprobability[np.invert(np.isfinite(sampler.lnprobability))]=-1.e9
    ax_prob.loglog(-sampler.lnprobability.T)
    ax_prob.loglog(np.median(-sampler.lnprobability, axis = 0), color = 'r', lw = 6)
    ax_prob.axvline(burn, color = 'k', ls = 'dashed')
    ax_prob.set_ylabel('-log probability')
    ax_prob.set_xlabel('Step')
    ax_prob.set_ylim(0, max(-sampler.lnprobability.T[0, :]))
    ax_prob.set_title('Background fit')

    varfig = plt.figure(figsize = (15, 15))
    eo = len(llim)%2
    for k in range(len(llim)):
        ax_left = varfig.add_subplot(len(llim)/2+eo, 2, k+1)
        ax_left.plot(sampler.chain[:, :, k].T, '.', color = 'k', markersize = 0.5)
    varfig.axes[0].set_title('Background fit')

    bkgfitfig = plt.figure(figsize = (15, 15))
    bkgax = bkgfitfig.add_subplot(111)
    bkgax.set_xlabel('Frequency [$/mu$Hz]')
    bkgax.set_ylabel('Power [ppm$^2$/muHz]')
    bkgax.set_title('Initial parameter guesses')
    bkgax.set_xlim(F[0], F[-1])
    bkgax.loglog(F, P, 'g', lw = 1, alpha = 0.5)
    bkgax.loglog(F, model)

    return [probfig, varfig, bkgfitfig]


def bin_the_spectrum(x, y, n):
    """Function for binning the spectrum by a factor of n. This is used in the
    -autoguess module for for fitting the background. Such a fit does not require
    high frequency resolution so a binned spectrum is used for speed."""
    if n == 1:
        return x, y

    f, S = np.zeros_like(y), np.zeros_like(y)

    for i in range(0, len(y), n):

        # Averaging over bins of width n
        S[i] = 1./ (n) * np.sum(y[i-int(n/2):i+int(n/2)])

        f[i] = x[i]

    idx = S != 0.0

    return f[idx], S[idx]


def HGN(x, HGNarr,  args):
    """Function for the -autoguess module. Computes a spectrum with 1,2 or 3
    harvey laws, and with optional white noise level added. Adding a gaussian
    power excess to represent the p-mode envelope is also possible."""
    output = np.zeros_like(x)

    if args['n_harvey'] == 1:
        output += flops.harvey(x, HGNarr[0], HGNarr[1], HGNarr[2])
        if args['noise']: output += HGNarr[3]
        if args['gaussian']: output += flops.gaussian(x, HGNarr[5], HGNarr[4], HGNarr[6])
        return output
    if args['n_harvey'] == 2:
        output += flops.harvey(x, HGNarr[0], HGNarr[2], HGNarr[4])
        output += flops.harvey(x, HGNarr[1], HGNarr[3], HGNarr[5])
        if args['noise']: output += HGNarr[6]
        if args['gaussian']: output += flops.gaussian(x, HGNarr[8], HGNarr[7], HGNarr[9])
        return output
    if args['n_harvey'] == 3:
        output += flops.harvey(x, HGNarr[0], HGNarr[3], HGNarr[6])
        output += flops.harvey(x, HGNarr[1], HGNarr[4], HGNarr[7])
        output += flops.harvey(x, HGNarr[2], HGNarr[5], HGNarr[8])
        if args['noise']: output += HGNarr[9]
        if args['gaussian']: output += flops.gaussian(x, HGNarr[11], HGNarr[10], HGNarr[12])
        return output


def bkg_lnprob(params, llim, ulim, x, y, settings):
    """Returns the logarithmic probability, used specifically for the background
    fit in the -autoguess module. Computes the prior, which in this case is a
    simple ignorance prior which is 1 between provided limits of each parameter
    in the fit, and 0 otherwise. Returns -inf if invalid values are encountered."""
    lp = lnprior(params, llim, ulim)

    if np.isinf(lp) or np.isnan(lp): return -np.inf

    lnlikelihood = lnlike(HGN(x, params, {'gaussian':True,'noise':True, 'n_harvey': settings.n_harvey }), y, settings.bins)

    if np.isinf(lnlikelihood) or np.isnan(lnlikelihood): return -np.inf

    return lp + lnlikelihood


def between(arr, llim, ulim):
    """Returns a boolean array which is True between the provided limits."""
    c = (llim <= arr) * (arr < ulim) #np.invert(np.invert(arr >= llim)+np.invert(arr < ulim))

    return c


def noiselevel(y, verbose = False):
    """Takes the mean value of the last 100 frequencies in the power spectrum to
    represent the white noise level"""
    noise = np.mean(y[-100:])

    if verbose:
        print'Noise level estimate: %0.3f ' % round(noise,2)

    return noise

def plot_best_fit(fit_dict, settings, figsize = (15,15), fig_obj = None ):
    """Plots a model spectrum using the best_fit parameters (median values of the chains).
    The residual is also plotted and the modes are color coded."""
    f, p = fit_dict['spectrum']['freq'], fit_dict['spectrum']['power']

    psmth = smooth(p, window_len=0.1/np.median(np.diff(f)))

    bf_vals = np.array([])
    for key in fit_dict['parameter_keys']['mode_fit_keys']+fit_dict['parameter_keys']['bkg_keys']:
        bf_vals = np.append(bf_vals, fit_dict['fit'][key]['best_fit'])

    enns, ells, emms, freqs, heights, widths, incls, h_exponent, h_frequency, h_power, w_noise = unpack_parameters(bf_vals, fit_dict, settings)
    model = flops.spectral_model(f,
                                 ells,
                                 abs(emms),
                                 freqs,
                                 heights,
                                 widths,
                                 incls*np.pi/180.0,
                                 h_exponent,
                                 h_frequency,
                                 h_power,
                                 w_noise)


    modelfig = plt.figure(figsize = figsize)
    modax    = modelfig.add_subplot(111)

    modax.plot(f, psmth,'k', alpha = 0.5, label = 'Smoothed spectrum')
    modax.plot(f, smooth(p/model-2, window_len=0.1/np.median(np.diff(f))), color = 'k', label = 'Smoothed residual')
    modax.plot(f, model, 'r', lw = 2, label = 'Model')
    modax.set_ylabel(r'Power Spectral Density [ppm$^2$/$\mu$Hz]')
    modax.set_xlabel(r'Frequency [$\mu$Hz]')
    modax.legend()

    cols = ['b', 'g', 'r', 'y']
    for i, l in enumerate(fit_dict['fit']['ells']):
            modax.axvline(fit_dict['fit']['mode_freqs']['best_fit'][i], color = cols[int(l)])

    for i ,ifreq in enumerate(freqs):
        nrg = flops.energy(ells[i],np.abs(emms[i]),incls*np.pi/180.)
        modax.vlines(ifreq,0,heights[i]*nrg/widths[i],color ='red',lw=2,alpha=.5)

    modelfig.tight_layout()

    if isinstance(fig_obj,list) : fig_obj.append([modelfig,modax])


def plot_lnprobabilities(fit_dict, settings):
    """Plots the negative double-logarithmic probabilities (-log(log(p))) for all
    walkers and all steps in the MCMC chain. Convergence toward the global maximum
    is indicated by the median (thick solid line) tending toward a low constant value."""
    

    probfig = plt.figure(figsize = (15, 15))
    ax_prob = probfig.add_subplot(111) 
    probs   = fit_dict['fit']['lnprobabilities']
    print np.shape(fit_dict['fit']['steps']), np.shape(probs)
    ax_prob.loglog(fit_dict['fit']['steps'],-probs, color = 'k')
    ax_prob.loglog(fit_dict['fit']['steps'],-probs[:,2], color = 'r', lw = 6)
    ax_prob.set_ylabel('-log probability')
    ax_prob.set_xlabel('Steps')


def plot_percentiles(fit_dict, settings):
    """Creates a figure for each parameter key in the fit dictionary,  e.g.,
    mode_freqs or mode_splits. Each figure will have npars subplots, one for
    each parameter of that type. The chain percentiles will be plotted for all
    steps in the chain, and the last part of the chain which is saved is also
    shown. If a prior has been added the median and percentiles of the prior be
    shown as horizontal lines in the plot for the relevant parameter, as well as
    the prior limits."""

    steps = fit_dict['fit']['steps']

    for i, key in enumerate(fit_dict['parameter_keys']['mode_fit_keys']+fit_dict['parameter_keys']['bkg_keys']):

        npars = fit_dict['fit'][key]['npars']

        varfig = plt.figure(figsize = (15, 15))

        c, myiter = np.array([1, 1]), cycle(range(2))
        while c[0]*c[1] < npars:  c[next(myiter)] +=1

        yax, xax = range(1, c[1]*c[0]+c[1], c[1]), range(npars-c[1]+1,npars+1)

        for k in range(npars):

            P = fit_dict['fit'][key]['chain_percentiles'][:, :, k]

            maxstep = int(steps[-1])
            
            maxsave = len(fit_dict['fit'][key]['chain'][0, :, 0])
            
            ax_left = varfig.add_subplot(c[0], c[1], k+1)

            ax_left.plot(range(maxstep-maxsave, maxstep), fit_dict['fit'][key]['chain'][:, :, k].T, 'k.', ms = 1, alpha = 0.5)
	    
            ax_left.fill_between(steps, P[0], P[4], facecolor='blue', alpha=0.1)
            ax_left.fill_between(steps, P[1], P[3], facecolor='blue', alpha=0.1)
            ax_left.plot(steps,P[2])

            ax_left.set_xlim(0, steps[-1])
            ax_left.set_xticks(np.linspace(0, steps[-1], 3))

            if k+1 in yax: ax_left.set_ylabel(key)
            if k+1 in xax: ax_left.set_xlabel('Walker steps')


            prior = fit_dict['fit'][key]['prior'][k][0]

            if prior != None:

                x = prior[0]

                pofx = np.exp(prior[1])

                x_llim = x[0]

                x_ulim = x[-1]

                ax_left.axhline(x_llim, color = 'k', ls='dotted', lw = 3)

                ax_left.axhline(x_ulim, color = 'k', ls='dotted', lw = 3)

                x_perc = integ_up_to(x, pofx, [0.50-0.9845/2, 0.50-0.6827/2, 0.50, 0.50+0.6827/2, 0.50+0.9845/2])

                for i,perc in enumerate(x[x_perc]):
                    if i == 2:
                        ax_left.axhline(perc, color = 'r', lw = 3)
                    else:
                        ax_left.axhline(perc, color = 'r', lw = 3, ls='dashed')

        varfig.tight_layout()



def plot_unpacked_all(fit_dict,settings):

    #patch_legacy_fit_dict(fit_dict)
    fit_keys = fit_dict['parameter_keys']['mode_fit_keys']+\
               fit_dict['parameter_keys']['bkg_keys']
    ndim=len(fit_keys)
    varfig = plt.figure(figsize = (15, 15))

    c, myiter = np.array([1, 1]), cycle(range(2))
    while c[0]*c[1] < ndim:  c[next(myiter)] +=1

    #yax, xax = range(1, c[1]*c[0]+c[1], c[1]), range(ndim-c[1]+1,ndim+1)

    for k, key in enumerate(fit_keys):

        x_par,y_par = get_unpacked_param(fit_dict, fit_dict['settings'], key)
        ax = varfig.add_subplot(c[0], c[1], k+1)

        if len(x_par) != len(y_par):
            y_par=np.zeros(len(x_par)) + y_par[0]
        ax.plot(x_par,y_par,'.-')
        ax.set_xlabel(r'$\nu (\mu$Hz)')
        ax.set_ylabel(key)

    varfig.tight_layout()

def plot_unpacked_param(fit_dict, settings,par_key):

    x_par,y_par = get_unpacked_param(fit_dict, fit_dict['settings'], par_key)

    varfig = plt.figure(figsize = (15, 15))

    ax = varfig.add_subplot(1,1,1)

    ax.plot(x_par,y_par,'.-')

    ax.set_xlabel(r'$\nu (\mu$Hz)')

    ax.set_ylabel(par_key)

    varfig.tight_layout()

def get_unpacked_param(fit_dict,settings,par_key):

    packed_freqs = fit_dict['fit']['mode_freqs']['best_fit']
    freqs_out,_ = fit_dict['fit']['mode_freqs']['unpack_func'](packed_freqs, 0,fit_dict)
    nu_max = fit_dict['star']['nu_max']

    if (par_key == 'mode_freqs'):
        return freqs_out, freqs_out

    packed_vals = fit_dict['fit'][par_key]['best_fit']
    unpacked_par,_ = fit_dict['fit'][par_key]['unpack_func'](\
                packed_vals,0,fit_dict,freqs = freqs_out, nu_max = nu_max)

    return freqs_out, unpacked_par



def plot_eschelle(fit_dict, settings, enns = [], ells = [], freqs = []):
    """Plots the eschelle diagram using either the best_fit key from the fit
    dictionary or the frequencies from the *initial_guesses.txt. Only the former
    has proper errors associated with them. The frequencies are color coded by ell"""
    if len(freqs) == 0:
        f  = fit_dict['fit']['mode_freqs']['best_fit']
        fl = f - fit_dict['fit']['mode_freqs']['16th']
        fu = fit_dict['fit']['mode_freqs']['84th'] - f
        ells = fit_dict['fit']['ells']
        enns = fit_dict['fit']['enns']
    else:
        f  = freqs
        fl = np.zeros_like(f)
        fu = np.zeros_like(f)
        if len(ells) == 0: ells = np.zeros_like(f, dtype = int) - 1
        if len(enns) == 0: enns = np.zeros_like(f, dtype = int) - 1

    idxl0 = fit_dict['fit']['ells'] == 0

    fit_dict['star']['Del_nu'] = np.median(np.diff(fit_dict['fit']['mode_freqs']['init_guess'][idxl0]))

    dnu = fit_dict['star']['Del_nu']

    # Plot colors
    colors = ['k', 'b', 'g', 'r', 'm']

    ell_colors = np.zeros_like(ells, dtype = str)
    for i, n in enumerate(range(-1, max(ells.astype(int))+1)):
        ell_colors[ells == n] = colors[i%len(colors)]

    eschfig = plt.figure(figsize = (15, 15))
    eschax = eschfig.add_subplot(111)
    eschax.set_xlabel('Frequency modulo %.2f  [$\mu$Hz]' % dnu)
    eschax.set_ylabel(r'Frequency [$\mu$Hz]')

    for i in range(len(f)):
        plt.plot(f[i]%dnu, f[i], 'o', color = ell_colors[i], markersize = 10)
        plt.errorbar(f[i]%dnu, f[i], xerr = [[fl[i]],[fu[i]]],ls = 'dotted', color = ell_colors[i])
    eschfig.tight_layout()



def integ_up_to(f,p,R):
    """Iteratively computes the integral of p on an axis f until the area
    under the curve reaches a ratio R of the total integral I of p."""
    a, i, h, idx = 0, 0, np.array([]), np.zeros_like(f)

    I = np.trapz(p, f)

    for k, r in enumerate(R):

        while (a <= r) and (i-1 < len(f)):

            i += 1

            a = np.trapz(p[:i], f[:i]) / I

            h = np.append(h, a)

        idx[i-2] = 1

    return idx.astype(bool)


##def update_settings(fit_dict, settings):
##    """
##        update/fixes the settings in fit_dict, giving priority to whats written in fit_dict
##        the current settings object is not changed
##        only fit_dict['settings'] is changed
##        This routine should not affect the execution of PME, it is just to not to have
##        discordant informations in fit_dict
##    """
##
##    fit_dict['settings'] = settings
##    fit_dict['settings'].n_harvey = fit_dict['fit']['harvey_frequencies']['npars']


