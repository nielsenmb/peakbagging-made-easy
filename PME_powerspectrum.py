import multiprocessing as mp
import numpy as np
import flops, errno, sys, os
from PME_FUNCTIONS import get_me_this_file
from matplotlib.pyplot import *

def for_fortran(task, f, out, d, t, w, res):
    """The job sent to each worker. The indices st and sl for the frequency range
    is taken by the worker, and the powerspectrum is then computed in this range
    by using either the ls_wo_w (without weights) or the ls_w_w (with weights)
    functions from ls.f90."""
    st, sl = task.get()

    freq = f[st:sl]

    if len(w) == 0: out[st:sl,0],out[st:sl,1],out[st:sl,2] = flops.ls_wo_w(t, d, freq)

    if len(w)==len(d): out[st:sl,0],out[st:sl,1],out[st:sl,2] = flops.ls_w_w(t, d, w, freq)

    res.put(out)

def setup_workers(f, d, t, w, out, n):
    """Initializes the workers and divides the frequency range into chunks that
    that are then handed to each worker. The result res is then joined when
    each worker reports it is finished."""

    if n==1 :
        if len(w) == 0: out[:,0],out[:,1],out[:,2] = flops.ls_wo_w(t, d, f)

        if len(w)==len(d): out[:,0],out[:,1],out[:,2] = flops.ls_w_w(t, d, w, f)

	return out

    res, task =mp.Queue(), mp.Queue()

    if n> mp.cpu_count(): n=mp.cpu_count()

    L = int(len(f)/n)

    workers = [mp.Process(target=for_fortran, args=(task, f, out, d, t, w, res)) for i in xrange(n)]

    for each in workers: each.start()

    for i in xrange(len(workers)):
        if i == len(workers)-1:
            task.put([i*L,len(f)-1])
        else:
            task.put([i*L,(i+1)*L])

    task.close()
    task.join_thread()

    i = 0
    while n:
        try:
            out += res.get()
        except IOError, e:
            if e.errno == errno.EINTR:
                continue
            else:
                raise
        n -=1
        i+=1

    return out


def setup_frequency_range(Delta_T,delta_T,settings):

    cpd_to_muHz = 1e6/(60*60*24)

    fstart, fend, df = settings.PS_fstart, settings.PS_fend, settings.PS_fres

    if df is None:
        df =  1./Delta_T * cpd_to_muHz
    elif df < 1./Delta_T * cpd_to_muHz:
        print """WARNING! input frequency resolution df < 1/T. Oversampling effects will likely appear in the output LS periodogram! This means that the PDF of the power is no longer chi^2 with 2 d.o.f., and your fit may be wrong!"""

    if fstart is None:
        fstart = 1./Delta_T * cpd_to_muHz

    if fend is None:
        fend = 1./(2*delta_T) * cpd_to_muHz

    if fend < fstart:
        print "WARNING! fstart > fend is a BAD INPUT!"
        print "Try again!"
        sys.exit()

    f = np.arange(fstart,fend,df) #fstart,fend and df should now be in muHz

    fc = f / cpd_to_muHz

    return f, fc, fstart, fend, df




def LombScargle(settings):
    """The input time array is supposed to be in days (KEPLER LEGACY).
         ALL OPTIONAL INPUT FREQUENCIES MUST BE IN MICROHz"""

    # factor for converting cycles per day to muHz
    cpd_to_muHz = 1e6/(60*60*24)

    # Find the input file
    fname = get_me_this_file(settings, name = settings.input, ext = '.dat')

    # Load the input file
    input_TS = np.genfromtxt(fname)

    # Transposing TS in case its in row format
    if min(np.shape(input_TS)) != np.shape(input_TS)[1]:
        input_TS = input_TS.T

    # The requested columns are pulled out of the input array. It is assumed
    # that col 0 is timestamps, 1 is variable, 2 is errors on variable
    cols = np.array(settings.TS_cols[0].split(',')).astype(int)


    if (len(cols) == 2) and settings.PS_weighted: # 1.
        print "Using 3rd column in input file as weights"
        cols = np.append(cols,2)
        T,D,W = input_TS[:,cols[0]],input_TS[:,cols[1]],input_TS[:,cols[2]]
    elif (len(cols) == 3): #3. and 4.
        settings.PS_weighted = True
        T,D,W = input_TS[:,cols[0]],input_TS[:,cols[1]],input_TS[:,cols[2]]

    elif (len(cols) < 2) or (len(cols) > 3): # 5. and 6.
        print "I don't understand the input format of the time series. Must be either Nx2 for unweighted spectra or Nx3 for weighted spectra"
        print "Exiting..."
        sys.exit()
    else: # 2.
        T,D,W = input_TS[:,cols[0]],input_TS[:,cols[1]],np.array([])


    # Removing potentially hazardous time stamps (infs and nans)
    idx = np.invert(np.array(np.isinf(D) | np.isnan(D)))


    T, D = T[idx], D[idx]
    if len(W) > 0: W = W[idx]

    # Zeroing in the mean of D
    D -= np.mean(D)

    # Zeroing the start of the time series
    T -= T[0]
    DT_eff = T[-1] - T[0]

    # Computing frequency range for window function
    f, fc, fstart, fend, df = setup_frequency_range(DT_eff,np.median(np.diff(T)),settings)

    print "Computing spectral window function"
    cos_win = np.cos(2*np.pi*np.mean(fc)*T)

    sin_win = np.sin(2*np.pi*np.mean(fc)*T)

    outcos = setup_workers(2*np.pi*fc, cos_win, T, W, np.zeros([len(fc),3]), settings.PS_threads)  #Calculating the powerspectrum of pure cosine wave

    outsin = setup_workers(2*np.pi*fc, sin_win, T, W, np.zeros([len(fc),3]), settings.PS_threads)  #Calculating the powerspectrum of pure sine wave

    pwin = 0.5*(outcos[:,2]+outsin[:,2])


    if settings.PS_weighted and (settings.PS_fres == None):
        print "Adjusting effective time series length to account for weights"
        DT_eff = 1./ np.trapz(pwin, fc)

        f, fc, fstart, fend, df = setup_frequency_range(DT_eff,np.median(np.diff(T)),settings)


    print
    print 'Total time series length: ',T[-1],' days'
    print 'Frequency range %.2f muHz to %.2f muHz:' % (fstart,fend)
    print 'Resolution  [muHz]: ',df
    print 'Number of test frequencies: ',len(f)
    print 'Number of multiprocessing threads:', settings.PS_threads
    print
    print 'Computing power spectrum'

    out = setup_workers(2*np.pi*fc, D, T, W, np.zeros([len(f),3]), settings.PS_threads)

    # power spectrum
    p = out[:,2]

    # Normalizing by effective obs. time and converting to muHz
    psd = DT_eff / 2 * p / cpd_to_muHz #power spectral density

    # Setting up output file name
    if '*_output.npz' not in settings.output:
        if settings.output == os.path.basename(settings.output):
            fname = os.path.join(settings.directory,settings.output)
        else:
            fname = settings.output

        if not fname.endswith('.pow'):
            print "WARNING! The requested output file name does not have the *.pow extension. You will need to manually provide PME with this file name if you plan to use it for -autoguess and -peakbagging etc."

    else:
        fname = os.path.join(settings.directory,settings.star+'.pow')

    np.savetxt(fname,np.column_stack([f[:-1], psd[:-1]]))
    print
    print "Power density spectrum saved as %s" % (os.path.basename(fname))