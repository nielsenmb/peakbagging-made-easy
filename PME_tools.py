

import PME_FUNCTIONS as PF

import pylab as plt
import numpy as np




from collections import OrderedDict

#**************************************************************************
#**********************************SOME TOOLS******************************
#**************************************************************************

class NumberedDict(OrderedDict):

    def __getitem__(self, index):

        if isinstance(index, int):
            return super(NumberedDict, self).__getitem__(self.keys()[index])
        else:
            return super(NumberedDict, self).__getitem__(index)


    def __setitem__(self, index,*args,**kwargs):

        if isinstance(index, int):
            return super(NumberedDict, self).__setitem__(self.keys()[index],*args,**kwargs)
        else:
            return super(NumberedDict, self).__setitem__(index,*args,**kwargs)


def find_nearest(array,value,index=False):

    if np.size(value) > 1 : return [find_nearest(array,i,index=index) for i in value]

    import math
    if np.size(array) > 1000:
        idx = np.searchsorted(array, value, side="left")
        if idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
            if index:
                return idx-1
            else:
                return array[idx-1]
        else:
            if index:
                return idx
            else:
                return array[idx]
    else:
        idx = (np.abs(array-value)).argmin()
        if (index == False) :
            return array[idx]
        else:
            return idx


def if_true(x,yn,out_if_false=None):
    """
    simple function for conditional input:
    INPUT
        x : any object

        yn: Boolean

    OPTIONAL INPUT :
        out_if_false = None :any object


    OUTPUT:
        if yn == True then it return x, 
        else it return out_if_false (default None)
    """
    if yn :
        return x
    else:
        return out_if_false





def get_palette(ncolors=255, start=0.,stop=1.):
    """
    ADAPTED FROM AN EXAMPLE I FOUND ON INTERNET
        import matplotlib.pyplot as plt
        
        from matplotlib import cm
        from numpy import linspace
        
        start = 0.0
        stop = 1.0
        number_of_lines= 1000
        cm_subsection = linspace(start, stop, number_of_lines) 
        
        colors = [ cm.jet(x) for x in cm_subsection ]
        
        for i, color in enumerate(colors):
            plt.axhline(i, color=color)
        
            plt.ylabel('Line Number')
            plt.show()
    """

    from matplotlib import cm

    cm_subsection = np.linspace(start, stop, ncolors)
    return [ cm.jet(x) for x in cm_subsection ]




def prob_dens_function( y, bins=10, yrange=None,fill_zeros = 1e-29 ,smooth_by = 0, **kwargs):
    """
    return the normalized distribution of y using numpy.histogram, but
    giving the central value of the bin instead of the edges.

    INPUT:
        y : array to compute the distribution of.
        
        **kwargs : all optional arguments of numpy.histogram.

    OPTIONAL INPUT:
        bins = 10 : number of bins (size) of the pdf to be returned.
    
        yrange = None : range of output bins (if None default is [min(y),max(y)].

        fill_zeros = 1e-29 : set zero values of pdf to the desired value.
    
        smooth_by = 0 : smooth the pdf with a gaussian filter. Allowed values are
                        0: no smooth
                        1: default smooth using a window lenght of 9
                        N>1: smooth using a window lenght of N 

    OUTPUT: (distr, bin_centers), tuple

        distr : array containing the normalized distribution f(x) of x.

        bin_centers : array of the independent variable x.

    """

    hist,bin_edges = np.histogram(y,bins=bins,range=yrange,**kwargs) 

    bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2.

    hist = hist.astype(np.float64)

    if fill_zeros is not None:
        hist[np.where(hist == 0)[0]] = fill_zeros



    if smooth_by == 1 :
        hist = PF.smooth(hist)

    if smooth_by > 1 :
        hist = PF.smooth(hist,window_len=smooth_by)

    hist /= np.trapz(hist,x= bin_centers)

    return hist, bin_centers



def make_prior_from_PME(fit_dict,key,bins=400, prange=None,fill_zeros = 1e-29 ,smooth_by = 5, save_to_file=True, return_prior = True):
    """
    Create a prior file containing the distribution of the priors for a parameter Px 
    as drawn by the chains from a previous fit.

    INPUT: 

        fit_dict : (dict) PME dictionary

        key : (str) key of the parameter to get the prior of.

    OPTIONAL INPUT
        bins = 10 : (int) number of bins (size) of the pdf of Px to be returned.
    
        prange = None : (None or list) range(s) of output bins (if None default is [min(Px),max(Px)].
                        If the fit parameter consists of more than one parameter (npar>1)
                        (e.g. if key='harvey_frequencies' and you have 2 frequencies,i.e., 2 params),
                        then prange is a list of shape (npar,2) :
        
                                    prange=[[minP1,maxP1],[minP2,maxP2],...]

        fill_zeros = 1e-29 : set zero values of pdf to the desired value.
    
        smooth_by = 0 : (int) smooth the pdf with a gaussian filter. Allowed values are
                        0: no smooth
                        1: default smooth using a window lenght of 9
                        N>1: smooth using a window lenght of N 


        save_to_file = True : (boolean)

    """






    #check if the key is present in the dictionary
    if key not in fit_dict['parameter_keys']['mode_fit_keys'] + fit_dict['parameter_keys']['bkg_keys']:

        print """ERROR! input key not found!, returning None."""
        return None


    fdc = fit_dict['fit'][key]

    npars = fdc['npars']

    chain = fdc['chain']

    chain = chain.reshape( np.prod(chain.shape[:-1]), chain.shape[-1] )

    priors = np.zeros ([bins,npars*2])

    if prange is None : 
        prange = []
        [prange.append(None) for i in range(npars)]

    if (npars == 1) & type(prange[0]) is not list: prange=[prange] 
    
    for i in range(npars):
        Px,x = prob_dens_function(chain[:,i],bins=bins,yrange=prange[i], \
                                  fill_zeros = fill_zeros ,smooth_by = smooth_by)
        priors[:,2*i] = x
        priors[:,2*i+1] = Px


    if save_to_file : np.savetxt(key+'.prior',priors)

    if return_prior : return priors








#**************************************************************************
#**********************************MAIN CLASS******************************
#**************************************************************************

class PME_LOAD(object):
    """
        NAME: PME (object)

        AUTHOR: Emanuele Papini

        AFFILIATION: MPI for Solar System Research and Astrophysik Institut Goettingen

        DATE CREATION: 10 February 2017

        This class provides auxiliary methods to manipulate the output
        results from the PME code developed by Martin Bo Nielsen.
        As such, it requires some functions provided by PME_FUNCTIONS.py
        
        This class has been extracted from process_star.py, which was created by my self 
        to provide a simplified way to load, process, and analyse kepler data of a star, 
        including interfaces to PME and adipls.
        

        CLASS INITIALIZATION:
            The class is initialized either by giving directly the PME output 
            dictionary (fit_dict) already loaded from , e.g., wc_output.npz 
            as an optional argument, e.g.

                import numpy as np
                import PME_tools
                
                fdict = np.load('wc_output.npz)['fit_dict'].item()
                pme_obj = PME_tools.PME_OBJ(fit_dict=dictionary)

            either by specifying the wild_card
            
                pme_obj = PME_tools.PME_LOAD(wild_card='wc')

            or by giving the filename

                pme_obj = PME_tools.PME_LOAD(files = 'wc_output.npz')
            

            N.B. The class can handle multiple dictionaries/.npz files at once
                 (though this may raise errors)

        INITIALIZATION DETAILS:

            PME_LOAD(fit_dict=None,wild_card=None, files=None)

            fit_dict: PME dictionary or list/tuple of PME dictionaries.
                      This argument OVERRIDES THE OTHERS.
            
            files: '*_output.npz' string or list of strings with file names (also full paths)
                   this argument OVERRIDES wild_card

            wild_card: PME wild card or list/tuple of wild cards.

                   
            N.B. in case of multiple files with multiple folders, the last approach 
                 (using files=...) is preferred.


        OUTPUT:
            The initialized object (pme_obj) contains a series of methods 
            (see each of them for usage details). pme_obj.fit_dict contains 
            all the loaded dictionaries.
        
                
            
    
    """

    def __init__(self,fit_dict=None, **kwargs):
        

        if fit_dict is not None:

            if isinstance(fit_dict,(list,tuple,NumberedDict)) :
                
                self.fit_dict = NumberedDict()
                for ifit in fit_dict: self.fit_dict[ ifit['star']['ID'] ] = ifit 
            
            else:
                
                self.fit_dict = NumberedDict()
                self.fit_dict[ fit_dict['star']['ID'] ]=fit_dict
        else:
            self.fit_dict = self._load_fit_dict_(**kwargs)

        self._plots_={}



    def _load_fit_dict_(self,wild_card=None, files=None):
        """
            load PME dictionary from file(s) or wild card(s).
            
            filename ovverrides wild_card.


        """
        if files is None and wild_card is None:
            raise NameError('Input not given to PME class!')
        
        fit_dict = NumberedDict()

        if files is not None:
            
            if type(files) is str: files = [files]
            
            
            for i,ifile in enumerate(files):
                fdummy =np.load(ifile)['fit_dict'].item() 
                fit_dict[ fdummy['star']['ID'] ]=fdummy
            
        else:

            if type(wild_card) is str:
                wild_card = [wild_card]

            for iwild in wild_card:
                fit_dict[iwild] = np.load(iwild+'_output.npz')['fit_dict'].item()
        
        return fit_dict


    def get_best_fit_model(self,wild_cards=None,**kwargs):
        """
            Get best fit model from all loaded dictionaries.
            If keys is specified, then it loads the best model only for the 
            selected wild_cards.

            **kwargs are defined in self._best_fit_model_

        
            OUTPUT:
            
            self.best_fit[wild_cards] : NumbDict of dictionaries. see 
            self._best_fit_model_ for details.
        
        
        """
        

        f_dict=self.fit_dict


        model = NumberedDict()
        if wild_cards is None :
            wild_cards = f_dict.keys()
        
        for key in wild_cards:

             
            model[key]=self._best_fit_model_(f_dict[key],get_unpacked = True,**kwargs)
            
   
        self.best_fit = model



    def _best_fit_model_(self,fit_dict,get_unpacked = True,with_errors = True):
        """
            Return a dictionary ['best_fit'] containing the best model 
            ['freq','fit_psd']=(frequency, PSD) calculated from fit_dict.
            
            if with_error = True 
                then returns a NumberedDict (ie list of dicts) 
                of ['best_fit','16th','84th'] with best model and models calculated from the
                16th and 84th percentiles of the parameters respectively.

            if get_unpacked = True 
                each dictionary, e.g., out['best_fit'] also has a keyword 'fit_params'  
                that point to a dictionary of all the unpacked parameters from PME_FUNCTIONS.

        """

        flops = PF.flops
        unpack_parameters = PF.unpack_parameters

        f, p = fit_dict['spectrum']['freq'], fit_dict['spectrum']['power']



        if with_errors:
            outkeys = ['best_fit','16th','84th']
        else:
            outkeys = ['best_fit']

        outdict= NumberedDict()

        

        for o_keys in outkeys:

            bf_vals = np.array([])
            for keys in fit_dict['parameter_keys']['mode_fit_keys']+ \
                        fit_dict['parameter_keys']['bkg_keys']:
                bf_vals = np.append(bf_vals, fit_dict['fit'][keys][o_keys])



            enns, ells, emms, freqs, heights, \
            widths, incls, h_exponent, h_frequency, \
            h_power, w_noise = \
            unpack_parameters(bf_vals, fit_dict, fit_dict['settings'])

            
            outp = NumberedDict()
            
            outp['freq'] = f
            outp['fit_psd'] =flops.spectral_model(f,\
                                 ells,\
                                 abs(emms),\
                                 freqs,\
                                 heights,\
                                 widths,\
                                 incls*np.pi/180.0,\
                                 h_exponent,\
                                 h_frequency,\
                                 h_power,\
                                 w_noise) 

            if get_unpacked :
                pars={'n':enns, 'l':ells, 'm':emms, 'freq':freqs, 'height':heights, \
                    'width':widths, 'incl':incls, 'h_exp':h_exponent, 'h_freq':h_frequency, \
                    'h_pow':h_power, 'w_noise':w_noise }
                outp['fit_params'] = pars

            outdict[o_keys]=outp


        return outdict


    def plot_best_fit(self,subplots = False , same = False, wildcard = 0, interactive = True,\
                      sharex = True, sharey=True, use_pme_functions = False):
        """
            Plot the best fit model(s) and the original spectrum

            subplots : boolean
                make 1 figure with n subplots, one for each wildcard best model
                OVERRIDES same

            same : boolean
                make 1 figure with 2 subplots: one with only the model for
                the selected wildcard (default = 0), the second with all the others.
                OVERRIDES use_pme_functions
            
            same : Boolean
                make a separate figure for each wild_card fit.

            use_pme_functions : Boolean
                uses PME_FUNCTIONS.plot_best_fit to make the plot
            
        
            figure and axis objects are saved in self._plots_ for further use
        """
        if 'plot_best_fit' not in self._plots_.keys() : 
            self._plots_['plot_best_fit']=type('',(),{})

        plots_ =self._plots_['plot_best_fit'] 
        lw=wildcard

        if lw == 0 : lw = self.fit_dict.keys()[0]

        if interactive : plt.ion()
        


        if subplots:

            from itertools import cycle

            varfig = plt.figure(figsize = (15, 15))

            ndim = len(self.fit_dict.keys())
            c, myiter = plt.array([1, 1]), cycle(range(2))
            while c[0]*c[1] < ndim:  c[next(myiter)] +=1
            yax, xax = range(1, c[1]*c[0]+c[1], c[1]), range(ndim-c[1]+1,ndim+1)


            #fig ,ax = plt.subplots(len(self.best_fit), sharex = sharex,sharey = sharey)
            plots_.fig = varfig
            ax = []
            plots_.ax = ax
            ccc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for j,i in enumerate(self.fit_dict.keys()):

                if j == 0:
                    ax.append(varfig.add_subplot(c[0], c[1], j+1))
                else:
                    ax.append(varfig.add_subplot(c[0], c[1], j+1,sharex=if_true(ax[0],sharex)\
                             ,sharey=if_true(ax[0],sharey) ))
                cc = ccc[j%len(ccc)]
                f, p = self.fit_dict[i]['spectrum']['freq'], self.fit_dict[i]['spectrum']['power']
                base_line = ax[j].plot(f,p,alpha=0.5,color=cc, label = i+' PSD')
                ax[j].plot( self.best_fit[i]['best_fit']['freq'],\
                            self.best_fit[i]['best_fit']['fit_psd'],color=cc ,label=i+' best fit')
            
                ax[j].legend()
                ax[j].set_xlabel(r'frequency / $\mu$Hz')
                ax[j].set_ylabel(r'PSD / (ppm$^2/\mu$Hz)')
            varfig.tight_layout()

        elif same is True:
            fig ,ax = plt.subplots(2, sharex = sharex,sharey = sharey)
            plots_.fig = fig
            plots_.ax = ax

            f, p = self.fit_dict[lw]['spectrum']['freq'], self.fit_dict[lw]['spectrum']['power']
            ax[0].plot(f,p,color='gray',alpha=0.5,label='PSD')

            ax[0].plot(self.best_fit[lw]['best_fit']['freq'],\
                       self.best_fit[lw]['best_fit']['fit_psd'],'red',label='best fit')

            ax[0].set_xlabel(r'frequency / $\mu$Hz')
            ax[0].set_ylabel(r'PSD / (ppm$^2/\mu$Hz)')
            ax[0].set_title("'"+lw+"'"+' spectrum')
            ax[0].legend()


            ccc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for j,i in enumerate(self.fit_dict.keys()):
                if i ==lw : continue

                cc = ccc[j]
                f, p = self.fit_dict[i]['spectrum']['freq'], self.fit_dict[i]['spectrum']['power']
                base_line = ax[1].plot(f,p,alpha=0.5,color=cc, label = i+' PSD')
                ax[1].plot( self.best_fit[i]['best_fit']['freq'],\
                            self.best_fit[i]['best_fit']['fit_psd'],color=cc ,label=i+' best fit')
            
            ax[1].legend()
            ax[1].set_xlabel(r'frequency / $\mu$Hz')
            ax[1].set_ylabel(r'PSD / (ppm$^2/\mu$Hz)')
            ax[1].set_title('other spectra')
            

        elif use_pme_functions :
            plots_.figs = []
            plots_.ax = []

            for j,i in enumerate(self.fit_dict.keys()):
                fig = []
                PF.plot_best_fit(self.fit_dict[i],self.fit_dict[i]['settings'],fig_obj = fig,figsize = None)
                plots_.figs.append(fig[0][0])
                plots_.ax.append(fig[0][1])
                plots_.ax[-1].set_title(i)

        else:
            plots_.figs = []
            for j,i in enumerate(self.fit_dict.keys()):
                
                fig = plt.figure()
                
                f, p = self.fit_dict[i]['spectrum']['freq'], self.fit_dict[i]['spectrum']['power']
                plt.plot(f,p,alpha=0.5,color='gray')
                plt.plot(self.best_fit[i]['best_fit']['freq'],self.best_fit[i]['best_fit']['fit_psd']\
                        ,color='red' ,label=i)
                plots_.figs.append(fig)


                
         
    def plot_param_vs_freq(self,param='height',with_errors = True, palette = 'rainbow',\
                           subplots=True):

        """
            Plot the desired unpacked fit parameter (previously stored in
            self.best_fit ,eg, best_fit[wild_card]['best_fit']['fit_params']) vs. mode frequency.

            subplots = True  : create a separate plot for each different l. 
                       False : plot all ells in the same plot       

            with_errors = True : plot also the errors as calculated from 16th and 84th
                                 percentiles:
                
                WARNING: the errors is not the 1-sigma error, but is the error 
                         as calculated using the 1-sigma uncertaintes on the fit parameters

            param='height' :parameter to plot (keyword of self.best_fit[wild_card]['']['fit_params'])


            fig = None : optional. if figure ob
        """

        if 'plot_param_vs_freq' not in self._plots_.keys() : 
            self._plots_['plot_param_vs_freq']=type('',(),{})

        plots_ = self._plots_['plot_param_vs_freq']

        bb = self.best_fit

        blmax = []#[int(np.max(bb[0]['best_fit'][2]['l']))]
        [blmax.append(int(np.max(bb[ib]['best_fit']['fit_params']['l']))) for ib in bb.keys()]
        lmax=np.max(blmax)
        if subplots :
            fig,ax = plt.subplots(1,lmax+1)
            ipl = np.zeros(lmax+1,dtype=int)
        else:
            fig = plt.figure(fig)
            ipl = range(lmax+1)
            ax = [plt]
            [ax.append(plt) for i in range(lmax)]


        plots_.fig = fig
        plots_.ax = ax

        #machinery for choosing the colors
        col_palette = plt.get_cmap(palette)
    
        ccc=get_palette( ncolors = len(bb) )

        ic = 0
        
        if len(bb[0].keys()) is 3:
            lmax = int(np.max(bb[0]['best_fit']['fit_params']['l']))
            
        for j,i in enumerate(bb.keys()):
            lmax = blmax[j]
            mask = []
                
            [mask.append( bb[i]['best_fit']['fit_params']['l'] == il) for il in range(lmax+1)]
     
                

            [ax[il].plot(bb[i]['best_fit']['fit_params']['freq'][mask[il]], \
                         bb[i]['best_fit']['fit_params'][param ][mask[il]],'o-',label=i+' l='+str(il),\
                         color=ccc[ic],linewidth=1+il) \
             for il in range(lmax+1)] 
             
            if with_errors:
                if len(bb[i].keys()) == 3:
                    [ax[il].fill_between (bb[i]['best_fit']['fit_params']['freq'][mask[il]], \
                                       bb[i]['16th']['fit_params'][param][mask[il]],\
                                       bb[i]['84th']['fit_params'][param][mask[il]],\
                                alpha=0.5,color=ccc[ic]) \
                     for il in range(lmax+1)] 
                else:
                    print 'No error values loaded for "'+i+'". Only plotting best_fit'
                    
            ic+=1

        
        plt.legend(loc='best') 
        plt.title(param)



    def extract_params (self,keys, using = 'freq'):
        """
        reorganize output unpacked params from different fits in a numpy array.
        This helps, eg, in plotting mode frequency as function of time etc.

        keys : keyword of the parameter to extract

        using: parameter to use for the extraction
        THIS HELP MUST BE IMPROVED :(
        """
        bb = self.best_fit
       
        

        to_fill = NumberedDict()
        for ikey in keys:
            to_fill[ikey] = [] 

        to_fill['m']=[]
        to_fill['l']=[]
        to_fill['n']=[]

        sizes = [len(bb[i]['best_fit']['fit_params']['n']) for i in range(len(bb))]

        nn = min(sizes)
        knn = np.where(np.asarray(sizes) == nn)[0][0]


        for imode,iuse in enumerate(bb[knn]['best_fit']['fit_params'][using]):

            imm = bb[knn]['best_fit']['fit_params']['m'][imode]

            matched_m = [ np.where(bb[i]['best_fit']['fit_params']['m'] == imm)[0] \
                          for i in range(len(bb)) ]
            
            common_idx = [ matched_m[i][find_nearest( \
                                        bb[i]['best_fit']['fit_params'][using][matched_m[i]] \
                                       ,iuse,index=True)] \
                           for i in range(len(bb)) ]
        
            [to_fill[ikey].append( [bb[i]['best_fit']['fit_params'][ikey][common_idx[i]] \
                                   for i in range(len(bb)) ]) \
                 for ikey in keys]

            to_fill['m'].append(int(bb[knn]['best_fit']['fit_params']['m'][imode]))
            to_fill['l'].append(int(bb[knn]['best_fit']['fit_params']['l'][imode]))
            to_fill['n'].append(int(bb[knn]['best_fit']['fit_params']['n'][imode]))

            
        for ikey in to_fill.keys():
            to_fill[ikey] = np.asarray(to_fill[ikey])
        return to_fill


    def extract_PME_params(self,key):
        """
        return the raw parameter from different fits of PME as a numpy array.
        This is useful when comparing different fit results, but be careful to 
        use the same parameterization (eg. sam width parameterization) for all fits.

        keys : keyword of the parameter to extract. Must be valid keys of 
               PME_dict['fit']

        RETURN:
            np.ndarray of shape(N_fits, 3, npars), where the second dimension identifies:
                0: 16th percentile
                1: best fit
                2: 84th percentile
            and the last dimension the number of parameters for that parameterization 
            (e.g if mode_splits is share all npars=1, if mode_split=acoeff_a2lin then npars=3)

        """
        bb = self.fit_dict

        to_fill=[]
        for i in bb.keys():

            to_fill.append([bb[i]['fit'][key]['best_fit'],\
                            bb[i]['fit'][key]['16th'],bb[i]['fit'][key]['84th']])

        return np.asarray(to_fill)

    def plot_PME_params(self,key):
        """
        COMMENT TO FILL
        """

        if 'plot_PME_params' not in self._plots_.keys() : 
            self._plots_['plot_PME_params']=type('',(),{})
      
        plots_ =self._plots_['plot_PME_params'] 


        to_plot = self.extract_PME_params(key)
       

        from itertools import cycle

        varfig = plt.figure(figsize = (15, 15))

        ndim = len( to_plot[0,0,:] )
        c, myiter = plt.array([1, 1]), cycle(range(2))
        while c[0]*c[1] < ndim:  c[next(myiter)] +=1
        yax, xax = range(1, c[1]*c[0]+c[1], c[1]), range(ndim-c[1]+1,ndim+1)


        #fig ,ax = plt.subplots(len(self.best_fit), sharex = sharex,sharey = sharey)
        plots_.fig = varfig
        ax = []
        plots_.ax = ax


        for j in range(ndim):
            
            ax.append(varfig.add_subplot(c[0], c[1], j+1))

            ax[j].plot(to_plot[:,0,j])

            ax[j].fill_between(range(len(to_plot[:,0,0])), to_plot[:,1,j].flatten(),to_plot[:,2,j].flatten(),alpha=0.5) 


    def PME_PLOT(self,wildcard = 0, interactive = True):
        """
            PLOT RESULTS OF SELECTED FIT (WILDCARD) using PME FUNCTIONS ONLY.
            The result is the same as if you execute:
                python PME.py ./ wildcard -plot
        """




        if interactive : plt.ion()


        fit_dict = self.fit_dict[wildcard]



        PF.plot_best_fit(fit_dict, fit_dict['settings'])
        
        try:
            PF.plot_lnprobabilities(fit_dict, fit_dict['settings'])
        except:
            print """WARNING: Something when wrong when trying to plot the likelihoods. This may mean that the fit is invalid."""
        
        PF.plot_percentiles(fit_dict, fit_dict['settings'])

        PF.plot_eschelle(fit_dict, fit_dict['settings'])




    def plot_surface_rotation(self, wildcard = 0, interactive = True):
        """
            Plot scatter plot of polar surface rotation against equator rotation
            using all the walkers from the last chain.

            So far only implemented if the parametrization function contains a_coefficients
        """
        

        fit_dict = self.fit_dict[wildcard]
        
        unpack_func = fit_dict['fit']['mode_splits']['unpack_func']
        
        if 'a_coeff' not in unpack_func.func_name:
            print "mode_splits parametrization incompatible with this function, returning None"
            return None


        






