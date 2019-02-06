# Peakbagging Made Easy#

Peakbagging Made Easy (PME) is a tool for fitting the oscillation power spectra of Sun-like stars. For a guide on how to use PME see the [User Manual](https://bitbucket.org/mbnielsen/peakbagging-made-easy/raw/ee5afdc3929df8a210cd390345775cef5ad770d9/user-guide-peakbagging.pdf). Refer to the manual for getting set up and the basic usage of PME, as well as some more advanced features. 

## Installation & Dependencies ##
[Download PME here](https://bitbucket.org/mbnielsen/peakbagging-made-easy/get/8d4fa4fc478f.zip)

* PME requires NumPy, Scipy and [EMCEE](http://dan.iel.fm/emcee/current/).
* A fortran compiler for the system you are using (Intel/AMD/others?)

PME is essentially just a big script, and requires little in the way of installation. However, before use the fortran file flops.f90 must be compiled using f2py on your system using your favorite fortran compiler (f2py is included in typical Numpy installations). Below is an example of how to compile flops.f90 using the intel fortran compiler.
```
f2py --f90exec=ifort --f90flags=-O3 -c flops.f90 -m flops
```
This creates a dynamic library flops.so which is what Python uses when calling some functions in the script. If edits to the fortran file are made, simply recompile the file. However, this should under normal circumstances not be necessary. 

## Who do I talk to? ##

* PME was original started by Martin Bo Nielsen (mbn4 at nyu.edu) with significant contributions from Emanuele Papini (papini at mps.mpg.de)