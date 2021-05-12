"""
Author: Miranda Herman
Created: 2020-11-20
Last Modified: 2021-05-11
Description: Uses the log likelihood output of logL.py to calculate the 
conditional and marginalized likelihood distributions (mostly useful for 
creating plots). Also calculates the constrained value, uncertainty, and 
significance for each parameter. This calculation is separate from logL.py 
so that the likelihood maps from multiple datasets can be combined before 
computing the parameter constraints.
"""

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d


def significance(param,margL):
	"""
	Calculates the median, 1-sigma uncertainties, and significance
	for a given parameter based on its marginalized likelihood
	distribution.
	"""
	cdf = np.cumsum(margL)
	cdf_norm = cdf/cdf[-1]
	interp_cdf = interp1d(cdf_norm, param)
	low1sig, x_med, up1sig = interp_cdf(np.array([0.5-0.68*0.5, 0.5, 0.5+0.68*0.5]))
	uncert_low, uncert_up = x_med-low1sig, up1sig-x_med
	snr_low, snr_up = x_med/(x_med-low1sig), x_med/(up1sig-x_med)
	
	if snr_low > snr_up:
		return x_med, uncert_low, uncert_up, snr_up
	else:
		return x_med, uncert_low, uncert_up, snr_low


def lowerlim(param,margL):
	"""
	Calculates the 1-sigma lower limit for a bounded parameter
	based on its marginalized likelihood distribution.
	"""
	cdf = np.cumsum(margL)
	cdf_norm = cdf/cdf[-1]
	interp_cdf = interp1d(cdf_norm, param)
	lowlim = interp_cdf(np.array([0.68]))
	return lowlim


###############################################################################

# Open lnL file and compute likelihood
path = './'
lnL = fits.open(path+'lnL_wasp33b_FeI.fits')[0].data

maximum = np.nanmax(lnL)
L = np.exp(lnL - maximum)

# Find maxima 
maxes = np.where(lnL == maximum)
fidx = maxes[0][0]
cidx = maxes[1][0]
oidx = maxes[2][0]
aidx = maxes[3][0]
kidx = maxes[4][0]
vidx = maxes[5][0]

# Create conditional distribution maps
famap = lnL[:,cidx,oidx,:,kidx,vidx]
fcmap = lnL[:,:,oidx,aidx,kidx,vidx]
fomap = lnL[:,cidx,:,aidx,kidx,vidx]
fkmap = lnL[:,cidx,oidx,aidx,:,vidx]
fvmap = lnL[:,cidx,oidx,aidx,kidx,:]
comap = lnL[fidx,:,:,aidx,kidx,vidx]
acmap = lnL[fidx,:,oidx,:,kidx,vidx]
aomap = lnL[fidx,cidx,:,:,kidx,vidx]
akmap = lnL[fidx,cidx,oidx,:,:,vidx]
avmap = lnL[fidx,cidx,oidx,:,kidx,:]
kcmap = lnL[fidx,:,oidx,aidx,:,vidx]
komap = lnL[fidx,cidx,:,aidx,:,vidx]
kvmap = lnL[fidx,cidx,oidx,aidx,:,:]
vcmap = lnL[fidx,:,oidx,aidx,kidx,:]
vomap = lnL[fidx,cidx,:,aidx,kidx,:]

# Compute marginalized distributions
margf = np.nansum(L[:,cidx,oidx,aidx,:,vidx],axis=1)
margc = np.nansum(L[fidx,:,oidx,aidx,:,vidx],axis=1)
margo = np.nansum(L[fidx,cidx,:,aidx,:,vidx],axis=1)
marga = np.nansum(L[fidx,cidx,oidx,:,:,vidx],axis=1)
margk = np.nansum(L[fidx,cidx,oidx,aidx,:,:],axis=1)
margv = np.nansum(L[fidx,cidx,oidx,aidx,:,:],axis=0)

# Print constraints
print 'Format: median, lower uncert, upper uncert, significance'
print 'alpha: %.2f, -%.2f, +%.2f, %.2f' % (significance(alpha, marga))
print 'Kp: %.2f, -%.2f, +%.2f, %.2f' % (significance(Kps, margk))
print 'Vsys: %.2f, -%.2f, +%.2f, %.2f' % (significance(Vsys, margv))
print 'VMR:  %.2f, -%.2f, +%.2f, %.2f' % (significance(vmrs, margf))
print 'off:  %.2f, -%.2f, +%.2f, %.2f' % (significance(offset, margo))
#print 'C:  %.2f, -%.2f, +%.2f, %.2f, %.2f' % (significance(contrast, margc))
print 'C: > %.2f' % (lowerlim(contrast,margc))
