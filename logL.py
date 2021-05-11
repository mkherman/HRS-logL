"""
Author: Miranda Herman
Created: 2020-10-28
Last Modified: 2021-05-11
Description: Calculates the 6-D log likelihood map for a series of atmospheric
models cross-correlated with planetary emission spectra. Parameters are log VMR, 
day-night contrast, peak phase offset, scaled line contrast, orbital velocity, 
and systemic velocity. 
NOTE: Because this computes the full likelihood map, not MCMC chains, this file 
is very computationally expensive to run when the full parameter grid is used, 
and the output can be multiple Gigabytes. Either run the file on a server that 
can handle this or reduce the ranges and/or stepsizes for the parameter arrays.
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import glob
import os
from astropy.convolution import convolve, Gaussian1DKernel
import argparse
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt


def planck(wavelength,temp):
	""" 
	Calculates the Planck function for a given temperature over a
	given wavelength range.
	"""
	c1 = 1.1911e-12
	c2 = 1.439
	y = 1e4/wavelength
	a = c1*(y**5.)
	tmp =  c2*y/temp
	b = np.exp(tmp) - 1.
	bbsor = a/b
	return bbsor


def remove_env(wave, spec, px):
	"""
	Subtracts the lower envelope from a model spectrum by finding 
	the minimum value in the given stepsize, then interpolating.
	"""
	low_wave, low_spec = [], []
	for i in range(len(spec)/px - 1):
		idx = np.nanargmin(spec[i*px:(i+1)*px])
		low_spec.append(spec[idx+i*px])
		low_wave.append(wave[idx+i*px])
	interp = interp1d(low_wave, low_spec, fill_value='extrapolate')
	envelope = interp(wave)
	corrected = spec - envelope
	return corrected


def wavegrid(wavemin,wavemax,res):
	"""
	Creates a wavelength array evenly spaced in resolution elements.
	"""
	c=299792458.
	dx=np.log(1.+1./res)
	x=np.arange(np.log(wavemin),np.log(wavemax),dx)
	wavelength=np.exp(x)
	#waveno=1e4/wavelength
	return wavelength #,waveno


def correlate(wave,spec,stdev,vgrid,minwave,maxwave,model_interp):
	"""
	Calculates the cross-correlation map for a given spectral order,
	along with the other two terms of the log likelihood equation:
	the spectra squared, and the base model squared.
	"""
	cmap = np.empty((len(spec),len(vgrid)))
	lnL_term1 = np.empty(len(spec))
	lnL_term2 = np.empty((len(spec),len(vgrid)))

	# Isolate wavelength range and scale data 
	w_idx = (wave[0,:] >= minwave) & (wave[0,:] <= maxwave)

	for frame in range(len(spec)):
		fixspec = spec[frame,w_idx] - np.nanmean(spec[frame,w_idx])
		fixspec /= stdev[frame,w_idx]

		# Calculate data term for log likelihood
		lnL_term1[frame] = np.nansum(fixspec**2)

		for i, vel in enumerate(vgrid):
			# Shift model to desired velocity and scale
			redshift = 1. - vel / 3e5
			shift_wave = wave[0,w_idx] * redshift
			mspec_shifted = model_interp(shift_wave)
			mspec_weighted = mspec_shifted - np.nanmean(mspec_shifted)
			mspec_weighted /= stdev[frame,w_idx]

			# Perform cross-correlation
			corr_top = np.nansum(mspec_weighted * fixspec)
			#corr_bot = np.sqrt(np.nansum(mspec_weighted**2) * np.nansum(fixspec**2))
			cmap[frame,i] = corr_top #/ corr_bot

			# Calculate model term for log likelihood
			lnL_term2[frame,i] = np.nansum(mspec_weighted**2)

	return cmap, lnL_term1, lnL_term2


def submed(cmap):
	"""
	Subtracts the median along the velocity axis from the 
	cross-correlation map.
	"""
	mdn = np.nanmedian(cmap,axis=1)
	sub = cmap - mdn[:,np.newaxis]
	return sub


def phasefold(Kps, vgrid, vsys, cmap, phase):
	"""
	Shifts the cross-correlation map to planet's rest frame and 
	creates the Kp-Vsys map.
	"""
	fmap = np.empty((len(Kps), len(vsys)))
	KTVmap = np.zeros((len(Kps), len(cmap), len(vsys)))

	for i, Kp in enumerate(Kps):
		fullmap = np.empty((len(cmap),len(vsys)))
		for frame in range(len(phase)):

			# Shift to planet's orbital velocity
			vp = Kp * np.sin(2.*np.pi*phase[frame])
			vshift = vgrid - vp

			shift = interp1d(vshift, cmap[frame,:], bounds_error=False)
			shifted_map = shift(vsys)
			fullmap[frame,:] = shifted_map

		KTVmap[i] = fullmap
		fmap[i,:] = np.nansum(fullmap, axis=0)
	return fmap, KTVmap


def chi2(cmap, merr, serr, alpha, Kps, vgrid, vsys, phase):
	"""
	Calculates the log likelihood from the previously computed 
	cross-correlation map and other base terms, for a given set
	of scaled line contrast values.
	"""
	X2 = np.zeros((len(alpha), len(Kps), len(vsys)))	# (alpha, Kps, Vsys)

	# Shift merr and cmap to the planet's velocity, so their axes are (Kp, time, Vsys)
	_, term2_shift = phasefold(Kps, vgrid, vsys, merr, phase)
	_, term3_shift = phasefold(Kps, vgrid, vsys, cmap, phase)

	# Calculate the log likelihood for each value of alpha	
	for i,a in enumerate(alpha):
		X2_KTV = serr[np.newaxis,:,np.newaxis] + a**2 * term2_shift - 2 * a * term3_shift
		
		# Sum the log likelihood in time
		X2[i] = np.nansum(X2_KTV, axis=1)
	return X2


def contrast_offset(phase, offset_deg, contrast):
	"""
	Computes the brightness variation for a given set of day-night 
	contrast and peak phase offset values over a given phase range.
	"""
	offset = offset_deg / 360.
	# Equation: A_p = C * sin^2(pi*(phi+theta)) + (1 - C)
	A_p = 1. - 	contrast[:,np.newaxis,np.newaxis] * \
			np.cos(np.pi*(phase[np.newaxis,np.newaxis,:] - \
					offset[np.newaxis,:,np.newaxis]))**2
	return A_p


def butterworth(x, order, freq, filt_type='highpass'):
	"""
	Applies a high-pass Butterworth filter, with a given order and 
	cut-off frequency, to the given model.
	"""
	butterfilt = butter(order, freq, btype=filt_type, output='sos')
	x_filtered = sosfiltfilt(butterfilt, x)
	return x_filtered


###############################################################################


parser = argparse.ArgumentParser(description="Likelihood Mapping of High-resolution Spectra")
parser.add_argument("-nights", nargs="*", help="MJD nights", type=str)
parser.add_argument("-d", '--datapath', default="./examples/", help="path to data")
parser.add_argument("-m", '--modelpath', default="./examples/", help="path to models")
parser.add_argument("-o", '--outpath', default="./examples/", help="path for output")
parser.add_argument("-ext", '--extension', default=".fits", help="output file name extension")
args = parser.parse_args()

nights = args.nights
data_path = args.datapath
model_path = args.modelpath
out_path = args.outpath
ext = args.extension

# Define parameter arrays
vmrs = np.arange(-5., -2.1, 0.1)
alpha = np.arange(0.5, 5., 0.1)
vgrid = np.arange(-600.,601.5, 1.5)
Vsys = np.arange(-150., 150., 0.5)
Kps = np.arange(175.,275., 0.5)
offset = np.arange(-30.,60., 1.)
contrast = np.arange(0.,1.1, 0.1)
lnL = np.zeros((len(vmrs),len(contrast), len(offset), len(alpha), len(Kps), len(Vsys)))

# Specify number of SYSREM iterations used on spectra for each MJD night
iters = {'56550': 5, '56561': 4, '56904': 4, '56915': 6, '56966': 6}

# Specify Butterworth filter cut-off frequency for each night
bfreq = {'56550': 0.035, '56561': 0.04, '56904': 0.03, '56915': 0.025, '56966': 0.055}


for night in nights:
	for v,vmr in enumerate(vmrs):

		# Read in data
		#spec = np.load(data_path+night+'_spectra.npy')[iters[night]-1] - 1.
		#wave = np.load(data_path+night+'_wavelength.npy')
		#phase = np.load(data_path+night+'_phase.npy')
		spec = np.load(data_path+'ESPaDOnS_spectra_whitenoise_injection_vmr4_a2_C0.99_o28.7_Kp221.9_Vsys-6.5.npy') - 1.
		wave = np.load(data_path+'ESPaDOnS_wavelength.npy')
		phase = np.load(data_path+'ESPaDOnS_phase.npy')
		
		# Only include phases below 0.41 and above 0.59, to avoid stellar Fe signal
		p_ind = np.where((phase < 0.41) & (phase > -0.41))[0]
		phase = phase[p_ind]
		spec = spec[:,p_ind,:]
		wave = wave[:,p_ind,:]
		
		# Determine size of arrays
		n_orders = spec.shape[0]
		n_frames = spec.shape[1]
		n_pix = spec.shape[2]
		
		# Get dayside model
		hdu = fits.open(model_path+'model_wasp33b_FeI_logvmr%.1f.fits' % (vmr))
		model = hdu[0].data
		
		# Interpolate model to wavelength grid with consistent resolution
		m_wave = wavegrid(model[0,0], model[0,-1], 3e5)
		wv_interp = interp1d(model[0],model[1], kind='linear', fill_value=0, bounds_error=False)
		m_spec = wv_interp(m_wave)

		# Convolve model with 1D Gaussian kernel, then filter
		FWHM_inst = {'CFHT': 4.48, 'Subaru': 1.8}
		mspec_conv = convolve(m_spec, Gaussian1DKernel(stddev=FWHM_inst['CFHT']/2.35))
		#mspec_day = remove_env(m_wave,mspec_conv, 250) 
		mspec_bf = butterworth(mspec_conv, 1, bfreq[night])
		
		# Create interpolator to put model onto data's wavelength grid
		filt_interp = interp1d(m_wave, mspec_bf, kind='linear', fill_value=0.,bounds_error=False)

		# Create variables/arrays for lnL components
		N = 0.
		cmap_osum = np.zeros((n_frames, len(vgrid)))
		merr_osum = np.zeros((n_frames, len(vgrid)))
		serr_osum = np.zeros((n_frames))
		
		# Compute brightness variation for given contrasts and offsets
		variation = contrast_offset(phase, offset, contrast)

		# Perform cross-correlation for orders redward of 600 nm, and sum together
		for i,o in enumerate(np.arange(24,37)): 
			# Calculate time- and wavelength-dependent uncertainties
			tsigma = np.nanstd(spec[o], axis=0)
			wsigma = np.nanstd(spec[o], axis=1)
			sigma = np.outer(wsigma, tsigma)
			sigma /= np.nanstd(spec[o,:,:])
			sigma[((sigma < 0.0005) | np.isnan(sigma))] = 1e20

			# Calculate number of data points in spectra
			minwave, maxwave = np.nanmin(wave[o,:,:]), np.nanmax(wave[o,:,:])
			minwidx, maxwidx = np.nanargmin(wave[o,0,:]), np.nanargmax(wave[o,0,:])
			N += len(wave[o,0,minwidx:maxwidx]) * len(phase)

			# Perform cross-correlation
			cmap0, serr, merr = correlate(wave[o,:,:], spec[o,:,:], sigma, vgrid, minwave, maxwave, filt_interp)
			cmap = submed(cmap0)
			
			cmap_osum +=  cmap
			merr_osum +=  merr
			serr_osum += serr

		# Apply brightness variation to lnL terms
		lnL_term1 = serr_osum
		lnL_term2 = merr_osum[np.newaxis,np.newaxis,:,:] * variation[:,:,:,np.newaxis]**2
		lnL_term3 = cmap_osum[np.newaxis,np.newaxis,:,:] * variation[:,:,:,np.newaxis]

		# Calculate lnL for given VMR
		for i in range(len(contrast)):
			for j in range(len(offset)):
				X2 = chi2(lnL_term3[i,j], lnL_term2[i,j], lnL_term1, alpha, Kps, vgrid, Vsys, phase)
				lnL[v,i,j] += -N/2. * np.log(X2 / N)


# Find highest likelihood values
maximum = np.nanmax(lnL)
maxes = np.where(lnL == maximum)
fidx = maxes[0][0]
cidx = maxes[1][0]
oidx = maxes[2][0]
aidx = maxes[3][0]
kidx = maxes[4][0]
vidx = maxes[5][0]

# Print highest likelihood values
print 'Location of highest likelihood:'
print 'logVMR = %.1f' % (vmrs[fidx])
print 'C = %.1f' % (contrast[cidx])
print 'off = %.1f' % (offset[oidx])
print 'a = %.1f' % (alpha[aidx])
print 'Kp = %.1f' % (Kps[kidx])
print 'Vsys = %.1f' % (Vsys[vidx])


# Write lnL to fits file
hdu2 = fits.PrimaryHDU(lnL)
hdu2.writeto(out_path+'lnL_wasp33b_FeI%s' % (ext), overwrite=True)
