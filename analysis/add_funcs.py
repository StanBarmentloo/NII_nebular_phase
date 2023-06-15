import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.integrate as integrate
from astropy.io import ascii
from scipy.optimize import curve_fit
from tqdm import tqdm
from astropy import cosmology

c = 299792.458 #km/s
M_sun = 1.989*10**33 #g




def redshift_correct_wl(wl, distance = 0, own_velocity = 0, own_z = 0):
    
    H0 = 70 #km/s/Mpc, so should give distance in Mpc
    
    if distance != 0: 
        recession_velocity = H0*distance
        correction_factor = (1 + recession_velocity / c)
    elif own_velocity != 0:
        recession_velocity = own_velocity
        correction_factor = (1 + recession_velocity / c)
    elif own_z != 0: #If we already know the recession velocity, simply take that
        correction_factor = (1 + own_z)
    
    return wl/correction_factor

def v_from_labda(labda, labda0):
    
    ratio = labda/labda0
    return (ratio**2-1)*c/(ratio**2+1)

def z_to_r(z):
    H0 = 70 #km/s/Mpc
    return z*c/(H0) 

def doppler_factor(v):
    
    return np.sqrt( (1+v/c)/(1-v/c) )

def gaussian(x, cntr, amp, vel):
    
    return amp * np.exp(-0.5 * ((x-cntr)/vel)**2 )

def tophat(x, cntr, amp, vel):
    
    return amp * (  abs(x-cntr) < 1.5*vel ).astype(int) #1.5 is simply redefining the tophat

def double_gaussian(x, amp1, amp2, vel1, vel2):
    
    part1 = amp1 * np.exp(-0.5 * ((x-6316)/vel1)**2 ) #6316 is the expected peak due to 3:1 of 6300 and 6363
    part2 = amp2 * np.exp(-0.5 * ((x-6575)/vel2)**2 ) #6575 is the expected peak due to 1:3 of 6548 and 6583
    return part1 + part2

def gaussian_plus_tophat(x, amp1, amp2, vel1, vel2):
    
    part1 = amp1 * np.exp(-0.5 * ((x-6316)/vel1)**2 ) #6316 is the expected peak due to 3:1 of 6300 and 6363
    part2 = amp2 * (  abs(x-6575) < 1.5*vel2 ).astype(int)
    
    return part1+part2

def fit_ols(wl, flux, lower_bounds, upper_bounds):
    
    amp1_range = np.linspace(lower_bounds[0], upper_bounds[0], 20)
    amp2_range = np.linspace(lower_bounds[1], upper_bounds[1], 20)
    vel1_range = np.linspace(lower_bounds[2], upper_bounds[2], 20)
    vel2_range = np.linspace(lower_bounds[3], upper_bounds[3], 20)
    
    current_min, params, params2 = np.inf, [0, 0, 0, 0], [0, 0, 0, 0]
    for i in range(len(amp1_range)):
        for j in range(len(amp1_range)):
            for k in range(len(amp1_range)):
                for l in range(len(amp1_range)):
                    
                    ypred_dg = double_gaussian(wl, amp1_range[i], amp2_range[j], vel1_range[k], vel2_range[l])
                    lsq_dg = np.sum((ypred_dg-flux)**2)
                    
                    ypred_gth = gaussian_plus_tophat(wl, amp1_range[i], amp2_range[j], vel1_range[k], vel2_range[l])
                    lsq_gth = np.sum((ypred_gth-flux)**2)
                    
                    if lsq_dg < lsq_gth:
                        best_fit = 'dg'
                    elif lsq_dg > lsq_gth:
                        best_fit = 'gth'
                        
                    min_lsq = np.min((lsq_dg, lsq_gth))
                    
                    
                    if min_lsq < current_min:
                        params2 = params
                        params = [amp1_range[i], amp2_range[j], vel1_range[k], vel2_range[l], best_fit]
                        current_min = min_lsq
                        
                        #print('Gauss: ', lsq_dg, ' versus tophat: ', lsq_gth)
                        #print(min_lsq)
                    
    return params
    

def galaxy_remover(labda0, fwzi, wl, flux):
    
    galaxy_line_window = (wl < labda0+2*fwzi) * (wl > labda0-2*fwzi)
    continuum_line_window_left = (wl > labda0-4*fwzi) * (wl < labda0-2*fwzi)
    continuum_line_window_right = (wl < labda0+4*fwzi) * (wl > labda0+2*fwzi)
    
    lmax, rmax = np.max(flux[continuum_line_window_left]), np.max(flux[continuum_line_window_right])
    
    if np.max(flux[galaxy_line_window]) > 1.5*lmax and np.max(flux[galaxy_line_window]) > 1.5*rmax:
        
        lmean, rmean = np.mean(flux[continuum_line_window_left]), np.mean(flux[continuum_line_window_right])
        
        flux[galaxy_line_window] = np.array([(lmean + rmean)/2] * len(flux[galaxy_line_window]))
        
    return flux

def observed_flux_from_fit(wl, flux, plot = False):
    
    #Define the regions
    fitting_mask = (wl < 6800) * (wl > 6100)
    fitting_wl, fitting_flux = wl[fitting_mask], flux[fitting_mask]
    
    OI_dom_mask =  (fitting_wl < 6320) * (fitting_wl > 6100)
    NII_dom_mask =  (fitting_wl < 6680) * (fitting_wl > 6500)
    right_hand_side_mask = (fitting_wl < 6800) * (fitting_wl > 6500)
    
    #Remove the continuum
    l_min, r_min = np.min(fitting_flux[OI_dom_mask]), np.min(fitting_flux[right_hand_side_mask])
    continuum_flux = (fitting_wl-fitting_wl[np.where(r_min == fitting_flux)])*((r_min-l_min)/(fitting_wl[np.where(r_min == fitting_flux)]-fitting_wl[np.where(l_min == fitting_flux)])) + r_min
    
    if plot == True:
        plt.plot(fitting_wl, continuum_flux, linestyle = '--', c = 'r')
        plt.plot(fitting_wl, fitting_flux, c = 'black')
    
    fitting_flux = fitting_flux-continuum_flux
    
    fitting_flux = galaxy_remover(6563, 15, fitting_wl, fitting_flux)
    
    if plot == True:
        plt.plot(fitting_wl, fitting_flux, c = 'green')
        plt.show()
    
    #Set the fitting bounds
    OI_max, NII_max = np.max(fitting_flux[OI_dom_mask]), np.max(fitting_flux[NII_dom_mask])
    
    lower_bounds = np.array([0.7*OI_max, 0.7*NII_max, 30, 30])
    upper_bounds = np.array([1.2*OI_max, 1.2*NII_max, 120, 120])
    
    #Fit the double gaussian
    popt = fit_ols(fitting_wl, fitting_flux, lower_bounds, upper_bounds)
    
    #Plot the result
    if plot == True:
        
        if popt[-1] == 'dg':
            plt.plot(fitting_wl, double_gaussian(fitting_wl, *popt[:-1]))
            plt.plot(fitting_wl, gaussian(fitting_wl, 6575, popt[1], popt[3]), linestyle = '--')
        elif popt[-1] == 'gth':
            plt.plot(fitting_wl, gaussian_plus_tophat(fitting_wl, *popt[:-1]))
            plt.plot(fitting_wl, tophat(fitting_wl, 6575, popt[1], popt[3]), linestyle = '--')
            
        plt.plot(fitting_wl, gaussian(fitting_wl, 6316, popt[0], popt[2]), linestyle = '--')
    
        plt.plot(fitting_wl, fitting_flux)
        plt.show()
    
    #Get the integrated fluxes
    integrated_total_flux = integrate.cumtrapz(flux, wl)[-1]
    
    if popt[-1] == 'dg':
        integrated_NII_flux = integrate.cumtrapz(gaussian(wl, 6575, popt[1], popt[3]), wl)[-1]
    elif popt[-1] == 'gth':
        integrated_NII_flux = integrate.cumtrapz(tophat(wl, 6575, popt[1], popt[3]), wl)[-1]
    
    #print("For", project_names[0],  "the NII doublet carries ", '{0:.1f}'.format(integrated_NII_flux*100/integrated_total_flux),  "% (observed) of all flux.")
    
    return integrated_NII_flux, popt

def observed_flux_AJ(wl, flux, vline, labda0blue = 0, labda0red = 0, plot = False):
    
    blue_wl_bound = labda0blue * (1 - 1.25*vline/c)
    red_wl_bound = labda0red * (1 + 1.25*vline/c)
    
    blue_region_mask = (wl < labda0blue) * (wl > blue_wl_bound)
    red_region_mask = (wl > labda0red) * (wl < red_wl_bound)
    
    blue_region_min, red_region_min = np.min(flux[blue_region_mask]), np.min(flux[red_region_mask])
    blue_min_wl, red_min_wl = wl[ np.where(flux == blue_region_min)[0][0]], wl[ np.where(flux == red_region_min)[0][0]]
    
    slope = (red_region_min-blue_region_min)/(red_min_wl-blue_min_wl)
    continuum_flux = (wl - blue_min_wl)*slope + blue_region_min
    
    corrected_flux = flux-continuum_flux
    integration_bounds = (wl > (1 - vline/c)*labda0blue) * (wl < (1 + vline/c)*labda0red)
    
    #=======================================================================================================
    if plot == True:
        plt.plot(wl, flux, label = 'Original flux')
        plt.plot(wl, continuum_flux, label = 'Selected continuum flux', linestyle = '--', alpha = 0.7, lw = 0.8)
        plt.plot(wl, corrected_flux, label = 'Corrected flux')

        plt.axvline(x = 6300, linestyle = '--', c = 'black', label = 'OI')
        plt.axvline(x = 6363, linestyle = '--', c = 'black')
        
        plt.axvline(x = labda0blue*(1-1.25*vline/c), linestyle = '--', c = 'r', label = 'Continuum limits')
        plt.axvline(x = labda0red*(1+1.25*vline/c), linestyle = '--', c = 'r')
        plt.axvline(x = labda0blue*(1-vline/c), linestyle = '--', c = 'orange', label = 'Integration limits')
        plt.axvline(x = labda0red*(1+vline/c), linestyle = '--', c = 'orange')

        plt.axhline(y = 0, linestyle = '--', lw = 1, c = 'black')

        plt.xlim(6100, 6800)
        plt.xlabel('Wavelength [Å]')
        plt.ylabel('Flux [erg s-1 cm-2]')
        plt.legend()

        #plt.savefig('AJ_fitting_NII_2011dh.pdf')
        plt.show()
    #=======================================================================================================
    
    integrated_NII_flux = integrate.cumtrapz(corrected_flux[integration_bounds], wl[integration_bounds])[-1]
    
    return integrated_NII_flux

def Lnorm_AJ(t, M_Ni):
    
    return 1.06*10**42 * (M_Ni/(0.075*M_sun)) * (np.exp(-t/111.4) - np.exp(-t/8.8)) #erg s-1