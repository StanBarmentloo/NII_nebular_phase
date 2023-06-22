import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.integrate as integrate
from astropy.io import ascii
from scipy.optimize import curve_fit
from tqdm import tqdm
from astropy import cosmology

#Constants
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
    
    return amp * (  abs(x-cntr) < vel ).astype(int) #1.5 is simply redefining the tophat

def double_gaussian(x, mu1, mu2, amp1, amp2, vel1, vel2):
    
    part1 = amp1 * np.exp(-0.5 * ((x-mu1)/vel1)**2 ) #6316 is the expected peak due to 3:1 of 6300 and 6363
    part2 = amp2 * np.exp(-0.5 * ((x-mu2)/vel2)**2 ) #6575 is the expected peak due to 1:3 of 6548 and 6583
    return part1 + part2

def gaussian_plus_tophat(x, mu1, mu2, amp1, amp2, vel1, vel2):
    
    part1 = amp1 * np.exp(-0.5 * ((x-mu1)/vel1)**2 ) #6316 is the expected peak due to 3:1 of 6300 and 6363
    part2 = amp2 * (  abs(x-mu2) < vel2 ).astype(int)
    
    return part1+part2

def fit_scipy_dg(wl, flux, p0, lower_bounds, upper_bounds):
    
    popt, pcov = curve_fit(double_gaussian, wl, flux*10**15, p0 = p0, bounds = (lower_bounds, upper_bounds), maxfev = 10**4)
    popt[2] *= 10**-15
    popt[3] *= 10**-15
    pcov_diag = np.sqrt(np.diag(pcov))
    pcov_diag[2] *= 10**-15
    pcov_diag[3] *= 10**-15
    
    return popt, pcov_diag

def fit_scipy_gth(wl, flux, p0, lower_bounds, upper_bounds):
    
    popt, pcov = curve_fit(gaussian_plus_tophat, wl, flux*10**15, p0 = p0, bounds = (lower_bounds, upper_bounds), maxfev = 10**4)
    popt[2] *= 10**-15
    popt[3] *= 10**-15
    pcov_diag = np.sqrt(np.diag(pcov))
    pcov_diag[2] *= 10**-15
    pcov_diag[3] *= 10**-15
    
    return popt, pcov_diag

def score_normaliser(score, fitting_wl, fitting_flux):
    
    #normalise for the amount of datapoints + the total flux
    score = score/len(fitting_wl)
    score = score/(integrate.cumtrapz(fitting_flux, fitting_wl)[-1])**2
    
    return score

def continuum_remover(wl, flux):
    
    left_continuum_wl =  (wl < 6150) * (wl > 6100)
    right_continuum_wl = (wl < 6900) * (wl > 6850)
    left_continuum_flux_mean = np.mean(flux[left_continuum_wl])
    right_continuum_flux_mean = np.mean(flux[right_continuum_wl])
    
    coefficient = (right_continuum_flux_mean-left_continuum_flux_mean) / (6875-6125)
    continuum_flux_final = coefficient*(wl-6875) + right_continuum_flux_mean
    
    corrected_flux = flux-continuum_flux_final
    return corrected_flux
    
def simultaneous_fitting_dg(wl_list, flux_list, epochs_list, plot = False):
    
    fitting_wl_list, fitting_flux_list = [], []
    for q in range(len(wl_list)):
        wl, flux = wl_list[q], flux_list[q]
        
        #Define the fitting regions
        fitting_mask = (wl < 6900) * (wl > 6100)
        fitting_wl, fitting_flux = wl[fitting_mask], flux[fitting_mask]
        fitting_flux = continuum_remover(fitting_wl, fitting_flux)
        
        
        fitting_wl_list.append(fitting_wl)
        fitting_flux_list.append(fitting_flux)
        
        
    tested_NII_velocities = np.arange(80, 140, 2)
    best_score, shape = np.inf, 'dg'
    for i in range(len(tested_NII_velocities)):
        
        popt_list, pcov_list = [], []
        total_score = 0
        for j in range(len(fitting_wl_list)):
            fitting_wl, fitting_flux = fitting_wl_list[j], fitting_flux_list[j]
            
            NII_dom_mask =  (fitting_wl < 6600) * (fitting_wl > 6540)
            
            velocity = tested_NII_velocities[i]
            time_corr_velocity = velocity - (epochs_list[j]-200)*2/50 #From the models + obs of 2011dh, we see decrease in NII velocity of about 3Å per 50 days 

            p0 = [6300, 6560, 0.1, 0.7*np.max(fitting_flux[NII_dom_mask])*10**15, 50, time_corr_velocity]

            #Some words on the fitting bounds:
            #For NII velocity, we fit all SNe simultaneously, as it should not change significantly with itme (~10-15Å from 200d to 400d)
            #For OI centroid, we allow shift of 1000 km/s or 20 Å (~ 400 km/s from thompson scattering (AJ17), ~200 km/s from viewing angles (vanBaal23)).
            #For NII centroid, we allow shift of 1500 km/s or 30 Å, for possible shifts in other direction than OI (vanBaal23)
            lower_bounds = [6280, 6540, 0, 0, 30, time_corr_velocity-(epochs_list[j]-np.min(epochs_list))*3/50] #time_corr_velocity-(epochs_list[j]-np.min(epochs_list))*3/50
            #10**15 as we do the same to the flux in the actual fitting to avoid numerical problems
            upper_bounds = [6320, 6600, np.inf, 1*np.max(fitting_flux[NII_dom_mask])*10**15, 90, time_corr_velocity+0.01]
            
            #Fit the double gaussian
            popt, pcov = fit_scipy_dg(fitting_wl, fitting_flux, p0, lower_bounds, upper_bounds)
            popt_list.append(popt)
            pcov_list.append(pcov)
        
            ypred = double_gaussian(fitting_wl, *popt)
            score = np.sum((ypred-fitting_flux)**2)
            
            normalised_score = score_normaliser(score, fitting_wl, fitting_flux)
            total_score += normalised_score
            
        if total_score < best_score:
            best_score = total_score
            popt_best = popt_list
            pcov_best = pcov_list
            
            
            
            if i == len(tested_NII_velocities)-1:
                for i in range(len(wl_list)):
                    plt.plot(fitting_wl_list[i], fitting_flux_list[i], label = 'Observed')
                    plt.plot(fitting_wl, gaussian(fitting_wl, popt_best[i][1], popt_best[i][3], popt_best[i][5]), linestyle = '--', label = 'NII')
                    plt.plot(fitting_wl, gaussian(fitting_wl, popt_best[i][0], popt_best[i][2], popt_best[i][4]), linestyle = '--', label = 'OI')
                    plt.plot(fitting_wl, double_gaussian(fitting_wl, *popt_best[i]), c = 'black')
                    plt.show()
                
                
                print("This profile is not well fitted by double gaussian, reverting to tophat now")
                print(popt_best)
                popt_best, pcov_best = simultaneous_fitting_gth(wl_list, flux_list, epochs_list, plot = plot)
                shape = 'gth'
    print(popt_best)
    if plot and shape == 'dg':
        for i in range(len(wl_list)):
            plt.plot(fitting_wl_list[i], fitting_flux_list[i], label = 'Observed')
            plt.plot(fitting_wl, gaussian(fitting_wl, popt_best[i][1], popt_best[i][3], popt_best[i][5]), linestyle = '--', label = 'NII')
            plt.plot(fitting_wl, gaussian(fitting_wl, popt_best[i][0], popt_best[i][2], popt_best[i][4]), linestyle = '--', label = 'OI')
            plt.plot(fitting_wl, double_gaussian(fitting_wl, *popt_best[i]), c = 'black')
            plt.show()
            
    return popt_best, pcov_best, shape
    
    
    
def simultaneous_fitting_gth(wl_list, flux_list, epochs_list, plot = False):
    
    fitting_wl_list, fitting_flux_list = [], []
    for q in range(len(wl_list)):
        wl, flux = wl_list[q], flux_list[q]
        
        #Define the fitting regions
        fitting_mask = (wl < 6900) * (wl > 6100)
        fitting_wl, fitting_flux = wl[fitting_mask], flux[fitting_mask]
        fitting_flux = continuum_remover(fitting_wl, fitting_flux)
        
        fitting_wl_list.append(fitting_wl)
        fitting_flux_list.append(fitting_flux)
        
        
    tested_NII_velocities = np.arange(140, 200, 2) 
    best_score = np.inf
    for i in range(len(tested_NII_velocities)):
    #Set the fitting bounds
        
        
        popt_list, pcov_list = [], []
        total_score = 0
        for j in range(len(fitting_wl_list)):
            fitting_wl, fitting_flux = fitting_wl_list[j], fitting_flux_list[j]
            
            NII_dom_mask =  (fitting_wl < 6600) * (fitting_wl > 6540)
            
            velocity = tested_NII_velocities[i]
            time_corr_velocity = velocity - (epochs_list[j]-200)*(2/50) #From the models + obs of 2011dh, we see decrease in NII velocity of about 3Å per 50 days 

            p0 = [6300, 6560, 0.1, 0.7*np.max(fitting_flux[NII_dom_mask])*10**15, 50, time_corr_velocity]

            #Some words on the fitting bounds:
            #For NII velocity, we fit all SNe simultaneously, as it should not change significantly with itme (~10-15Å from 200d to 400d)
            #For OI centroid, we allow shift of 1000 km/s or 20 Å (~ 400 km/s from thompson scattering (AJ17), ~200 km/s from viewing angles (vanBaal23)).
            #For NII centroid, we allow shift of 1500 km/s or 30 Å, for possible shifts in other direction than OI (vanBaal23)
            lower_bounds = [6280, 6540, 0, 0, 30, time_corr_velocity-(epochs_list[j]-np.min(epochs_list))*3/50]
            #10**15 as we do the same to the flux in the actual fitting to avoid numerical problems
            upper_bounds = [6320, 6600, np.inf, 1*np.max(fitting_flux[NII_dom_mask])*10**15, 90, time_corr_velocity+0.01]
            
            
            #Fit the double gaussian
            popt, pcov = fit_scipy_gth(fitting_wl, fitting_flux, p0, lower_bounds, upper_bounds)
            popt_list.append(popt)
            pcov_list.append(pcov)
        
            ypred = double_gaussian(fitting_wl, *popt)
            score = np.sum((ypred-fitting_flux)**2)
            
            normalised_score = score_normaliser(score, fitting_wl, fitting_flux)
            total_score += normalised_score
        
        if total_score < best_score:
            best_score = total_score
            popt_best = popt_list
            pcov_best = pcov_list
                
    if plot:        
        for i in range(len(wl_list)):
            plt.plot(fitting_wl_list[i], fitting_flux_list[i], label = 'Observed')
            plt.plot(fitting_wl, tophat(fitting_wl, popt_best[i][1], popt_best[i][3], popt_best[i][5]), linestyle = '--', label = 'NII')
            plt.plot(fitting_wl, gaussian(fitting_wl, popt_best[i][0], popt_best[i][2], popt_best[i][4]), linestyle = '--', label = 'OI')
            plt.plot(fitting_wl, gaussian_plus_tophat(fitting_wl, *popt_best[i]), c = 'black')
            plt.show()
            
    return popt_best, pcov_best
    
def uncertainty_gaussian(amp, vel, sigma_amp, sigma_vel):
    sigma_g = np.sqrt(2*np.pi * (amp**2 * sigma_vel**2 + vel**2 * sigma_amp**2))
    return sigma_g
    
def uncertainty_tophat(amp, vel, sigma_amp, sigma_vel):
    sigma_th = 2* np.sqrt(amp**2 * sigma_vel**2 + vel**2 * sigma_amp**2)
    return sigma_th

def uncertainty_ratio(NII_flux, OI_flux, sigma_NII, sigma_OI):
    
    return np.sqrt ( (sigma_OI/OI_flux)**2 + (NII_flux*sigma_NII/(OI_flux**2))**2 )
    
    
def observed_flux_from_fit(wl, flux, popt, pcov, lambda_min, lambda_max, shape):
    
    
    #Get the integrated fluxes
    normalisation_mask = (wl > lambda_min) * (wl < lambda_max)
    integrated_OI_flux = integrate.cumtrapz(gaussian(wl, popt[0], popt[2], popt[4]), wl)[-1]
    uncertainty_OI = uncertainty_gaussian(popt[2], popt[4], pcov[2], pcov[4])
    
    if shape == 'dg':
        integrated_NII_flux = integrate.cumtrapz(gaussian(wl, popt[1], popt[3], popt[5]), wl)[-1]
        uncertainty_NII = uncertainty_gaussian(popt[3], popt[5], pcov[3], pcov[5])
        
    elif shape == 'gth':
        integrated_NII_flux = integrate.cumtrapz(tophat(wl, popt[1], popt[3], popt[5]), wl)[-1]
        uncertainty_NII = uncertainty_tophat(popt[3], popt[5], pcov[3], pcov[5])
        
    
    sigma_ratio = uncertainty_ratio(integrated_NII_flux, integrated_OI_flux, uncertainty_NII, uncertainty_OI)
    print(sigma_ratio)
    return integrated_NII_flux/integrated_OI_flux, sigma_ratio


