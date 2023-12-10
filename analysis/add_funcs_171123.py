import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.integrate as integrate
import time
from astropy.io import ascii
from scipy.optimize import curve_fit
from tqdm import tqdm
from astropy import cosmology

#Constants
c = 299792.458 #km/s
M_sun = 1.989*10**33 #g
escatt_data_loc = '/home/stba7609/NII_nebular_phase/analysis/escatt_data/'


#Some conversion functions to simplify calculations
#=============================================================================================================================
def redshift_correct_wl(wl, own_z = 0):
    
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
#=============================================================================================================================





#Three functions: one to score our fitting result, the other two to remove (pseudo)continuum fluxes
#=============================================================================================================================
def score_normaliser(score, fitting_wl, fitting_flux):
    
    #normalise for the amount of datapoints + the total flux
    score = score/len(fitting_wl)
    score = score/(integrate.cumtrapz(fitting_flux, fitting_wl)[-1])**2
    
    return score

def continuum_remover(wl, flux):
    #This function removes the continuum in the case of fitting the OI and NII profiles
    
    left_continuum_wl =  (wl < 6150) * (wl > 6100)
    right_continuum_wl = (wl < 6900) * (wl > 6850)
    left_continuum_flux_mean = np.mean(flux[left_continuum_wl])
    right_continuum_flux_mean = np.mean(flux[right_continuum_wl])
    
    coefficient = (right_continuum_flux_mean-left_continuum_flux_mean) / (6875-6125)
    continuum_flux_final = coefficient*(wl-6875) + right_continuum_flux_mean
    
    corrected_flux = flux-continuum_flux_final
    return corrected_flux

def tot_continuum_remover(wl, flux):
    #This function removes the continuum when determining the total flux of the spectrum.
    #As the continuum is quite varying between sources, this function is somewhat involved
    #We base it on four regions which according to our models typically have no emission lines and therefore little flux
    
    ranges = [[5740, 5790], [6020, 6070], [6850, 6900], [7950, 8000]]
    means, stds = [], []
    for i in range(len(ranges)):
        mask = (wl > ranges[i][0]) * (wl < ranges[i][1])
        means.append(np.mean(flux[mask]))
        stds.append(np.std(flux[mask]))
        
    sorted_means = sorted(means)
    sorted_stds = sorted(stds)
    
    #Now pick the two lowest values of these 5 typically 'continuum-like' regions
    continuum_flux = ( sorted_means[0] + sorted_means[1] + sorted_means[2])/3
    final_std = (sorted_stds[0] + sorted_stds[1] + sorted_stds[2])/3
    
    corrected_flux = flux-continuum_flux
    
    return corrected_flux, final_std
#=============================================================================================================================





#The three basic functions that we will use to fit
#=============================================================================================================================
def escatt(x, cntr, amp, vel, tau):
    
    escattering_data = np.loadtxt(escatt_data_loc + 'tau_' +  str(int(10*tau)) + '.csv', delimiter = ',') #*10 as names were given this way
            
    abs_wl_pt1 = escattering_data[:, 0] * vel + cntr
    abs_flux_pt1 = amp * escattering_data[:, 1]

    return np.interp(x, abs_wl_pt1, abs_flux_pt1)

def gaussian(x, cntr, amp, vel):
    
    return amp * np.exp(-0.5 * ((x-cntr)/vel)**2 )

def thickshell(x, cntr, amp, vel, vmax):
    
    mask1 = (x <= cntr + vel) * (x >= cntr -vel)
    mask2 = x > cntr + vel
    mask3 = x < cntr - vel

    y = np.zeros(len(mask1))
    y[mask1] = amp
    y[mask2] = amp * (1-( (x[mask2]-(cntr+vel))/vmax )**2 )
    y[mask3] = amp * (1-( (x[mask3]-(cntr-vel))/vmax )**2 )

    y[y < 0] = 0
    
    return y
#=============================================================================================================================





#The four composite functions that will be fit
#=============================================================================================================================
def escatt_plus_gaussian_global(x, mu1, mu2, amp1, amp2, vel1, vel2, tau):
    
    part1 = escatt(x, mu1, amp1, vel1, tau)
    part2 = gaussian(x, mu2, amp2, vel2)

    return part1 + part2

def double_gaussian(x, mu1, mu2, amp1, amp2, vel1, vel2):
    
    part1 = gaussian(x, mu1, amp1, vel1)
    part2 = gaussian(x, mu2, amp2, vel2)
    
    return part1 + part2


def escatt_plus_thickshell_global(x, mu1, mu2, amp1, amp2, vel1, vel2, vmax, tau):
    
    part1 = escatt(x, mu1, amp1, vel1, tau)
    part2 = thickshell(x, mu2, amp2, vel2, vmax)
   
    return part1 + part2

def gaussian_plus_thickshell_global(x, mu1, mu2, amp1, amp2, vel1, vel2, vmax):
    
    part1 = gaussian(x, mu1, amp1, vel1)
    part2 = thickshell(x, mu2, amp2, vel2, vmax)
    
    return part1 + part2
#=============================================================================================================================





#Functions trying to fit each of the four possible composite functions
#=============================================================================================================================
def fit_scipy_eg(wl, flux, p0, lower_bounds, upper_bounds, rounded_tau):
    
    def escatt_plus_gaussian_local(x, mu1, mu2, amp1, amp2, vel1, vel2):
        
        part1 = escatt(x, mu1, amp1, vel1, rounded_tau)
        part2 = gaussian(x, mu2, amp2, vel2)
        
        return part1 + part2
    
    
    popt, pcov = curve_fit(escatt_plus_gaussian_local, wl, flux*10**15, p0 = p0, bounds = (lower_bounds, upper_bounds))
    pcov_diag = np.sqrt(np.diag(pcov))
    popt[2] *= 10**-15
    popt[3] *= 10**-15
        
    return popt, pcov_diag

def fit_scipy_dg(wl, flux, p0, lower_bounds, upper_bounds):
    
    popt, pcov = curve_fit(double_gaussian, wl, flux*10**15, p0 = p0, bounds = (lower_bounds, upper_bounds)) 
    pcov_diag = np.sqrt(np.diag(pcov))
    popt[2] *= 10**-15
    popt[3] *= 10**-15
    
    return popt, pcov_diag


def fit_scipy_ets(wl, flux, p0, lower_bounds, upper_bounds, rounded_tau):
    
    def escatt_plus_thickshell_local(x, mu1, mu2, amp1, amp2, vel1, tested_vel, ratio):
        
        part1 = escatt(x, mu1, amp1, vel1, rounded_tau)
        part2 = thickshell(x, mu2, amp2, ratio*tested_vel, (1-ratio)*tested_vel)
        
        return part1 + part2
    
    popt, pcov = curve_fit(escatt_plus_thickshell_local, wl, flux*10**15, p0 = p0, bounds = (lower_bounds, upper_bounds))
    pcov_diag = np.sqrt(np.diag(pcov))
    popt[2] *= 10**-15
    popt[3] *= 10**-15
        
    return popt, pcov_diag
    

def fit_scipy_gts(wl, flux, p0, lower_bounds, upper_bounds):
    
    def gaussian_plus_thickshell_local(x, mu1, mu2, amp1, amp2, vel1, tested_vel, ratio):
        
        part1 = gaussian(x, mu1, amp1, vel1)
        part2 = thickshell(x, mu2, amp2, ratio*tested_vel, (1-ratio)*tested_vel)
        
        return part1 + part2
    
    popt, pcov = curve_fit(gaussian_plus_thickshell_local, wl, flux*10**15, p0 = p0, bounds = (lower_bounds, upper_bounds))
    pcov_diag = np.sqrt(np.diag(pcov))
    popt[2] *= 10**-15
    popt[3] *= 10**-15
    
    return popt, pcov_diag
#=============================================================================================================================




    
#The two functions doing the main job: They check for the case of an NII gaussian (g) and an NII thickshell (ts), whether an OI gaussian or an OI escatt fits best
#=============================================================================================================================
def simultaneous_fitting_g(wl_list, flux_list, epochs_list, plot = False):
    
    fitting_wl_list, fitting_flux_list = [], []
    noise_list = []
    
    #Reduce the spectra to only the region of interest (i.e. where NII and OI are)
    for q in range(len(wl_list)):
        wl, flux = wl_list[q], flux_list[q]
        
        #Define the fitting regions
        flux = continuum_remover(wl, flux)
        fitting_mask = (wl < 6900) * (wl > 6100)
        fitting_wl, fitting_flux = wl[fitting_mask], flux[fitting_mask]
        
        fitting_wl_list.append(fitting_wl)
        fitting_flux_list.append(fitting_flux)
        
        
    #Define the tested ranges
    tested_NII_velocities = np.arange(80, 150, 2)
    tested_tau_init = np.arange(0, 3, 0.2)
    avail_tau = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3]) #These are the 'true' values for which we have data. Later, when evaluating a given tau, it will be rounded to the closest value in this array
    best_score, best_i = np.inf, 0
    
    #Loop over the two ranges
    for i in tqdm(range(len(tested_NII_velocities))):
        if i-best_i > 7: #Break the testing when we don't see any improvement
            break
            
        for q in range(len(tested_tau_init)):
        
            popt_list_eg, pcov_list_eg, opt_tau_list = [], [], []
            popt_list_dg, pcov_list_dg = [], []
            total_score_eg, total_score_dg = 0, 0
            
            velocity = tested_NII_velocities[i]
            tau = tested_tau_init[q]
            
            #This is the main idea of our fitting algorithm: using one given set of NII velocities (and tau scaling), fit all observations simultaneously, and use their total score to find out what set of NII velocities fits the obs best.
            
            for j in range(len(fitting_wl_list)):
                fitting_wl, fitting_flux = fitting_wl_list[j], fitting_flux_list[j]

                NII_dom_mask =  (fitting_wl < 6600) * (fitting_wl > 6540)

                
                time_corr_velocity = velocity - (epochs_list[j]-100)*2/50 #From the models + obs of 2011dh, we see decrease in NII velocity of about 3Å per 50 days 
                time_corr_tau = tau/(epochs_list[j]/np.min(epochs_list))**2 #Tau scales roughly with t^-2, force this here
                rounded_tau = avail_tau[np.where(abs(time_corr_tau-avail_tau) == np.min(abs(time_corr_tau-avail_tau)))[0][0]]
                
                
                #Some words on the fitting bounds:
                #For NII velocity, we fit all SNe simultaneously, as it should not change significantly with itme (~10-15Å from 200d to 400d)
                #For OI centroid, we allow shift of 1000 km/s or 20 Å (~ 600 km/s from thompson scattering (AJ17), ~200 km/s from viewing angles (vanBaal23)).
                #For NII centroid, we allow shift of 1500 km/s or 30 Å, for possible shifts in other direction than OI (vanBaal23)
                p0 = [6310, 6560, 1*np.max(fitting_flux)*10**15, 0.7*np.max(fitting_flux[NII_dom_mask])*10**15, 50, time_corr_velocity]
                lower_bounds = [6290, 6540, 0, 0, 30, time_corr_velocity-(epochs_list[j]-100)*2/50] 
                upper_bounds = [6340, 6590, 2.5*np.max(fitting_flux)*10**15, 1*np.max(fitting_flux[NII_dom_mask])*10**15, 130, time_corr_velocity+0.01]

                
                #Fit the escatt plus gaussian!
                
                try:
                    popt_eg, pcov_eg = fit_scipy_eg(fitting_wl, fitting_flux, p0, lower_bounds, upper_bounds, rounded_tau)
                except RuntimeError: #The fitting did not converge: return flags
                    popt_eg, pcov_eg = [-1, -1, -1, -1, -1, -1, 0.0], [-1, -1, -1, -1, -1, -1, 0.0]
                 
                
                #Fit the double gaussian!
                
                try:
                    popt_dg, pcov_dg = fit_scipy_dg(fitting_wl, fitting_flux, p0, lower_bounds, upper_bounds)
                except:
                    popt_dg, pcov_dg = [-2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2]

                #And now we score the results:
                ypred_eg = escatt_plus_gaussian_global(fitting_wl, *popt_eg, rounded_tau)
                ypred_dg = double_gaussian(fitting_wl, *popt_dg)

                score_eg = np.sum((ypred_eg-fitting_flux)**2)
                score_dg = np.sum((ypred_dg-fitting_flux)**2)
                
                normalised_score_eg = score_normaliser(score_eg, fitting_wl, fitting_flux)
                normalised_score_dg = score_normaliser(score_dg, fitting_wl, fitting_flux)
                
                if rounded_tau == 0: #If tau = 0, revert to gaussian
                    popt_eg, pcov_eg = popt_dg, pcov_dg
                    normalised_score_eg = normalised_score_dg

                #Store the best fit for each individual epoch in a list
                popt_list_eg.append(popt_eg)
                pcov_list_eg.append(pcov_eg)
                opt_tau_list.append(rounded_tau)
                
                popt_list_dg.append(popt_dg)
                pcov_list_dg.append(pcov_dg)
                
                total_score_eg += normalised_score_eg
                total_score_dg += normalised_score_dg
                
            if total_score_eg < total_score_dg: 
                total_score, popt_list, pcov_list, opt_tau_list = total_score_eg, popt_list_eg, pcov_list_eg, opt_tau_list
            else:
                total_score, popt_list, pcov_list, opt_tau_list = total_score_dg, popt_list_dg, pcov_list_dg, [-1]*len(fitting_wl_list)
                
            if total_score < best_score:
                best_score = total_score
                popt_best = popt_list
                pcov_best = pcov_list
                tau_best = opt_tau_list
                print('We have a new best fit! (NII_vel, eg_score, dg_score): ', velocity, '{:.3g}'.format(total_score_eg), '{:.3g}'.format(total_score_dg))
                best_i = i
            
            
    print('This is our final best fit: ', popt_best)
    print('The opt taulist for gaussian is: ', tau_best)
    print('And the best fits look as follows: ')
    
    if plot:
        for i in range(len(wl_list)):
            plt.plot(fitting_wl_list[i], fitting_flux_list[i], label = 'Observed')
            plt.plot(fitting_wl, gaussian(fitting_wl, popt_best[i][1], popt_best[i][3], popt_best[i][5]), linestyle = '--', label = 'NII')
            if tau_best[i] < 0.1:
                plt.plot(fitting_wl, gaussian(fitting_wl, popt_best[i][0], popt_best[i][2], popt_best[i][4]), linestyle = '--', label = 'OI')
                plt.plot(fitting_wl, double_gaussian(fitting_wl, *popt_best[i]), c = 'black')
            else:
                plt.plot(fitting_wl, escatt(fitting_wl, popt_best[i][0], popt_best[i][2], popt_best[i][4], tau_best[i]), linestyle = '--', label = 'OI')
                plt.plot(fitting_wl, escatt_plus_gaussian_global(fitting_wl, *popt_best[i], tau_best[i]), c = 'black')
            plt.show()
            
            
    return popt_best, pcov_best, best_score, tau_best
    
    
def simultaneous_fitting_ts(wl_list, flux_list, epochs_list, plot = False):
    
    fitting_wl_list, fitting_flux_list = [], []
    noise_list = []
    
    #Reduce the spectra to only the region of interest (i.e. where NII and OI are)
    for q in range(len(wl_list)):
        wl, flux = wl_list[q], flux_list[q]
        
        #Define the fitting regions
        flux = continuum_remover(wl, flux)
        fitting_mask = (wl < 6900) * (wl > 6100)
        fitting_wl, fitting_flux = wl[fitting_mask], flux[fitting_mask]
        
        fitting_wl_list.append(fitting_wl)
        fitting_flux_list.append(fitting_flux)
        
        
    #Define the tested ranges
    tested_NII_velocities = np.arange(170, 250, 2)
    tested_tau_init = np.arange(0, 3, 0.2)
    avail_tau = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3]) #These are the 'true' values for which we have data. Later, when evaluating a given tau, it will be rounded to the closest value in this array
    best_score, best_i = np.inf, 0
    
    #Loop over the two ranges
    for i in tqdm(range(len(tested_NII_velocities))):
        if i-best_i > 7: #Break the testing when we don't see any improvement
            break
        
        for q in range(len(tested_tau_init)):
        
            popt_list_ets, pcov_list_ets, opt_tau_list = [], [], []
            popt_list_gts, pcov_list_gts = [], []
            total_score_ets, total_score_gts = 0, 0
            
            velocity = tested_NII_velocities[i]
            tau = tested_tau_init[q]
            
            #This is the main idea of our fitting algorithm: using one given set of NII velocities (and tau scaling), fit all observations simultaneously, and use their total score to find out what set of NII velocities fits the obs best.
            
            for j in range(len(fitting_wl_list)):
                fitting_wl, fitting_flux = fitting_wl_list[j], fitting_flux_list[j]

                NII_dom_mask =  (fitting_wl < 6600) * (fitting_wl > 6540)

                
                time_corr_velocity = velocity - (epochs_list[j]-100)*2/50 #From the models + obs of 2011dh, we see decrease in NII velocity of about 3Å per 50 days 
                time_corr_tau = tau/(epochs_list[j]/np.min(epochs_list))**2 #Tau scales roughly with t^-2, force this here
                rounded_tau = avail_tau[np.where(abs(time_corr_tau-avail_tau) == np.min(abs(time_corr_tau-avail_tau)))[0][0]]
                
                
                #Some words on the fitting bounds:
                #For NII velocity, we fit all SNe simultaneously, as it should not change significantly with itme (~10-15Å from 200d to 400d)
                #For OI centroid, we allow shift of 1000 km/s or 20 Å (~ 600 km/s from thompson scattering (AJ17), ~200 km/s from viewing angles (vanBaal23)).
                #For NII centroid, we allow shift of 1500 km/s or 30 Å, for possible shifts in other direction than OI (vanBaal23)
                p0 = [6310, 6560, 1*np.max(fitting_flux)*10**15, 0.7*np.max(fitting_flux[NII_dom_mask])*10**15, 50, time_corr_velocity, 0.5]
                lower_bounds = [6290, 6540, 0, 0, 30, time_corr_velocity-(epochs_list[j]-100)*2/50, 0.4] 
                upper_bounds = [6340, 6590, 2.5*np.max(fitting_flux)*10**15, 1*np.max(fitting_flux[NII_dom_mask])*10**15, 130, time_corr_velocity+0.01, 1]

                
                #Fit the escatt plus thickshell!
                
                try:
                    popt_ets, pcov_ets = fit_scipy_ets(fitting_wl, fitting_flux, p0, lower_bounds, upper_bounds, rounded_tau)
                except RuntimeError: #The fitting did not converge: return flags
                    popt_ets, pcov_ets = [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]
                
                #Fit the gaussian plus thickshell!
                
                try:
                    popt_gts, pcov_gts = fit_scipy_gts(fitting_wl, fitting_flux, p0, lower_bounds, upper_bounds)
                except:
                    popt_gts, pcov_gts = [-2, -2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2]

                #Translate the tested_vel and ratio parameters back to vel2 and vmax
                popt_ets[5], popt_ets[6] = popt_ets[5]*popt_ets[6], popt_ets[5]*(1-popt_ets[6])
                popt_gts[5], popt_gts[6] = popt_gts[5]*popt_gts[6], popt_gts[5]*(1-popt_gts[6])
                
                #And now we score the results:
                ypred_ets = escatt_plus_thickshell_global(fitting_wl, *popt_ets, rounded_tau)
                ypred_gts = gaussian_plus_thickshell_global(fitting_wl, *popt_gts)

                score_ets = np.sum((ypred_ets-fitting_flux)**2)
                score_gts = np.sum((ypred_gts-fitting_flux)**2)
                
                normalised_score_ets = score_normaliser(score_ets, fitting_wl, fitting_flux)
                normalised_score_gts = score_normaliser(score_gts, fitting_wl, fitting_flux)
                
                if rounded_tau == 0: #If tau = 0, revert to gaussian
                    popt_ets, pcov_ets = popt_gts, pcov_gts
                    normalised_score_ets = normalised_score_gts

                #Store the best fit for each individual epoch in a list
                popt_list_ets.append(popt_ets)
                pcov_list_ets.append(pcov_ets)
                opt_tau_list.append(rounded_tau)
                
                popt_list_gts.append(popt_gts)
                pcov_list_gts.append(pcov_gts)
                
                total_score_ets += normalised_score_ets
                total_score_gts += normalised_score_gts
                
            if total_score_ets < total_score_gts: 
                total_score, popt_list, pcov_list, opt_tau_list = total_score_ets, popt_list_ets, pcov_list_ets, opt_tau_list
            else:
                total_score, popt_list, pcov_list, opt_tau_list = total_score_gts, popt_list_gts, pcov_list_gts, [-1]*len(fitting_wl_list)
                
            if total_score < best_score:
                best_score = total_score
                popt_best = popt_list
                pcov_best = pcov_list
                tau_best = opt_tau_list
                best_i = i
                print('We have a new best fit! (NII_vel, ets_score, gts_score): ', velocity, '{:.3g}'.format(total_score_ets), '{:.3g}'.format(total_score_gts))
            
            
    print('This is our final best fit: ', popt_best)
    print('The opt taulist for thickshell is: ', tau_best)
    print('And the best fits look as follows: ')
    
    if plot:
        for i in range(len(wl_list)):
            plt.plot(fitting_wl_list[i], fitting_flux_list[i], label = 'Observed')
            plt.plot(fitting_wl, thickshell(fitting_wl, popt_best[i][1], popt_best[i][3], popt_best[i][5], popt_best[i][6]), linestyle = '--', label = 'NII')
            if tau_best[i] < 0.1:
                plt.plot(fitting_wl, gaussian(fitting_wl, popt_best[i][0], popt_best[i][2], popt_best[i][4]), linestyle = '--', label = 'OI')
                plt.plot(fitting_wl, gaussian_plus_thickshell_global(fitting_wl, *popt_best[i]), c = 'black')
            else:
                plt.plot(fitting_wl, escatt(fitting_wl, popt_best[i][0], popt_best[i][2], popt_best[i][4], tau_best[i]), linestyle = '--', label = 'OI')
                plt.plot(fitting_wl, escatt_plus_thickshell_global(fitting_wl, *popt_best[i], tau_best[i]), c = 'black')
            plt.show()
            
    return popt_best, pcov_best, best_score, tau_best
    
    
def observed_flux_from_fit(wl, flux, popt, pcov, lambda_min, lambda_max, shape):
    
    
    #Get the integrated fluxes
    normalisation_mask = (wl > lambda_min) * (wl < lambda_max)
    
    tot_cont_removed_flux, noise_level = tot_continuum_remover(wl[normalisation_mask], flux[normalisation_mask]) #flux[normalisation_mask]
    integratable_flux = np.copy(tot_cont_removed_flux)
    integratable_flux[integratable_flux < 0] = 0
    
    #Plot the total continuum removal
    plt.plot(wl[normalisation_mask], flux[normalisation_mask])
    plt.plot(wl[normalisation_mask], tot_cont_removed_flux)
    plt.axhline(y = 0, linestyle = '--')
    plt.show()
    
    integrated_total_flux = integrate.cumtrapz(integratable_flux, wl[normalisation_mask])[-1]
    
    if shape == 'g':
        integrated_NII_flux = integrate.cumtrapz(gaussian(wl, popt[1], popt[3], popt[5]), wl)[-1]
        
    elif shape == 'ts':
        integrated_NII_flux = integrate.cumtrapz(thickshell(wl, popt[1], popt[3], popt[5], popt[6]), wl)[-1]
        
    
    SNR = popt[3]/noise_level
    
    SNR_sigma = integrated_NII_flux/SNR
    
    #sigma = uncertainty_NII/integrated_total_flux
    print('The SNR for this spectrum is: ', SNR)
    print('With uncertainty: ', SNR_sigma/integrated_total_flux)
    
    return integrated_NII_flux/integrated_total_flux, SNR_sigma/integrated_total_flux


