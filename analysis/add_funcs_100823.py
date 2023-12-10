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

def escatt(x, cntr, amp, vel, tau):
    
    escattering_data = np.loadtxt(escatt_data_loc + 'tau_' +  str(int(10*tau)) + '.csv', delimiter = ',') #*10 as names were given this way
            
    abs_wl_pt1 = escattering_data[:, 0] * vel + cntr
    abs_flux_pt1 = amp * escattering_data[:, 1]

    return np.interp(x, abs_wl_pt1, abs_flux_pt1)

def gaussian(x, cntr, amp, vel):
    
    return amp * np.exp(-0.5 * ((x-cntr)/vel)**2 )

def tophat(x, cntr, amp, vel):
    
    return amp * (  abs(x-cntr) < vel ).astype(int) #1.5 is simply redefining the tophat

def escatt_plus_gaussian_global(x, mu1, mu2, amp1, amp2, vel1, vel2, tau):
    
    escattering_data = np.loadtxt(escatt_data_loc + 'tau_' +  str(int(10*tau)) + '.csv', delimiter = ',') #*10 as names were given this way
            
    abs_wl_pt1 = escattering_data[:, 0] * vel1 + mu1
    abs_flux_pt1 = amp1 * escattering_data[:, 1]

    part1 = np.interp(x, abs_wl_pt1, abs_flux_pt1)
    part2 = amp2 * np.exp(-0.5 * ((x-mu2)/vel2)**2 )

    return part1 + part2

def double_gaussian(x, mu1, mu2, amp1, amp2, vel1, vel2):
    
    part1 = amp1 * np.exp(-0.5 * ((x-mu1)/vel1)**2 ) #6316 is the expected peak due to 3:1 of 6300 and 6363
    part2 = amp2 * np.exp(-0.5 * ((x-mu2)/vel2)**2 ) #6575 is the expected peak due to 1:3 of 6548 and 6583
    return part1 + part2

def escatt_plus_tophat_global(x, mu1, mu2, amp1, amp2, vel1, vel2, tau):
    
    escattering_data = np.loadtxt(escatt_data_loc + 'tau_' +  str(int(10*tau)) + '.csv', delimiter = ',') #*10 as names were given this way
            
    abs_wl_pt1 = escattering_data[:, 0] * vel1 + mu1
    abs_flux_pt1 = amp1 * escattering_data[:, 1]

    part1 = np.interp(x, abs_wl_pt1, abs_flux_pt1)
    part2 = amp2 * (  abs(x-mu2) < vel2 ).astype(int)
    
    return part1 + part2

def gaussian_plus_tophat(x, mu1, mu2, amp1, amp2, vel1, vel2):
    
    part1 = amp1 * np.exp(-0.5 * ((x-mu1)/vel1)**2 ) #6316 is the expected peak due to 3:1 of 6300 and 6363
    part2 = amp2 * (  abs(x-mu2) < vel2 ).astype(int)
    
    return part1 + part2


def fit_scipy_eg(wl, flux, p0, lower_bounds, upper_bounds, rounded_tau):
    
    escattering_data = np.loadtxt(escatt_data_loc + 'tau_' +  str(int(10*rounded_tau)) + '.csv', delimiter = ',') #*10 as names were given this way

    def escatt_plus_gaussian_local(x, mu1, mu2, amp1, amp2, vel1, vel2):

        abs_wl_pt1 = escattering_data[:, 0] * vel1 + mu1
        abs_flux_pt1 = amp1 * escattering_data[:, 1]
        
        part1 = np.interp(x, abs_wl_pt1, abs_flux_pt1)
        
        part2 = amp2 * np.exp(-0.5 * ((x-mu2)/vel2)**2 )

        return part1 + part2

    popt, pcov, infodict, mesg, ier = curve_fit(escatt_plus_gaussian_local, wl, flux*10**15, p0 = p0, bounds = (lower_bounds, upper_bounds), full_output = True)
    popt[2] *= 10**-15
    popt[3] *= 10**-15
    pcov_diag = np.sqrt(np.diag(pcov))
    pcov_diag[2] *= 10**-15
    pcov_diag[3] *= 10**-15
        
    return popt, pcov_diag
    

def fit_scipy_dg(wl, flux, p0, lower_bounds, upper_bounds):
    
    popt, pcov = curve_fit(double_gaussian, wl, flux*10**15, p0 = p0, bounds = (lower_bounds, upper_bounds))
    popt[2] *= 10**-15
    popt[3] *= 10**-15
    pcov_diag = np.sqrt(np.diag(pcov))
    pcov_diag[2] *= 10**-15
    pcov_diag[3] *= 10**-15
    
    return popt, pcov_diag

def fit_scipy_eth(wl, flux, p0, lower_bounds, upper_bounds, rounded_tau):
    
    escattering_data = np.loadtxt(escatt_data_loc + 'tau_' +  str(int(10*rounded_tau)) + '.csv', delimiter = ',') #*10 as names were given this way

    def escatt_plus_tophat_local(x, mu1, mu2, amp1, amp2, vel1, vel2):

        abs_wl_pt1 = escattering_data[:, 0] * vel1 + mu1
        abs_flux_pt1 = amp1 * escattering_data[:, 1]
        
        part1 = np.interp(x, abs_wl_pt1, abs_flux_pt1)
            
        part2 = amp2 * (  abs(x-mu2) < vel2 ).astype(int)

        return part1 + part2

    popt, pcov, infodict, mesg, ier = curve_fit(escatt_plus_tophat_local, wl, flux*10**15, p0 = p0, bounds = (lower_bounds, upper_bounds), full_output = True)
    popt[2] *= 10**-15
    popt[3] *= 10**-15
    pcov_diag = np.sqrt(np.diag(pcov))
    pcov_diag[2] *= 10**-15
    pcov_diag[3] *= 10**-15
        
    return popt, pcov_diag

def fit_scipy_gth(wl, flux, p0, lower_bounds, upper_bounds):
    
    popt, pcov = curve_fit(gaussian_plus_tophat, wl, flux*10**15, p0 = p0, bounds = (lower_bounds, upper_bounds))
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
    #This function removes the continuum in the case of fitting the OI and NII profiles
    
    left_continuum_wl =  (wl < 6150) * (wl > 6100)
    right_continuum_wl = (wl < 6900) * (wl > 6850)
    left_continuum_flux_mean = np.mean(flux[left_continuum_wl])
    right_continuum_flux_mean = np.mean(flux[right_continuum_wl])
    
    coefficient = (right_continuum_flux_mean-left_continuum_flux_mean) / (6875-6125)
    continuum_flux_final = coefficient*(wl-6875) + right_continuum_flux_mean
    
    corrected_flux = flux-continuum_flux_final
    return corrected_flux

def tot_continuum_remover(wl, flux, lambda_min, lambda_max):
    #This function removes the continuum in the case of measuring the total integrated flux
    
    wl_window_size = 50 #Å
    wl_windows = np.arange(lambda_min, lambda_max, wl_window_size) #Å
    
    lowest_avg = np.inf
    for i in range(len(wl_windows)-1):
        window_left, window_right = wl_windows[i], wl_windows[i+1]
        window_mask = (wl < window_right) * (wl > window_left)
        avg_flux = np.mean(flux[window_mask])
        if avg_flux < lowest_avg:
            lowest_avg = avg_flux
            
    flux = flux-lowest_avg #Now subtract the lowest avg flux present, effectively removing the continuum
    return flux

def simultaneous_fitting_g(wl_list, flux_list, epochs_list, plot = False):
    
    fitting_wl_list, fitting_flux_list, g_check = [], [], True
    for q in range(len(wl_list)):
        wl, flux = wl_list[q], flux_list[q]
        
        #Define the fitting regions
        fitting_mask = (wl < 6900) * (wl > 6100)
        fitting_wl, fitting_flux = wl[fitting_mask], flux[fitting_mask]
        fitting_flux = continuum_remover(fitting_wl, fitting_flux)
        
        
        fitting_wl_list.append(fitting_wl)
        fitting_flux_list.append(fitting_flux)
        
        
    tested_NII_velocities = np.arange(80, 140, 2)
    tested_tau_init = np.arange(0, 3, 0.1)
    best_score = np.inf
    for i in tqdm(range(len(tested_NII_velocities))):
        for q in range(len(tested_tau_init)):
        
            popt_list_eg, pcov_list_eg, opt_tau_list = [], [], []
            popt_list_dg, pcov_list_dg = [], []
            total_score_eg, total_score_dg = 0, 0
            for j in range(len(fitting_wl_list)):
                fitting_wl, fitting_flux = fitting_wl_list[j], fitting_flux_list[j]

                NII_dom_mask =  (fitting_wl < 6600) * (fitting_wl > 6540)

                velocity = tested_NII_velocities[i]
                tau = tested_tau_init[q]
                time_corr_velocity = velocity - (epochs_list[j]-200)*2/50 #From the models + obs of 2011dh, we see decrease in NII velocity of about 3Å per 50 days 
                time_corr_tau = tau/(epochs_list[j]/np.min(epochs_list))**2 #Tau scales roughly with t^-2, force this here

                p0 = [6310, 6560, 1*np.max(fitting_flux), 0.7*np.max(fitting_flux[NII_dom_mask])*10**15, 50, time_corr_velocity]

                #Some words on the fitting bounds:
                #For NII velocity, we fit all SNe simultaneously, as it should not change significantly with itme (~10-15Å from 200d to 400d)
                #For OI centroid, we allow shift of 1000 km/s or 20 Å (~ 600 km/s from thompson scattering (AJ17), ~200 km/s from viewing angles (vanBaal23)).
                #For NII centroid, we allow shift of 1500 km/s or 30 Å, for possible shifts in other direction than OI (vanBaal23)
                lower_bounds = [6290, 6540, 0, 0, 30, time_corr_velocity-(epochs_list[j]-np.min(epochs_list))*3/50] 
                #10**15 as we do the same to the flux in the actual fitting to avoid numerical problems
                upper_bounds = [6340, 6590, 1.2*np.max(fitting_flux)*10**15, 1*np.max(fitting_flux[NII_dom_mask])*10**15, 120, time_corr_velocity+0.01] #velocity+0.01

                #Fit the gaussian profiles
                avail_tau = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
                rounded_tau = avail_tau[np.where(abs(time_corr_tau-avail_tau) == np.min(abs(time_corr_tau-avail_tau)))[0][0]]
                
                popt_eg, pcov_eg = fit_scipy_eg(fitting_wl, fitting_flux, p0, lower_bounds, upper_bounds, rounded_tau)
                popt_dg, pcov_dg = fit_scipy_dg(fitting_wl, fitting_flux, p0, lower_bounds, upper_bounds)

                ypred_eg = escatt_plus_gaussian_global(fitting_wl, *popt_eg, rounded_tau)
                ypred_dg = double_gaussian(fitting_wl, *popt_dg)

                score_eg = np.sum((ypred_eg-fitting_flux)**2)
                score_dg = np.sum((ypred_dg-fitting_flux)**2)
                
                normalised_score_eg = score_normaliser(score_eg, fitting_wl, fitting_flux)
                normalised_score_dg = score_normaliser(score_dg, fitting_wl, fitting_flux)
                
                if rounded_tau == 0: #If tau = 0, revert to gaussian
                    popt_eg, pcov_eg = popt_dg, pcov_dg
                    normalised_score_eg = normalised_score_dg

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
                print(velocity, total_score_eg, total_score_dg)

                if i == len(tested_NII_velocities)-1: #The velocity for a gaussian at NII is too high: continue to tophat
                    g_check = False
            
    print(popt_best)        
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
            
    print('The opt taulist for gaussian is', tau_best)
            
    return popt_best, pcov_best, best_score, g_check
    
    
def simultaneous_fitting_th(wl_list, flux_list, epochs_list, plot = False):
    
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
    tested_tau_init = np.arange(0, 3, 0.1)
    best_score = np.inf
    for i in tqdm(range(len(tested_NII_velocities))):
        for q in range(len(tested_tau_init)):
            popt_list_eth, pcov_list_eth, opt_tau_list = [], [], []
            popt_list_gth, pcov_list_gth = [], []
            total_score_eth, total_score_gth = 0, 0
        
            for j in range(len(fitting_wl_list)):
                fitting_wl, fitting_flux = fitting_wl_list[j], fitting_flux_list[j]

                NII_dom_mask =  (fitting_wl < 6600) * (fitting_wl > 6540)

                velocity = tested_NII_velocities[i]
                tau = tested_tau_init[q]
                time_corr_velocity = velocity - (epochs_list[j]-200)*2/50 #From the models + obs of 2011dh, we see decrease in NII velocity of about 3Å per 50 days 
                time_corr_tau = tau/(epochs_list[j]/np.min(epochs_list))**2 #Tau scales roughly with t^-2, force this here

                p0 = [6310, 6560, 1*np.max(fitting_flux), 0.7*np.max(fitting_flux[NII_dom_mask])*10**15, 50, time_corr_velocity]

                #Some words on the fitting bounds:
                #For NII velocity, we fit all SNe simultaneously, as it should not change significantly with itme (~10-15Å from 200d to 400d)
                #For OI centroid, we allow shift of 1000 km/s or 20 Å (~ 400 km/s from thompson scattering (AJ17), ~200 km/s from viewing angles (vanBaal23)).
                #For NII centroid, we allow shift of 1500 km/s or 30 Å, for possible shifts in other direction than OI (vanBaal23)
                lower_bounds = [6290, 6540, 0, 0, 30, time_corr_velocity-(epochs_list[j]-np.min(epochs_list))*3/50] 
                #10**15 as we do the same to the flux in the actual fitting to avoid numerical problems
                upper_bounds = [6340, 6590, 1.2*np.max(fitting_flux)*10**15, 1*np.max(fitting_flux[NII_dom_mask])*10**15, 120, time_corr_velocity+0.01] #velocity+0.01

                #Fit the tophat profiles
                avail_tau = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
                rounded_tau = avail_tau[np.where(abs(time_corr_tau-avail_tau) == np.min(abs(time_corr_tau-avail_tau)))[0][0]]
                
                try:
                    popt_eth, pcov_eth = fit_scipy_eth(fitting_wl, fitting_flux, p0, lower_bounds, upper_bounds, rounded_tau)
                except RuntimeError:
                    #Scipy did not find the best params: this is probably not the optimal solution, so set the values to -1 
                    popt_eth, pcov_eth = [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]
                    
                try:
                    popt_gth, pcov_gth = fit_scipy_gth(fitting_wl, fitting_flux, p0, lower_bounds, upper_bounds)
                except RuntimeError:
                    #Scipy did not find the best params: this is probably not the optimal solution, so set the values to -2
                    popt_eth, pcov_eth = [-2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2]

                ypred_eth = escatt_plus_tophat_global(fitting_wl, *popt_eth, rounded_tau)
                ypred_gth = gaussian_plus_tophat(fitting_wl, *popt_gth)

                score_eth = np.sum((ypred_eth-fitting_flux)**2)
                score_gth = np.sum((ypred_gth-fitting_flux)**2)
                
                normalised_score_eth = score_normaliser(score_eth, fitting_wl, fitting_flux)
                normalised_score_gth = score_normaliser(score_gth, fitting_wl, fitting_flux)
                
                if rounded_tau == 0: #If tau = 0, revert to gaussian
                    popt_eth, pcov_eth = popt_gth, pcov_gth
                    normalised_score_eth = normalised_score_gth
                
                popt_list_eth.append(popt_eth)
                pcov_list_eth.append(pcov_eth)
                opt_tau_list.append(rounded_tau)
                    
                popt_list_gth.append(popt_gth)
                pcov_list_gth.append(pcov_gth)

                total_score_eth += normalised_score_eth
                total_score_gth += normalised_score_gth

            if total_score_eth < total_score_gth:
                total_score, popt_list, pcov_list, opt_tau_list = total_score_eth, popt_list_eth, pcov_list_eth, opt_tau_list
            else:
                total_score, popt_list, pcov_list, opt_tau_list = total_score_gth, popt_list_gth, pcov_list_gth, [-1]*len(fitting_wl_list)

            if total_score < best_score:
                best_score = total_score
                popt_best = popt_list
                pcov_best = pcov_list
                tau_best = opt_tau_list
                print(velocity, total_score_eth, total_score_gth)
            
            
    print(popt_best)        
    if plot:
        for i in range(len(wl_list)):
            plt.plot(fitting_wl_list[i], fitting_flux_list[i], label = 'Observed')
            plt.plot(fitting_wl, tophat(fitting_wl, popt_best[i][1], popt_best[i][3], popt_best[i][5]), linestyle = '--', label = 'NII')
            if tau_best[i] < 0.1:
                plt.plot(fitting_wl, gaussian(fitting_wl, popt_best[i][0], popt_best[i][2], popt_best[i][4]), linestyle = '--', label = 'OI')
                plt.plot(fitting_wl, gaussian_plus_tophat(fitting_wl, *popt_best[i]), c = 'black')
            else:
                plt.plot(fitting_wl, escatt(fitting_wl, popt_best[i][0], popt_best[i][2], popt_best[i][4], tau_best[i]), linestyle = '--', label = 'OI')
                plt.plot(fitting_wl, escatt_plus_tophat_global(fitting_wl, *popt_best[i], tau_best[i]), c = 'black')
            plt.show()
            
            
    print('The opt taulist for tophat is', tau_best)
            
    return popt_best, pcov_best, best_score

    
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
    
    plt.plot(wl[normalisation_mask], flux[normalisation_mask])
    tot_cont_removed_flux = tot_continuum_remover(wl[normalisation_mask], flux[normalisation_mask], lambda_min, lambda_max)
    plt.plot(wl[normalisation_mask], tot_cont_removed_flux)
    plt.axhline(y = 0, linestyle = '--')
    plt.show()
    
    integrated_total_flux = integrate.cumtrapz(tot_cont_removed_flux, wl[normalisation_mask])[-1]
    
    if shape == 'g':
        integrated_NII_flux = integrate.cumtrapz(gaussian(wl, popt[1], popt[3], popt[5]), wl)[-1]
        uncertainty_NII = uncertainty_gaussian(popt[3], popt[5], pcov[3], pcov[5])
        
    elif shape == 'th':
        integrated_NII_flux = integrate.cumtrapz(tophat(wl, popt[1], popt[3], popt[5]), wl)[-1]
        uncertainty_NII = uncertainty_tophat(popt[3], popt[5], pcov[3], pcov[5])
        
    
    sigma = uncertainty_NII/integrated_total_flux
    return integrated_NII_flux/integrated_total_flux, sigma


