import numpy as np
import matplotlib.pyplot as plt

model_masses = [3.3, 4.0, 6.0, 8.0]

#First interprolate between two models at a time
def get_interpolated_line(epochs, model_tracks, model_masses, m_guess):
    
    if m_guess <= model_masses[1]:
        m1, m2 = model_masses[0], model_masses[1]
        track1, track2 = np.copy(model_tracks[0]), np.copy(model_tracks[1])
    elif m_guess > model_masses[1] and m_guess < model_masses[2]:
        m1, m2 = model_masses[1], model_masses[2]
        track1, track2 = np.copy(model_tracks[1]), np.copy(model_tracks[2])
    elif m_guess >= model_masses[2]:
        m1, m2 = model_masses[2], model_masses[3]
        track1, track2 = np.copy(model_tracks[2]), np.copy(model_tracks[3])
        
    y1, y2 = track_epoch_adjuster(epochs, track1, track2)
    
    m_dist = m2-m1
    m_dist_m1 = m_guess-m1
    m_factor = m_dist_m1/m_dist
    
    y_interp = np.zeros(len(y1))
    for i in range(len(y1)):
        y_diff = y2[i]-y1[i]
        y_new = y1[i] + y_diff*m_factor
        y_interp[i] = y_new
        
    return y_interp

def track_epoch_adjuster(epochs, track1, track2):
    
    model_epochs, model_y1, model_y2 = track1[:, 0], track1[:, 1], track2[:, 1]
    y1_final, y2_final = np.zeros(len(epochs)), np.zeros(len(epochs))
    
    for i in range(epochs.shape[0]):
        y1_new = np.interp(epochs[i], model_epochs, model_y1)
        y2_new = np.interp(epochs[i], model_epochs, model_y2)
        
        y1_final[i], y2_final[i] = y1_new, y2_new
        
    return y1_final, y2_final

def mass_determinator_mcmc(epochs, NII_flux, NII_sig, model_tracks, plot = False):
    
    n_mc = 2500
    test_masses = np.arange(3.3, 8.01, 0.1)
    best_masses = np.zeros(n_mc)
    
    
    for q in range(n_mc):
        
        track1, track2 = np.copy(model_tracks[0]), np.copy(model_tracks[1])
        track3, track4 = np.copy(model_tracks[2]), np.copy(model_tracks[3])
        local_NII_flux, local_NII_sig = np.copy(NII_flux), np.copy(NII_sig)
        local_scores = np.zeros(len(test_masses))
        #Perturb the models
        for j in range(len(track1[:, 0])):
            track1[j, 1] += np.random.normal(loc = 0, scale = track1[j, 2], size = 1)
            track2[j, 1] += np.random.normal(loc = 0, scale = track2[j, 2], size = 1)
            track3[j, 1] += np.random.normal(loc = 0, scale = track3[j, 2], size = 1)
            track4[j, 1] += np.random.normal(loc = 0, scale = track3[j, 2], size = 1)
            
        perturbed_tracks = [track1, track2, track3, track4]
        
        #Perturb the observations
        for k in range(len(epochs)):
            local_NII_flux[k] += np.random.normal(loc = 0, scale = local_NII_sig[k])
            
        #Get the scores for these perturbed tracks + obs
        for p in range(len(test_masses)):
            m_guess = test_masses[p]
            score = mass_guess_scorer_mcmc(m_guess, epochs, local_NII_flux, local_NII_sig, perturbed_tracks)
            local_scores[p] = score
            
            
        guessed_mass = test_masses[np.argmin(local_scores)]
        best_masses[q] = guessed_mass
        
    mass_pde, bin_edges = np.histogram(best_masses, bins = test_masses)
    best_mass, mass_68_left, mass_68_right = minimal68(mass_pde, test_masses, True)
    
    if plot == True:    
        plt.hist(best_masses, bins = test_masses)
        plt.xlabel('Best Fit Mass (Msun)')
        plt.ylabel('Count')
        plt.show()
    
    print('Best fitting mass: ', '{0:.2f}'.format(best_mass), ' + ',
          '{0:.2f}'.format(mass_68_right-best_mass), ' - ', 
          '{0:.2f}'.format(best_mass-mass_68_left) )
    
    return best_mass, mass_68_left, mass_68_right
            
def mass_guess_scorer_mcmc(m_guess, epochs, NII_flux, NII_sig, perturbed_tracks):
    
    y_guess = get_interpolated_line(epochs, perturbed_tracks, model_masses, m_guess)
    
    score = np.sum((y_guess-NII_flux)**2)
        
    return score

def minimal68(prediction, local_labels, mcmc):
    centre_index = np.argmax(prediction)
    mle = local_labels[centre_index]

    total_prob = prediction[centre_index]/np.sum(prediction)
    left, right = np.argmax(prediction), np.argmax(prediction)
    
    while total_prob < 0.68:
        
        if left > 0 and right < len(prediction)-1:
            if prediction[left-1] > prediction[right+1]:
                total_prob += prediction[left-1]/np.sum(prediction)
                left -= 1
            elif prediction[left-1] < prediction[right+1]:
                total_prob += prediction[right+1]/np.sum(prediction)
                right += 1
            elif prediction[left-1] == prediction[right+1]:
                #This clause is entered when the prob on both sides is equal.
                #We then look at one further index for both sides to determine which way to go
                if left > 1 and right < len(prediction)-2:
                    if prediction[left-2] > prediction[right+2]:
                        total_prob += prediction[left-1]/np.sum(prediction)
                        left -= 1
                    elif prediction[left-2] < prediction[right+2]:
                        total_prob += prediction[right+1]/np.sum(prediction)
                        right += 1
                    elif prediction[left-2] == prediction[right+2]:
                        random_number = np.random.uniform(low = 0, high = 1, size = 1)
                        if random_number > 0.5:
                            right +=1
                        else:
                            left -=1
                        
                        
                elif left < 2:
                    total_prob += prediction[right+1]/np.sum(prediction)
                    right +=1
                elif right > len(prediction)-3:
                    total_prob += prediction[left-1]/np.sum(prediction)
                    left -=1
                
            
        elif left < 1:
            total_prob += prediction[right+1]/np.sum(prediction)
            right +=1
        elif right > len(prediction)-2:
            total_prob += prediction[left-1]/np.sum(prediction)
            left -=1
            
    left68, right68 = np.copy(left), np.copy(right)
    return mle+0.1, local_labels[left68]+0.1, local_labels[right68]+0.1














#=====================================================================================================================
#These are legacy functions, no longer in use


def mass_determinator(epochs, NII_flux, NII_sig, plot = False):
    
    test_masses = np.arange(3.3, 6.01, 0.1)
    scores_mid, scores_low, scores_high = np.zeros(len(test_masses)), np.zeros(len(test_masses)), np.zeros(len(test_masses))
    
    for q in range(len(test_masses)):
        m_guess = test_masses[q]
        score_mid, score_low, score_high = mass_guess_scorer(m_guess, epochs, NII_flux, NII_sig)
        scores_mid[q], scores_low[q], scores_high[q] = score_mid, score_low, score_high
    
    mass_mid = test_masses[np.argmin(scores_mid)]
    if score_method == 'test3':
        mass_low = test_masses[np.argmin(scores_low)]
        mass_high = test_masses[np.argmin(scores_high)]
    elif score_method == 'factor2':
        try:
            left_err_index = np.where(scores_mid[:np.argmin(scores_mid)] > 2*np.min(scores_mid))[0][-1]
        except:
            left_err_index = 0
        try:
            right_err_index = np.where(scores_mid[np.argmin(scores_mid):] > 2*np.min(scores_mid))[0][0] + np.argmin(scores_mid)
        except:
            right_err_index = -1
        
        mass_low = test_masses[left_err_index]
        mass_high = test_masses[right_err_index]
    
    if plot == True:
        #plt.plot(test_masses, scores_low, c = 'r')
        plt.plot(test_masses, scores_mid, c = 'orange')
        plt.axvline(x = mass_low)
        plt.axvline(x = mass_high)
        #plt.plot(test_masses, scores_high, c = 'green')
        plt.ylim(0, np.min(scores_mid)*2)
        plt.show()
        
    print('Best fitting mass: ', '{0:.2f}'.format(mass_mid), ' + ',
          '{0:.2f}'.format(mass_high-mass_mid), ' - ', 
          '{0:.2f}'.format(mass_mid-mass_low) )
    
    return mass_mid, mass_low, mass_high

def mass_guess_scorer(m_guess, epochs, NII_flux, NII_sig):
    
    y_guess_mid = get_interpolated_line(epochs, model_tracks, model_masses, m_guess, mode = 'mid')
    y_guess_low = get_interpolated_line(epochs, model_tracks, model_masses, m_guess, mode = 'low')
    y_guess_high = get_interpolated_line(epochs, model_tracks, model_masses, m_guess, mode = 'high')
    
    score_mid = np.sum((y_guess_mid-NII_flux)**2 / NII_sig**2)
    score_low = np.sum((y_guess_low-(NII_flux+NII_sig))**2  / NII_sig**2)
    score_high = np.sum((y_guess_high-(NII_flux-NII_sig))**2  / NII_sig**2)
        
    
    return score_mid, score_low, score_high


#Legacy part of get interpolated line
#if mode == 'mid':
#        pass
#    elif mode == 'low':
#        track1[:, 1] -= track1[:, 2]
#        track2[:, 1] -= track2[:, 2]
#    elif mode == 'high':
#        track1[:, 1] += track1[:, 2]
#        track2[:, 1] += track2[:, 2]
        
    #print(track1)
    #print(m_guess)
    #plt.plot(track1[:, 0], track1[:, 1])
    #plt.plot(track2[:, 0], track2[:, 1])
    #plt.plot(obs_track[:, 0], obs_track[:, 1])
    #plt.ylim(5, 14)
    #plt.show()