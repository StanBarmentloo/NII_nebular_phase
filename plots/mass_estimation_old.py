import numpy as np
import matplotlib.pyplot as plt

model_masses = [3.3, 4.0, 5.0, 6.0, 8.0]

#First interprolate between two models at a time
def get_interpolated_line(epochs, model_tracks, model_masses, m_guess):
    
    if m_guess <= model_masses[1]:
        m1, m2 = model_masses[0], model_masses[1]
        track1, track2 = np.copy(model_tracks[0]), np.copy(model_tracks[1])
    elif m_guess > model_masses[1] and m_guess < model_masses[2]:
        m1, m2 = model_masses[1], model_masses[2]
        track1, track2 = np.copy(model_tracks[1]), np.copy(model_tracks[2])
    elif m_guess > model_masses[2] and m_guess < model_masses[3]:
        m1, m2 = model_masses[2], model_masses[3]
        track1, track2 = np.copy(model_tracks[2]), np.copy(model_tracks[3])
    elif m_guess >= model_masses[3]:
        m1, m2 = model_masses[3], model_masses[4]
        track1, track2 = np.copy(model_tracks[3]), np.copy(model_tracks[4])
        
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
    global_scores = np.zeros(len(test_masses))
    
    
    for q in range(n_mc):
        
        track1, track2 = np.copy(model_tracks[0]), np.copy(model_tracks[1])
        track3, track4 = np.copy(model_tracks[2]), np.copy(model_tracks[3])
        track5 = np.copy(model_tracks[4])
        local_NII_flux, local_NII_sig = np.copy(NII_flux), np.copy(NII_sig)
        local_scores = np.zeros(len(test_masses))
        #Perturb the models
        for j in range(len(track1[:, 0])):
            track1[j, 1] += np.random.normal(loc = 0, scale = track1[j, 2], size = 1)
            track2[j, 1] += np.random.normal(loc = 0, scale = track2[j, 2], size = 1)
            track3[j, 1] += np.random.normal(loc = 0, scale = track3[j, 2], size = 1)
            track4[j, 1] += np.random.normal(loc = 0, scale = track4[j, 2], size = 1)
            track5[j, 1] += np.random.normal(loc = 0, scale = track5[j, 2], size = 1)
            
        perturbed_tracks = [track1, track2, track3, track4, track5]
        
        #Perturb the observations
        for k in range(len(epochs)):
            local_NII_flux[k] += np.random.normal(loc = 0, scale = local_NII_sig[k])
            local_NII_flux[k] = np.max((local_NII_flux[k], 0)) #Negative values are non physical
            
        #Get the scores for these perturbed tracks + obs
        for p in range(len(test_masses)):
            m_guess = test_masses[p]
            score = mass_guess_scorer_mcmc(m_guess, epochs, local_NII_flux, local_NII_sig, perturbed_tracks)
            local_scores[p] = score
            

            
        guessed_mass = test_masses[np.argmin(local_scores)]
        
        #plt.scatter(epochs, local_NII_flux, c = 'r')
        #plt.plot(perturbed_tracks[0][:, 0], perturbed_tracks[0][:, 1])
        #plt.plot(perturbed_tracks[1][:, 0], perturbed_tracks[1][:, 1])
        #plt.plot(perturbed_tracks[2][:, 0], perturbed_tracks[2][:, 1])
        #plt.plot(perturbed_tracks[3][:, 0], perturbed_tracks[3][:, 1])
        #plt.plot(perturbed_tracks[4][:, 0], perturbed_tracks[4][:, 1])
        #plt.xlim(100, 450)
        #plt.ylim(0, 20)
        #plt.show()
        
        #plt.plot(test_masses, local_scores)
        #plt.show()
        #print('Above we guess: ', guessed_mass)
        
        best_masses[q] = guessed_mass
        global_scores[:] += local_scores
                             
    print('The final score plot: ')
    plt.plot(test_masses, global_scores)
    plt.axhline(y = np.min(global_scores), linestyle = '--')
    plt.axhline(y = 2*np.min(global_scores), linestyle = '--', c = 'r')
    plt.yscale('log')
    plt.show()
        
    mass_pde, bin_edges = np.histogram(best_masses, bins = np.arange(3.29,8.1,0.1)) #Different bins than test masses to make the boundaries clearer for the histogram. Settings bins = test_masses gives highest possible mass as 7.9 instead of 8.0
    best_mass, mass_68_left, mass_68_right = minimal68(mass_pde, test_masses, True)
    
    if plot == True:    
        plt.hist(best_masses, bins = bin_edges)
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
    return mle, local_labels[left68], local_labels[right68]


#Below are some functions used to convert between different masses

def Mzams_to_Mhe(Mzams):
    #These formulae are based on Woosley et al., 2019 equations 4 and 5
    try:
        if float(Mzams) > 30:
            return float('{0:.1f}'.format(0.5*float(Mzams) - 5.87))
        else:
            return float('{0:.1f}'.format(0.0385*float(Mzams)**1.603))
    except ValueError:
        return -3
    
def Mhe_i_to_Mhe_f(Mhe_i):
    
    Mhe_i_ertl = np.array([3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 
                          4.2, 4.3, 4.4, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0,
                          10, 11, 12, 14])
    Mhe_f_ertl = np.array([2.67, 2.74, 2.81, 2.88, 2.95, 3.02, 3.09, 3.15, 3.22, 3.29, 
                          3.36, 3.42, 3.49, 3.82, 4.14, 4.45, 4.75, 5.05, 5.35, 5.64, 6.20,
                          6.75, 7.05, 7.27, 8.04])
    
    if Mhe_i in Mhe_i_ertl:
        return Mhe_f_ertl[ np.where(Mhe_i == Mhe_i_ertl)[0][0]]
    
    else:
        closest_index = 0
        for i in range(len(Mhe_i_ertl)-1):
            if Mhe_i_ertl[i] < Mhe_i and Mhe_i_ertl[i+1] > Mhe_i:
                closest_index = i
                
        rel_distance_lowest_mass = (Mhe_i-Mhe_i_ertl[closest_index]) / (Mhe_i_ertl[closest_index+1]-Mhe_i_ertl[closest_index])
        return rel_distance_lowest_mass * (Mhe_f_ertl[closest_index+1]-Mhe_f_ertl[closest_index]) + Mhe_f_ertl[closest_index]
    
def Mhe_f_to_Mej(Mhe_f):
    
    Mhe_f_ertl = np.array([2.67, 2.81, 3.15, 3.49, 3.82, 4.45, 5.05, 5.64, 6.19, 6.75,
                           7.05, 7.27, 8.04])
    Mej_f_ertl = np.array([1.20, 1.27, 1.62, 1.89, 2.21, 2.82, 3.33, 3.95, 4.45, 5.19,
                           5.21, 5.32, 6.3]) #6.3 is inteprolated from Ertl paper
    
    if Mhe_f in Mhe_f_ertl:
        return Mej_f_ertl[ np.where(Mhe_f == Mhe_f_ertl)[0][0]]
    
    else:
        closest_index = 0
        for i in range(len(Mhe_f_ertl)-1):
            if Mhe_f_ertl[i] < Mhe_f and Mhe_f_ertl[i+1] > Mhe_f:
                closest_index = i
                
        rel_distance_lowest_mass = (Mhe_f-Mhe_f_ertl[closest_index]) / (Mhe_f_ertl[closest_index+1]-Mhe_f_ertl[closest_index])
        return rel_distance_lowest_mass * (Mej_f_ertl[closest_index+1]-Mej_f_ertl[closest_index]) + Mej_f_ertl[closest_index]


def Mej_to_Mhe_f(Mej):
    
    Mhe_f_ertl = np.array([2.67, 2.81, 3.15, 3.49, 3.82, 4.45, 5.05, 5.64, 6.19, 6.75,
                           7.05, 7.27, 8.04])
    Mej_f_ertl = np.array([1.20, 1.27, 1.62, 1.89, 2.21, 2.82, 3.33, 3.95, 4.45, 5.19,
                           5.21, 5.32, 6.3]) #6.3 is inteprolated from Ertl paper
    
    
    if Mej in Mej_f_ertl:
        return Mhe_f_ertl[ np.where(Mej == Mej_f_ertl)[0][0]]
    
    else:
        closest_index = 0
        for i in range(len(Mej_f_ertl)-1):
            if Mej_f_ertl[i] < Mej and Mej_f_ertl[i+1] > Mej:
                closest_index = i
            if i == len(Mej_f_ertl)-2 and closest_index == 0: #The ejecta mass is out of range, so we simply return the biggest value
                if Mej < Mej_f_ertl[0]:
                    return Mhe_f_ertl[0]
                else:
                    return Mhe_f_ertl[-1]

        rel_distance_lowest_mass = (Mej-Mej_f_ertl[closest_index]) / (Mej_f_ertl[closest_index+1]-Mej_f_ertl[closest_index])
        return rel_distance_lowest_mass * (Mhe_f_ertl[closest_index+1]-Mhe_f_ertl[closest_index]) + Mhe_f_ertl[closest_index]



