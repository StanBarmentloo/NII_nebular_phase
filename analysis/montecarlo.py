import random
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

usesize = 12
plt.rcParams.update({'font.size': usesize})

N = 1000000 # Number of packets in simulation 1E5 default

type = 'linescattering' # 'electronscattering' or 'linescattering'

save = 1

colorvec=['b','r','k']


if (type == 'electronscattering'):
    Ntau = 5
    tau0vec = [0.,1.,1.5, 2.,3.]  # For continous simulations (elecronscattering)
    peakval = [0., 0., 0., 0.,0.]
    peakind = [0, 0, 0, 0, 0]
    xpeak = [0., 0., 0., 0., 0]
elif (type == 'linescattering'):
    Ntau = 3
    tau0vec = [0.,100., 100., 0.,100.,100.]
    peakval = [0.,0.,0.,0.,0.,0.]
    peakind = [0.,0.,0.,0.,0.,0]
    xpeak = [0.,0.,0.,0.,0.,0]
    #tau0vec = [1.,1.,1.,1.,1.,1.] # For line simulation
    vthermal = [0.,0.,0.,0.,0.,0] # For line simulation
    #wllines = np.array([1001.,1002.,1003.x,1004.,1005.,1006.,1007.,1008.,1009.,1010])
    wllines = np.zeros(shape=(6,16))
    #wllines[0,:] = np.zeros(shape=(19))
    wllines[0,:] = [1005.]
    wllines[1,:] = [1005.]
    wllines[2,:] = [1005.]
    wllines[3,:] = [1005.]
    wllines[4,:] = [1005.]
    wllines[5,:] = [1005.]
    #wllines[0,:] = np.arange(1003., 1050.,3.)
    #wllines[1,:] = np.arange(1003., 1050.,3.)
    #wllines[2,:] = np.arange(1003., 1050.,3.)
    #wllines[3,:] = np.arange(1010., 1170.,10.)
    #$wllines[4,:] = np.arange(1010., 1170.,10.)
    #wllines[5,:] = np.arange(1010., 1170.,10.)
    #wllines[1,:] = [1005., 3000.]s5
    #wllines[2,:] = [1001., 3000.]

    fluorprob = [1., 1., 0, 1., 1., 0.] # For line simulation

v_out = 3000
c = 3E5
lambda0 = 1000.


wlmin = 1000*(1-v_out/c)
wlmax = 1000*(1+4*v_out/c)
Nbin = 200

binarr = np.linspace(wlmin,wlmax,Nbin)

hist = np.zeros(Nbin)


for j in range(0,Ntau):

    if (type == 'electronscattering'):
        
        tau0 = tau0vec[j]    

    hist = np.zeros(Nbin)

    for i in range(0,N):

        wl = lambda0

        x = random.random()

        #if (x < 0.5 and j <= 2):
        #    wl = lambda0 + 5 # TEMP for simulating doublet

        x = random.random()

        r = x**(1./3)

        mu = 1.-2*random.random()

        escaped = False

        destroyed = False

        while (escaped == False and destroyed == False):

            distedge = -r*mu + np.sqrt((r*mu)**2 - (r**2 - 1**2))
            
            z = random.random()

            tau_thisphoton = -np.log(1-z) # TEMP righ log?

            distscatt = 10

            if (type == 'electronscattering'):

                #print "tau", tau_thisphoton

                distscatt = tau_thisphoton/tau0

            elif (type == 'linescattering'):

                ind = (wllines[j,:] > wl)

                #print "ind", wllines[ind][0], wllines[len(wllines)-1], wl

                if (wllines[j,len(wllines[j,:])-1] > wl):

                    distscatt = (wllines[j,ind][0]-wl)/wl*c/v_out

                    #print "distscatt", distscatt

                else:

                    #print "ok"

                    distscatt = 10.

            sin_theta = np.sqrt(1.-mu**2)

            #print "In", r, mu, distedge, wl

            if (distscatt < distedge):

                #print "OK"

                dr = distscatt
                
                r_nextpos = np.sqrt(r**2 + dr**2 - 2*r*dr*(-mu))

                #print "r", r, r_nextpos, mu

                gamma = 1./np.sqrt(1.-(dr*v_out/c)**2)

                wl = wl*(1 + v_out/c*dr)*gamma    # Rybicki Lightman eq 4.11 with cos_theta = -1 (homologous flow)

                #x = random.random()

                #dwl_thermal = wl*vthermal[j]/c*(1-2*x)

                #wl = wl + dwl_thermal

                if (type == 'electronscattering'):

                    mu = 1 - 2*random.random()

                else:

                    x = random.random()

                    if (x < 1. - np.exp(-tau0vec[j])): # optical depth of each line TEMP

                        mu = 1. - 2*random.random()  # Isotropic scattering

                        x = random.random()

                        if (x < fluorprob[j]):

                            destroyed = True # TEMP
                    
                            #print "scattered", wl, r_nextpos

                    else:

                        sin_theta = r/r_nextpos*sin_theta

                        if (mu < 0 and dr < abs(r*mu)):
                    
                            mu = -np.sqrt(1.-sin_theta**2)

                        else:
                    
                            mu = np.sqrt(1.-sin_theta**2)

                            #    #print "noscatter", wl, r_nextpos

                    r = r_nextpos                    

            else:

                dr = distedge

                #print "distedge", distedge, distscatt, wl

                # Comoving frame properties at edge

                r_nextpos = 1.

                sin_theta = r/r_nextpos*sin_theta

                mu = np.sqrt(1.-sin_theta**2) # mu most be positive reaching surface

                gamma = 1./np.sqrt(1.-(dr*v_out/c)**2)

                wl =  wl*(1 + v_out/c*dr)*gamma

                # In observer frame

                gamma = 1./np.sqrt(1.-(v_out/c)**2)  

                wl_escaped = wl*(1. - v_out/c*mu)*gamma  # RybickiLightman eq 4.11

                #print "escaped", wl, mu, wl_escaped

                
                # Bin it

                new = (binarr - wl_escaped)**2

                #print new
        
                ind = new.argmin()

                #print ind

                #print "Out", r_nextpos, mu, wl, wl_escaped, ind

                #print ind

                hist[ind] = hist[ind] + 1

                escaped = True

    ynorm = 0.077

    gauss_kernel = Gaussian1DKernel(2)
    hist = convolve(hist, gauss_kernel) #

    if (j <= 2):

        plt.plot((binarr-1000)/1000/(v_out/c),hist/N/ynorm*Nbin/50,color=colorvec[j]) # TEMP

    elif (j == 3):

        plt.plot((binarr-1000)/1000/(v_out/c),hist/N/ynorm*Nbin/50/2,'--',color=colorvec[j-3])

    elif (j == 4):

        plt.plot((binarr-1000)/1000/(v_out/c),hist/N/ynorm*Nbin/50/2,':',color=colorvec[j-3])

    elif (j == 5):

        plt.plot((binarr-1000)/1000/(v_out/c),hist/N/ynorm*Nbin/50/2,'-.',color=colorvec[j-3])

    fidsave = open("mcsim_"+str(tau0vec[j]),'w')
    np.savetxt(fidsave,np.c_[(binarr-1000)/1000/(v_out/c),hist/N/ynorm*Nbin/50/2])
    fidsave.close()
        

    #print(hist[0:int(Nbin/2)])
    peakval[j] = max(hist[0:int(Nbin/2)]/N/ynorm*Nbin/50)
    peakind[j] = np.argmax(hist[0:int(Nbin/2)]/N/ynorm*Nbin/50)
    xpeak[j] =  (binarr[peakind[j]]-1000.)/1000./(v_out/c)
    print(j, peakval[j], peakind[j], (binarr[peakind[j]]-1000.)/1000./(v_out/c))

    #plt.plot((binarr-1000)/1000/(v_out/c), 0.06*(binarr[2]-binarr[1])/(40/50.)*(1-((binarr-1000)/10)**2))

plt.ylim([0,1])

plt.xlim([-1,3])



if (type == 'electronscattering'):
    plt.plot([0.48,0.55],[0.06/ynorm,0.065/ynorm],'k')
    plt.text(0.56,0.066/ynorm,r'$\tau=0$')
    plt.plot([0.64,0.75],[0.035/ynorm,0.045/ynorm],'k')
    plt.text(0.8,0.046/ynorm,r'$\tau=1$')
    plt.plot([0.71,0.78],[0.028/ynorm,0.035/ynorm],'k')
    plt.text(0.82,0.036/ynorm,r'$\tau=2$')
    plt.plot([0.73,0.8],[0.024/ynorm,0.030/ynorm],'k')
    plt.text(0.8,0.031/ynorm,r'$\tau=3$')
else:
    plt.text(0.01,0.95,'Optically thin')
    plt.text(0.9,0.8,'Optically thick,')
    plt.text(0.9,0.75,'scatters')
    plt.text(0.01,0.6,'Optically thick,')
    plt.text(0.01,0.55,'destroyed')
#plt.plot([xpeak[1],xpeak[1]],[peakval[1],peakval[1]+0.02],'k')
#plt.plot([xpeak[2],xpeak[2]],[peakval[2],peakval[2]+0.02],'k')
#plt.plot([xpeak[3],xpeak[3]],[peakval[3],peakval[3]+0.02],'k')
if (type == 'electronscattering'):
    plt.plot([-0.13,-0.13],[peakval[1],peakval[1]+0.02],'k')
    plt.plot([-0.20,-0.20],[peakval[2],peakval[2]+0.02],'k')
    plt.plot([-0.27,-0.27],[peakval[3],peakval[3]+0.02],'k')

    plt.text(xpeak[1]-0.1,peakval[1]+0.03,'-0.13')
    plt.text(xpeak[2]-0.1,peakval[2]+0.03,'-0.20')
    plt.text(xpeak[3]-0.1,peakval[3]+0.03,'-0.27')



plt.plot([0,0],[0,1],'k--')
plt.plot([0.5,0.5],[0,1],'k--')

plt.xlabel('Wavelength ($\Delta \lambda/\lambda_0/(V_{max}/c)$)')
plt.ylabel('Flux')

if (save == 0):
    plt.show()
else:
    if (type == 'electronscattering'):
        plt.savefig("electronscattering.eps",bbox_inches='tight')
    else:
        plt.xlim([-1,1.5])
        plt.savefig("linescattering.eps",bbox_inches='tight')
        
 
