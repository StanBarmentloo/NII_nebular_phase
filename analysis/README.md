This folder contains all the files necessary for the analysis of the resulting spectrum as presented in 'SUMO_results'

At the location of this README (level 1), the following files + folders exist:

- add_funcs.py : A python file containing a plethora of auxilary functions used during analysis. Think of definitions for a Gaussian, or routines to translate wavelengths to velocities and much more.
- spectrum_plotter.ipynb : This notebook simply allows the user to nicely plot any spectrum (modelled or observed), provided the spectrum file is in the correct format
- NII_time_tracks.ipynb : This notebook plots out the NII-statistic's evolution over time for each SN (modelled or observed), also known as an NII_time_track.
                          It stores all the time tracks for the observed SNe in a .csv file
- NII_in_observed_spectra.ipynb : This notebook compares the model spectra to two well observed SNe, 2007Y and 2011dh
- Diagnostic_usage.ipynb : This notebook uses the NII-statistic for each SN and turns it into a predicted M_He and M_ZAMS mass.
                           Secondly, it compares the predicted masses with the masses available in the SECRETO repository (the data of which is provided in a .csv file)
  


This README was last updated on: 14-06-2023
