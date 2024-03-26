# NII_nebular_phase
This Repository contains all the files necessary to reproduce the work for my PhD's first paper, titled:

Nebular Nitrogen Line Emission in Stripped-Envelope Supernovae -- a New Progenitor Mass Diagnostic

Each folder in the repository will contain a short README on the purpose of that specific folder.

**IMPORTANT: The observational part of this study has its own repository, at https://github.com/StanBarmentloo/SECRETO . Some of the results from that repository are used to create the figures present in this repository, so that to fully reproduce the work, both repo's are needed. **

At the location of this README, the following files + folders exist:

- plots: This is by far the most relevant folder. In this folder, notebooks for all plots that are used in the paper are stored. Each figure has its own, dedicated notebook.
- data_creation : This folder contains notebooks to turn the raw, Helium-star progenitor data from Woosley et al. 2019 into files that can be read by SUMO.
- SUMO_results : Here, all the output spectra from SUMO that are used in the analysis are stored.
- prerequisites: Here, some notebooks that create data used in the figures are stored.
- job_automator : This folder is only relevant for those at Stockholm University. It contains all the files necessary for our private cluster to allow for the user to run a modelling job with a single command.


This README was last updated on: 26-03-2024
