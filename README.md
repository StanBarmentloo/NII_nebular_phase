# NII_nebular_phase
This Repository contains all the files necessary to reproduce the work for my PhD's first paper.
Each folder in the repository will contain a short README on the purpose of the files + folders in thhat specific folder.

At the location of this README (level 0), the following files + folders exist:

- job_automator : This folder contains all the files necessary for the Dardel cluster to allow for the user to run a modelling job with a single command.
- data_creation : This folder contains notebooks to turn the raw, Helium-star progenitor data from Woosley et al. 2019 into files that can be read by SUMO.
- SUMO_results : Here, all the output spectra from SUMO that are used in the analysis are stored.
- analysis: Here, all the analysis notebooks are stored that allow the user to get insight into the spectra that were produced.
- plots: In this folder, notebooks for all plots that are used in the paper are stored. Each figure has its own, dedicated notebook.

This README was last updated on: 12-06-2023



## To do list
Here I write all the things that still need to be done to this repo, in somewhat reasonable order

- Update README's
- Upload the files promised in the README. Make a checklist in the local README for files that still need pylint + black checks.
- Make a description for the different model versions in the 'data_creation/expl_models' folder
- Make sure that the file paths are compatible with eachother

Finally some wishes:
- Merge header_maker.py into add_funcs.py
- Have a look at turning everything into .py
- Have a look at getting wrappers for the most important scripts
