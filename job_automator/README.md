This folder contains all the files necessary for the Dardel cluster to allow for the user to run a modelling job with a single command.
Besides the files themselves, the paths in which to place them when on cluster are also mentioned 

At the location of this README (level 1), the following files + folders exist:

- slurm_file_maker.py : 
- ip_template :
- ip_file_maker.py :
- slurm_template_pdc :
- slurm_template_snic :
- dirmake.sh : This file is already present in SUMO, but requires some adjustments for our specific goals. Simply replace the 'dirmake.sh' in your SUMO installation with the file in this repo, 
               and things should be fine!
- iterate_mpirun.sh : This file is already present in SUMO, but requires some adjustments for our specific goals. Simply replace the 'iterate_mpirun.sh' in your SUMO installation with the file in this repo, 
                      and things should be fine!

This README was last updated on: 12-06-2023
