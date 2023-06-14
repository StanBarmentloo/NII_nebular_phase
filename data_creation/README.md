This folder contains all the files necessary for the Dardel cluster to allow for the user to run a modelling job with a single command.
Besides the files themselves, the paths in which to place them when on cluster are also mentioned 

At the location of this README (level 1), the following files + folders exist:

- woosley_data_converter_AJ_2015_mixing_v1p0.ipynb : This notebook is the translator from the Woosley data to something that can be understood by SUMO.
                                                     Taking the files from 'woosley_data' as input, its outputs are stored in 'explosion_models'
- header_maker.py :
- woosley_data : This folder contains all the output_files for the Helium star models from Woosley et al. 2019, kindly provided by the Garching group
- explosion_models: This folder contains all the explosion models created using the 'converter' notebook, to be feeded in to SUMO. 
                    Some previous iterations of the models are also present.

This README was last updated on: 14-06-2023
