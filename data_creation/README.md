# Data Creation

This folder contains all the files necessary for creating the explosion model data from the Woosley et al. 2019 stellar models.

At the location of this README, the following files + folders exist:

- woosley_data_converter_AJ_2015_HeC.ipynb : This notebook is the translator from the Woosley data to something that can be understood by SUMO.
                                                     Taking the files from 'woosley_data' as input, its outputs are stored in 'explosion_models'
- header_maker.py : This is a purely auxillary file, used in the 'converter' notebook. You should in no case have to look at it!
- woosley_data : This folder contains all the output_files for the Helium star models from Woosley et al. 2019, kindly provided by the Garching group
- explosion_models: This folder contains all the explosion models created using the 'converter' notebook, to be feeded in to SUMO. 

This README was last updated on: 27-03-2024


