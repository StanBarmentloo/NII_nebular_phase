This folder contains all the files necessary for the Dardel cluster to allow for the user to run a modelling job with a single command.
Besides the files themselves, the paths in which to place them when on cluster are also mentioned 

At the location of this README (level 1), the following files + folders exist:

- runscript_S.py : This is the only file that needs to be run to complete a full job cycle. There are only four parameters that could ever require changing:
  
                   - masses : In this list, fill in the masses of the models that you want to run
                   - mixings : In this list, fill in the mixings that you want to run. Now, for each mass, all mixings will be run. The amount of runs started is thus len(masses) * len(mixings).
                               note that as of 14-06-2023, each run will claim a single node, so be careful with starting too many runs!
                   - t_eval : This gives the epoch at which to evaluate all the models
                   - initial_run : This boolean should be turned to True when it is the first t_eval for these specific models. If all models have at least one completed run (no matter at what t_eval),
                                   runs with only a different t_eval can use these as intial guesses to speed up the runs. In this case, put the bool to False.

                   once these parameters are set, be sure to adjust the file at $SUMODIR/DATA/ip_file_maker.py. The only thing to do is to put the correct t_eval. Once this is done,
                   simply do 'python runscript_S.py'. The runs will now start. This process goes as follows:
  
                   1 Create a separate directory for a run
                   2 Run the dirmake.sh file, which copies all necessary files for the run to the separate directory that was just created
                   3 Run the ip_file_maker, which creates the ip_file for this specific run's parameters
                   4 Run the slurm_file_maker, which creates the slurm_file for this specific run's parameters
                   5 Sbatch the slurm_file, actually requesting the cpu_time for the run and hopefully starting it immediately (as long as there is no queue)
  
- slurm_file_maker.py : see above
- ip_file_maker.py : see above
- ip_template : Used in the ip_file_maker as a template, in which only the specific runs parameters need to be changed
- slurm_template_pdc : Used in slurm_file_maker as a template, in which only the specific runs parameters need to be changed. This template is specifically for the PDC-BUS-2022-2 cluster (default)
- slurm_template_snic : Used in slurm_file_maker as a template, in which only the specific runs parameters need to be changed. This template is specifically for the SNIC cluster (currently not available)
- dirmake.sh : This file is already present in SUMO, but requires some adjustments for our specific goals. Simply replace the 'dirmake.sh' in your SUMO installation with the file in this repo, 
               and things should be fine!
- iterate_mpirun.sh : This file is already present in SUMO, but requires some adjustments for our specific goals. Simply replace the 'iterate_mpirun.sh' in your SUMO installation with the file in this repo, 
                      and things should be fine!

This README was last updated on: 14-06-2023
