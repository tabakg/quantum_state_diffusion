#!/usr/bin/python env

'''
Run Markov model: Submit jobs on SLURM
'''

import os
import logging
import sys
sys.path.append('/scratch/users/tabakg/qsd_dev')

from utils import get_params_json
from utils import load_trajectory

diffusion_maps_folder='/scratch/users/tabakg/qsd_output/diffusion_maps_fast_euler/diffusion_maps_data'
output_dir='/scratch/users/tabakg/qsd_output/markov_model_data_fast_euler'

# Variables for each job
memory = 4000
partition = 'normal'

# Create subdirectories for job, error, and output files
job_dir = "%s/.job" %(output_dir)
out_dir = "%s/.out" %(output_dir)
for new_dir in [output_dir,job_dir,out_dir]:
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

OVERWRITE=True

files = [f for f in os.listdir(diffusion_maps_folder) if f[-3:] == 'pkl']

## Used in generating the HMM, e.g. for the initial distributions
## NOTE: if the trajectory fails, the seed value may be incremented
## See markov_model.py.
seed = 1
n_clusters_list = [10,15,20,25,30,35]
n_iter_list = [10,20,30,40]

for n_clusters in n_clusters_list:
    for n_iter in n_iter_list:
        for diffusion_maps_file in files:
            diffusion_maps_path = os.path.join(diffusion_maps_folder, diffusion_maps_file)

            ## For the time being, we are only interested in reducing one system and feeding it into another.
            ## For this reason, if the input system was actually 2 systems, we just skip it.
            num_systems = get_params_json(load_trajectory(diffusion_maps_path)['traj_list'][0])['num_systems']
            if num_systems == 2:
                continue

            hash_name = diffusion_maps_file[len('diffusion_maps_')-1:][:-len('.pkl')]
            name = 'markov_model_' + hash_name + "_" + str(seed) + "_" + str(n_clusters) + "_" + str(n_iter) + ".pkl"
            output_file_path = os.path.join(output_dir, name)
            file_exists = os.path.isfile(output_file_path)
            if OVERWRITE or not file_exists:
                print("OVERWRITE is %s and file %s existence is %s" %(OVERWRITE, output_file_path, file_exists))
                print("If overwriting or file does not exist, going to process new seed.")
                print ("Processing seed %s" %(seed))
                # Write job to file
                filey = "%s/qsd_%s.job" %(job_dir, seed)
                filey = open(filey,"w")
                filey.writelines("#!/bin/bash\n")
                filey.writelines("#SBATCH --job-name=markov_%s\n" %(seed))
                filey.writelines("#SBATCH --output=%s/markov_model_seed_%s_diffusion_map_%s.out\n" %(out_dir,seed,hash_name))
                filey.writelines("#SBATCH --error=%s/markov_model_seed_%s_diffusion_map_%s.err\n" %(out_dir,seed,hash_name))
                filey.writelines("#SBATCH --time=2-00:00\n")
                filey.writelines("#SBATCH --mem=%s\n" %(memory))
                filey.writelines("python /scratch/users/tabakg/qsd_dev/markov_model.py --seed %s --n_clusters %s --n_iter %s --input_file_path '%s' --output_file_path '%s'" % (seed, n_clusters, n_iter, diffusion_maps_path, output_file_path))
                filey.close()
                os.system("sbatch -p %s %s/qsd_%s.job" %(partition,job_dir,seed))
