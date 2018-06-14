#!/usr/bin/python env

'''
Run Markov model: Submit jobs on SLURM
'''

import os
import logging

diffusion_maps_folder='/scratch/users/tabakg/qsd_output/diffusion_maps/diffusion_maps_data'
output_dir='/scratch/users/tabakg/qsd_output/markov_model_data'

# Variables for each job
memory = 16000
partition = 'normal'

# Create subdirectories for job, error, and output files
job_dir = "%s/.job" %(output_dir)
out_dir = "%s/.out" %(output_dir)
for new_dir in [output_dir,job_dir,out_dir]:
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

OVERWRITE=False

files = [f for f in os.listdir(diffusion_maps_folder) if f[-3:] == 'pkl']

## Used in generating the HMM, e.g. for the initial distributions
## NOTE: if the trajectory fails, the seed value may be incremented
## See markov_model.py.
seed = 1

for diffusion_maps_file in files:
    diffusion_maps_path = os.path.join(diffusion_maps_folder, diffusion_maps_file)
    hash_name = diffusion_maps_file[len('diffusion_maps_')-1:][:-len('.pkl')]
    name = 'markov_model_' + hash_name + ".pkl"
    output_file_path = os.path.join(output_dir, name)
    file_exists = os.path.isfile(output_file_path)
    if not OVERWRITE and not file_exists:
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
        filey.writelines("python /scratch/users/tabakg/qsd_dev/markov_model.py --input_file_path '%s' --output_file_path '%s'" % (diffusion_maps_path, output_file_path))
        filey.close()
        os.system("sbatch -p %s %s/qsd_%s.job" %(partition,job_dir,seed))
