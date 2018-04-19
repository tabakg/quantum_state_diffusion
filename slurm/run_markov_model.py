#!/usr/bin/python env

'''
Run Markov model: Submit jobs on SLURM
'''

import os

diffusion_maps_folder='/scratch/users/tabakg/qsd_output/diffusion_maps/diffusion_maps_data'
output_dir='/scratch/users/tabakg/qsd_output/diffusion_maps/markov_model_data'

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

for diffusion_maps_path in files:

    _, diffusion_maps_file = os.path.split(diffusion_maps_path)
    hash_name = diffusion_maps_file.lstrip('diffusion_maps_').rstrip('.pkl')
    name = 'markov_model_' + hash_name
    output_file_path = os.path.join(output_dir, name)
    file_exists = os.path.isfile(output_file_path)

    print("OVERWRITE is %s and file %s existence is %s" %(OVERWRITE, output_file_path, file_exists))
    print("If overwriting or file does not exist, going to process new seed.")

    if OVERWRITE or not file_exists:
        print "Processing seed %s" %(seed)
        # Write job to file
        filey = ".job/qsd_%s.job" %(seed)
        filey = open(filey,"w")
        filey.writelines("#!/bin/bash\n")
        filey.writelines("#SBATCH --job-name=making_%s\n" %(output_file_path))
        filey.writelines("#SBATCH --output=%s/markov_model_seed%s.out\n" %(out_dir,seed))
        filey.writelines("#SBATCH --error=%s/markov_model_seed%s.err\n" %(out_dir,seed))
        filey.writelines("#SBATCH --time=2-00:00\n")
        filey.writelines("#SBATCH --mem=%s\n" %(memory))
        filey.close()
        os.system("sbatch -p %s .job/qsd_%s.job" %(partition,seed))
