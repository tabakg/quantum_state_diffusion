#!/usr/bin/python env

'''
Run the hybrid_qsd.py code using recoded Markov models for system 1
and input parameters for system 2.
'''

import os

markov_model_folder='/scratch/users/tabakg/qsd_output/markov_model_data'
output_dir='/scratch/users/tabakg/qsd_output/hybrid_trajectory_data'

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

files = [f for f in os.listdir(markov_model_folder) if f[-3:] == 'pkl']

# Seed for generating trajectory of system 1 and stochastic terms of system 2.
seed = 101

for markov_model_file in files:
    markov_model_path = os.path.join(markov_model_folder, markov_model_file)
    hash_name = markov_model_file[len('markov_model_')-1:len('.pkl')]
    name = 'hybrid_QSD_' + hash_name
    output_file_path = os.path.join(output_dir, name)
    file_exists = os.path.isfile(output_file_path)

    print("OVERWRITE is %s and file %s existence is %s" %(OVERWRITE, output_file_path, file_exists))
    print("If overwriting or file does not exist, going to process new seed.")

    if OVERWRITE or not file_exists:
        print "Processing seed %s" %(seed)
        # Write job to file
        filey = "%s/hybrid_%s.job" %(job_dir, seed)
        filey = open(filey,"w")
        filey.writelines("#!/bin/bash\n")
        filey.writelines("#SBATCH --job-name=making_%s\n" %(output_file_path))
        filey.writelines("#SBATCH --output=%s/markov_model_seed%s.out\n" %(out_dir,seed))
        filey.writelines("#SBATCH --error=%s/markov_model_seed%s.err\n" %(out_dir,seed))
        filey.writelines("#SBATCH --time=2-00:00\n")
        filey.writelines("#SBATCH --mem=%s\n" %(memory))
        filey.writelines("python /scratch/users/tabakg/qsd_dev/hybrid_qsd.py --input_file '%s' --output_dir '%s'" % (markov_model_path, output_dir))
        filey.close()
        os.system("sbatch -p %s %s/qsd_%s.job" %(partition,job_dir,seed))
