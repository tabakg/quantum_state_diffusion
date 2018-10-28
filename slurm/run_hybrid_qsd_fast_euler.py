#!/usr/bin/python env

r'''
Run the hybrid_qsd.py code using recoded Markov models for system 1
and input parameters for system 2.
'''

import os
import sys
sys.path.append('/scratch/users/tabakg/qsd_dev')

from utils import get_params_json
from utils import load_trajectory

markov_model_folder='/scratch/users/tabakg/qsd_output/markov_model_data_fast_euler'
output_dir='/scratch/users/tabakg/qsd_output/hybrid_trajectory_data_fast_euler'
diffusion_maps_folder='/scratch/users/tabakg/qsd_output/diffusion_maps_fast_euler/diffusion_maps_data'
# Variables for each job
memory = 64000
partition = 'normal'

# Create subdirectories for job, error, and output files
job_dir = "%s/.job" %(output_dir)
out_dir = "%s/.out" %(output_dir)
for new_dir in [output_dir,job_dir,out_dir]:
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

OVERWRITE=True

files = [f for f in os.listdir(markov_model_folder) if f[-3:] == 'pkl']

# Seed for generating trajectory of system 1 and stochastic terms of system 2.
seed = 101
duration = 0.1 ## For testing purposes

for markov_model_file in files[:5]: ## stop at 5 files for testing purposes

    markov_params = markov_model_file.split('/')[-1].split('.')[0].split('_')[-3:]
    markov_seed, n_clusters, n_iter = markov_params

    markov_model_path = os.path.join(markov_model_folder, markov_model_file)
    hash_name = markov_model_file[len('markov_model_'):][:64]
    diffusion_map_file = 'diffusion_map_'+hash_name+'.pkl'
    diffusion_maps_path = os.path.join(diffusion_maps_folder, diffusion_map_file)

    params = get_params_json(load_trajectory(diffusion_maps_path)['traj_list'][0])
    num_systems = params['num_systems']
    if not 'regime' in params:
        regime = "kerr_bistable"
    else:
        regime = params['regime']

    ## For the time being, kerr qubit is not yet implemented, so we just skip it here.
    if regime == 'kerr_qubit':
        continue

    ## For the time being, we are only interested in reducing one system and feeding it into another.
    ## For this reason, if the input system was actually 2 systems, we just skip it.

    if num_systems == 2:
        continue

    print("hash name: %s " %hash_name)

    name = 'hybrid_QSD_' + str(seed) + "_" + str(duration) + "_" + hash_name + "_" + markov_seed  + "_" + n_clusters  + "_" + n_iter +'.pkl'
    output_file_path = os.path.join(output_dir, name)
    file_exists = os.path.isfile(output_file_path)

    print("OVERWRITE is %s and file %s existence is %s" %(OVERWRITE, output_file_path, file_exists))
    print("If overwriting or file does not exist, going to process new seed.")

    if OVERWRITE or not file_exists:
        print ("Processing seed %s" %(seed))
        # Write job to file
        filey = "%s/qsd_%s.job" %(job_dir, seed)
        filey = open(filey,"w")
        filey.writelines("#!/bin/bash\n")
        filey.writelines("#SBATCH --job-name=hybrid_%s\n" %(seed))
        filey.writelines("#SBATCH --output=%s/hybrid_qsd_seed_%s_hash_%s.out\n" %(out_dir,seed,hash_name))
        filey.writelines("#SBATCH --error=%s/hybrid_qsd_seed_%s_hash_%s.err\n" %(out_dir,seed,hash_name))
        filey.writelines("#SBATCH --time=2-00:00\n")
        filey.writelines("#SBATCH --mem=%s\n" %(memory))
        filey.writelines("python /scratch/users/tabakg/qsd_dev/hybrid_qsd.py --seed %s --duration %s --markov_file '%s' --diffusion_file '%s' --output_file_path '%s' --save2pkl" % (seed, duration, markov_model_path, diffusion_maps_path, output_file_path))
        filey.close()
        os.system("sbatch -p %s %s/qsd_%s.job" %(partition,job_dir,seed))
