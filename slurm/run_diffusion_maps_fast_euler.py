#!/usr/bin/python env

'''
Run diffusion maps.

'''
import sys
import os
import hashlib
sys.path.append('/scratch/users/tabakg/qsd_dev')
from utils import get_params
from utils import files_by_params
from utils import bools
from utils import make_hash
overwrite=False

# Variables to run jobs
## basedir = os.path.abspath(os.getcwd())
output_dir='/scratch/users/tabakg/qsd_output/diffusion_maps_fast_euler'
trajectory_folder='/scratch/users/tabakg/qsd_output/fast_out_pickle'

## make a folder for trajectory data
diffusion_maps_folder = os.path.join(output_dir, "diffusion_maps_data")
try:
    os.stat(diffusion_maps_folder)
except:
    os.mkdir(diffusion_maps_folder)

# Variables for each job
memory = 64000
partition = 'normal'

# Create subdirectories for job, error, and output files
job_dir = "%s/.job" %(output_dir)
out_dir = "%s/.out" %(output_dir)
for new_dir in [output_dir,job_dir,out_dir]:
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

files = [os.path.join(trajectory_folder,f) for f in os.listdir(trajectory_folder) if f[-3:] == 'pkl']
file_lists = files_by_params(files, bools, duration=50., max_seed = 0)

for file_list in file_lists:

    traj = ",".join(sorted(file_list))
    hash_code = make_hash(traj)
    output = '%s/diffusion_map_%s.pkl' %(diffusion_maps_folder, hash_code)

    filey_loc = os.path.join(job_dir, "diff_map.job")
    filey = open(filey_loc,"w")
    filey.writelines("#!/bin/bash\n")
    filey.writelines("#SBATCH --job-name=diff_map\n")
    filey.writelines("#SBATCH --output=%s/diff_map_%s.out\n" %(out_dir,hash_code))
    filey.writelines("#SBATCH --error=%s/diff_map_%s.err\n" %(out_dir,hash_code))
    filey.writelines("#SBATCH --time=2-00:00\n")
    filey.writelines("#SBATCH --mem=%s\n" %(memory))

    if os.path.exists(output) and overwrite is False:
        print("File exists and overwrite is False! Aborting diffusion map:\n"
                + output)
        continue

    filey.writelines("python /scratch/users/tabakg/qsd_dev/diffusion_maps.py --traj '%s' --output_dir %s --output_name %s" % (traj, output_dir, output))
    filey.close()
    os.system("sbatch -p %s %s" %(partition, filey_loc))
