#!/usr/bin/python env

'''
Run diffusion maps.

'''

import os
import hashlib
OVERWRITE=False

# Variables to run jobs
## basedir = os.path.abspath(os.getcwd())
output_dir='/scratch/users/tabakg/qsd_output/diffusion_maps'
trajectory_folder='/scratch/users/tabakg/qsd_output/trajectory_data'

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

def get_file_params(file):
    splits = file[:-4].split('-')[1:]
    s = 0
    while s < len(splits)-1:
        if splits[s][-1]=='e':
            splits[s] = splits[s] + '-' + splits[s+1]
            splits = splits[:s+1] + splits[s+2:]
        s+=1
    return splits

def files_by_params(files, params_bool):
  """
  Return a list of lists, each one having the unique files with distinct params determined by params_bool
  """
  D = {file: "".join([p for i,p in enumerate(get_file_params(file))
                      if params_bool[i]])
           for file in files}
  return [[k for k in D if D[k] == v]
          for v in set(D.values())]

def make_hash(traj):
    """We make a name using a hash because there could be multiple
    trajectories in traj_list feeding into a single set of diffusion maps"""
    hash_code = hashlib.sha256(traj.encode('utf-8'))
    return hash_code.hexdigest()

bools = [False] + [True]*12
files = [os.path.join(trajectory_folder,f) for f in os.listdir(trajectory_folder) if f[-3:] == 'pkl']
file_lists = files_by_params(files, bools)

for file_list in file_lists:

    filey_loc = os.path.join(job_dir, "diff_map.job")
    filey = open(filey_loc,"w")
    filey.writelines("#!/bin/bash\n")
    filey.writelines("#SBATCH --job-name=diff_map\n")
    filey.writelines("#SBATCH --output=%s/diff_map.out\n" %(out_dir))
    filey.writelines("#SBATCH --error=%s/diff_map.err\n" %(out_dir))
    filey.writelines("#SBATCH --time=2-00:00\n")
    filey.writelines("#SBATCH --mem=%s\n" %(memory))

    traj = ",".join(file_list)
    hash_code = make_hash(traj)
    output = '%s/diffusion_map_%s.pkl' %(diffusion_maps_folder, hash_code)

    if os.path.exists(output) and overwrite is False:
        print("File exists and overwrite is False! Aborting diffusion map:\n"
                + output)
        continue

    filey.writelines("python /scratch/users/tabakg/qsd_dev/diffusion_maps.py --traj '%s' --output_dir %s --output_name" % (traj, output_dir, output))
    filey.close()
    os.system("sbatch -p %s %s" %(partition, filey_loc))
