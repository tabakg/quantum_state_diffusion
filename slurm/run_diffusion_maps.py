#!/usr/bin/python env

'''
Run diffusion maps.

'''

import os
import hashlib
overwrite=False

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

def get_file_params(traj):
    params={}
    things = traj[4:].split('-')
    first = things[0].split('_')
    params['seed'] = int(first[-1])
    params['regime'] = str('_'.join(first[:-1])).split('/')[-1]
    params['ntraj'] = int(things[1])
    try:
        (delta_t1,delta_t2,Nfock_a,Nfock_j,
            duration,downsample,method,num_systems,
            R,EPS,noise_amp,trans_phase,drive) = things[2:]
        params['delta_t'] = float("".join([delta_t1,'-',delta_t2]))
    except:
        (delta_t,Nfock_a,Nfock_j,
            duration,downsample,method,num_systems,
            R,EPS,noise_amp,trans_phase,drive) = things[2:]
        params['delta_t'] = float(delta_t)

    drive = drive[:-4]

    params['Nfock_j'] = int(Nfock_j)
    params['duration'] = float(duration)
    params['downsample'] = int(downsample)
    params['method'] = str(method)
    params['num_systems'] = int(num_systems)
    params['R'] = float(R)
    params['EPS'] =float(EPS)
    params['noise_amp'] =float(noise_amp)
    params['trans_phase'] =float(trans_phase)
    params['drive'] = True if drive == "True" else False
    return params

def files_by_params(files, bools):
    """
    Return a list of lists, each one having the unique files with distinct params determined by params_bool
    """
    params_each_file = {f: get_file_params(f) for f in files}
    relevant_params_each_file = {f : tuple(sorted((k,v) for k,v in p.items() if bools[k]))
        for f, p in params_each_file.items()}
    all_possible_params = set(relevant_params_each_file.values())
    groups = {p : [f for f in files if relevant_params_each_file[f] == p]
        for p in all_possible_params}
    return list(groups.values())

def make_hash(traj):
    """We make a name using a hash because there could be multiple
    trajectories in traj_list feeding into a single set of diffusion maps"""
    hash_code = hashlib.sha256(traj.encode('utf-8'))
    return hash_code.hexdigest()

## Which values to use to distinguish groups of files
bools = {'seed': False,
         'regime': True,
         'ntraj': True,
         'delta_t': True,
         'Nfock_j': True,
         'duration': True,
         'downsample': True,
         'method': True,
         'num_systems': True,
         'R': True,
         'EPS': True,
         'noise_amp': True,
         'trans_phase': True,
         'drive': True}

files = [os.path.join(trajectory_folder,f) for f in os.listdir(trajectory_folder) if f[-3:] == 'pkl']
file_lists = files_by_params(files, bools)

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
