import sys
sys.path.append('/scratch/users/tabakg/qsd_dev')
import os
try:
   import cPickle as pickle
except:
   import pickle
from utils import files_by_params
from utils import bools
from utils import load_trajectory
from utils import save
from utils import make_hash
import numpy as np
import logging

sample_size=10000

overwrite=False
traj_dir='/scratch/users/tabakg/qsd_output/trajectory_data'
expects_dir='/scratch/users/tabakg/qsd_output/expects'

## Make directories if they don't exist
for new_dir in [traj_dir,expects_dir]:
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

## Get all pickle files in traj_dir
files = [f for f in os.listdir(traj_dir) if f[-3:] == 'pkl']
file_lists = files_by_params(files, bools, max_seed=16)

for file_list in file_lists:

    traj = ",".join(sorted(file_list))
    hash_code = make_hash(traj)
    traj_list = [item for item in traj.split(',')]
    print("Extracting expects for hashcode %s" %hash_code)
    expects_name = 'expects_%s.pkl' %(hash_code)
    expects_loc = os.path.join(expects_dir, expects_name)

    if overwrite is False and os.path.exists(expects_loc):
        continue

    coords_dict = {'expects': [], 'times': [], 'traj_list': traj_list}
    num_successful = 0
    for traj in traj_list:
        try:
            loaded = load_trajectory(traj)
            num_successful += 1
        except pickle.UnpicklingError:
            logging.info("Could not open trajectory %s" %traj)

        expects_current_traj = np.concatenate(loaded['expects'])

        ## Find downsample factor to avoid using too much memory
        every_other_n = int(expects_current_traj.shape[0] * len(traj_list) / (sample_size))
        if every_other_n == 0:
            every_other_n = 1

        coords_dict['expects'].append(expects_current_traj[::every_other_n])
        coords_dict['times'].append(loaded['times'][::every_other_n])


    ## Consolidate expects and times for consistency
    coords_dict['times'] = np.concatenate(coords_dict['times'])
    coords_dict['expects'] = np.concatenate(coords_dict['expects'])

    ## Output messages
    logging.info("Successfully loaded %s/%s trajectories." %(len(traj_list), num_successful))
    save(expects_loc, coords_dict)
