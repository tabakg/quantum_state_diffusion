qsd_dev='/scratch/users/tabakg/qsd_dev/'
diffusion_maps='/scratch/users/tabakg/qsd_output/diffusion_maps/diffusion_maps_data'
clear_reg='QSD_kerr_bistableB'

import sys
import os
import pickle
sys.path.append(qsd_dev)
from utils import get_params

print("about to clear all regimes %s" %clear_reg)

def clear_regimes(directory=diffusion_maps, clear_reg=clear_reg):
    files = os.listdir(directory)
    for f in files:
        pkl_loc = os.path.join(directory, f)
        pkl_file = open(pkl_loc, 'rb')
        data = pickle.load(pkl_file)
        traj_params = {traj:get_params(traj) for traj in data['traj_list']}
        regime = list(traj_params.values())[0]['regime']
        print("Read regime %s" %regime)
        pkl_file.close()
        if regime[:len(clear_reg)] == clear_reg:
            os.remove(pkl_loc)

clear_regimes(directory=diffusion_maps, clear_reg=clear_reg)
