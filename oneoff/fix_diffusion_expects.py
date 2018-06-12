import pickle
import os

def load_trajectory(loc):
    pkl_file = open(loc, 'rb')
    pkl_dict = pickle.load(pkl_file)
    pkl_file.close()
    return pkl_dict

def save_diffusion_coordinates(loc, pkl_dict):

    pkl_file = open(loc, 'wb')
    pickle.dump(pkl_dict, pkl_file, protocol=0)
    pkl_file.close()

    return

diffusion_map_data='/scratch/users/tabakg/qsd_output/diffusion_maps/diffusion_maps_data'

for file in os.listdir(diffusion_map_data):
    D = load_trajectory(file)
    expects = D['expects']
    if len(expects.shape) == 1:
        print("Found a buggy one!")
        times = D['times']
        num_expects = int(expects.shape[0] / times.shape[0])
        D['expects'] = expects.reshape(int(expects.shape[0] / num_expects),
                                      num_expects)
        save_diffusion_coordinates(file, D)
    else:
        print("Found a non-buggy one")
