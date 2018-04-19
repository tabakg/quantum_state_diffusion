import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pickle
import os
import numpy as np

OVERWRITE=False
parent_dir, _ = os.path.split(os.getcwd())
traj_folder = os.path.join(parent_dir,"trajectory_data")
bools = [False] + [True]*12

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

files = [f for f in os.listdir(traj_folder) if f[-3:] == 'pkl']
file_lists = files_by_params(files, bools)

for file_lst in file_lists:
    name = "N_plot_timeseries_" + "-".join([par for i,par in enumerate(get_file_params(file_lst[0])) if bools[i]]) + ".pdf"
    plot_path = os.path.join(os.getcwd(), name)
    if not OVERWRITE and os.path.exists(plot_path):
        pass
    for file in file_lst:
        print ("File name: %s" % file)
        try:
            D = pickle.load( open( os.path.join(traj_folder,file), "rb" ) )
        except EOFError:
            print("Warning, could not open pickle file.")
        t_span = D['times']
        n_traj, n_t, n_obs = D['expects'].shape
        print (D['observable_str'])

        for i in range(n_traj):
            for j in [0,3]:
                plt.plot(t_span, D['expects'][i,:,j].real)
    plt.savefig(name)
    plt.gcf().clear()

for file_lst in file_lists:
    name = "N_plot_scatter_" + "-".join([par for i,par in enumerate(get_file_params(file_lst[0])) if bools[i]]) + ".pdf"
    plot_path = os.path.join(os.getcwd(), name)
    if not OVERWRITE and os.path.exists(plot_path):
        pass
    for file in file_lst:
        print ("File name: %s" % file)
        try:
            D = pickle.load( open( os.path.join(traj_folder,file), "rb" ) )
        except EOFError:
            print("Warning, could not open pickle file.")
        t_span = D['times']
        n_traj, n_t, n_obs = D['expects'].shape
        print (D['observable_str'])

        for i in range(n_traj):
            ## make a scatterplot of N2 versus N1 for the second half of the trajectory
            traj_length = D['expects'].shape[1]
            plt.scatter( D['expects'][i,traj_length/2:,0].real, D['expects'][i,traj_length/2:,3].real, s=3, edgecolors='none', alpha=0.3)
    plt.savefig(name)
    plt.gcf().clear()
