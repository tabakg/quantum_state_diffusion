#!/usr/bin/python env

'''
Make plots: Submit jobs on SLURM

'''

import os

# Variables to run jobs
## basedir = os.path.abspath(os.getcwd())
output_dir='/scratch/users/tabakg/qsd_output/make_plots'
trajectory_folder='/scratch/users/tabakg/qsd_output/trajectory_data'

# Variables for each job
memory = 16000
partition = 'normal'

# Create subdirectories for job, error, and output files
job_dir = "%s/.job" %(output_dir)
out_dir = "%s/.out" %(output_dir)
for new_dir in [output_dir,job_dir,out_dir]:
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

# for traj_file in [f if f[-4:] == '.pkl' for f in os.listdir(trajectory_folder)]:

filey = ".job/plots.job"
filey = open(filey,"w")
filey.writelines("#!/bin/bash\n")
filey.writelines("#SBATCH --job-name=plots\n")
filey.writelines("#SBATCH --output=%s/plots.out\n" %(out_dir))
filey.writelines("#SBATCH --error=%s/plots.err\n" %(out_dir))
filey.writelines("#SBATCH --time=2-00:00\n")
filey.writelines("#SBATCH --mem=%s\n" %(memory))

filey.writelines("python make_plots.py")
filey.close()
os.system("sbatch -p %s .job/plots.job" %(partition))
