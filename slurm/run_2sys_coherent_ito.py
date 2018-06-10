#!/usr/bin/python env

'''
Quantum State Diffusion: Submit jobs on SLURM

'''

import os

# Variables to run jobs
## basedir = os.path.abspath(os.getcwd())
output_dir='/scratch/users/tabakg/qsd_output/trajectory_data'

# Variables for each job
memory = 16000
partition = 'normal'

DELTA_T=1e-5
DURATION=30
DOWNSAMPLE=1000
NUM_SEEDS=8

REGIME='kerr_bistable'
SDE_METHODS='itoEuler','itoImplicitEuler'

# Create subdirectories for job, error, and output files
job_dir = "%s/.job" %(output_dir)
out_dir = "%s/.out" %(output_dir)
for new_dir in [output_dir,job_dir,out_dir]:
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

# We are going to vary the seed argument, and generate and submit a job for each
seeds = range(1, NUM_SEEDS + 1)
for method in SDE_METHODS:
    for seed in seeds:
        print "Processing seed %s" %(seed)
        # Write job to file
        filey = ".job/qsd_%s.job" %(seed)
        filey = open(filey,"w")
        filey.writelines("#!/bin/bash\n")
        filey.writelines("#SBATCH --job-name=qsd_%s\n" %(seed))
        filey.writelines("#SBATCH --output=%s/qsd_%s.out\n" %(out_dir,seed))
        filey.writelines("#SBATCH --error=%s/qsd_%s.err\n" %(out_dir,seed))
        filey.writelines("#SBATCH --time=2-00:00\n")
        filey.writelines("#SBATCH --mem=%s\n" %(memory))
        filey.writelines("module load singularity\n")
        filey.writelines("module load system\n")
        filey.writelines("module load singularity/2.4\n")
        filey.writelines("singularity run --bind %s:/data qsd..img --output_dir /data "
                         "--seed %s --save2pkl --regime '%s' --num_systems 2 "
                         "--delta_t %s --duration %s --downsample %s --sdeint_method_name '%s'"
                         "\n" %(output_dir,seed,REGIME,DELTA_T,DURATION,DOWNSAMPLE,method))
        filey.close()
        os.system("sbatch -p %s .job/qsd_%s.job" %(partition,seed))
