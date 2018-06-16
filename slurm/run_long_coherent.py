#!/usr/bin/python env

'''
Quantum State Diffusion: Submit jobs on SLURM

'''

import os
import numpy as np

# Variables to run jobs
## basedir = os.path.abspath(os.getcwd())
output_dir='/scratch/users/tabakg/qsd_output/trajectory_data'

# Variables for each job
memory = 16000
partition = 'normal'

# Create subdirectories for job, error, and output files
job_dir = "%s/.job" %(output_dir)
out_dir = "%s/.out" %(output_dir)
for new_dir in [output_dir,job_dir,out_dir]:
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

OVERWRITE=False

NUM_SEEDS=16
REGIME='kerr_bistable'
ntraj=1
delta_t=1e-5
Nfock_a=50
Nfock_j=2
duration=100.0
downsample=1000
num_systems=2
noise_amp=1.0
trans_phase=1.0

SEEDs=range(1, NUM_SEEDS + 1)
R=0.
EPS=0.
drive=False

REGIME='kerr_bistable'
SDE_METHODS='itoEuler','itoImplicitEuler','itoSRI2'

for seed in SEEDs:
    for method in SDE_METHODS:
        param_str = ("%s-"*14)[:-1] %(seed,
                                    ntraj,
                                    delta_t,
                                    Nfock_a,
                                    Nfock_j,
                                    duration,
                                    downsample,
                                    method,
                                    num_systems,
                                    R,
                                    EPS,
                                    noise_amp,
                                    trans_phase,
                                    drive)
        file_name = 'QSD_%s_%s.pkl' %(REGIME,param_str)
        file_loc = os.path.join(output_dir,file_name)
        file_exists = os.path.isfile(file_loc)

        print("OVERWRITE is %s and file %s existence is %s" %(OVERWRITE,file_name,file_exists))
        print("If overwriting or file does not exist, going to process new seed.")

        if OVERWRITE or not file_exists:
            print ("Processing seed %s" %(seed))
            # Write job to file
            filey_loc = os.path.join(job_dir, "qsd_%s.job" %(seed))
            filey = open(filey_loc,"w")
            filey.writelines("#!/bin/bash\n")
            filey.writelines("#SBATCH --job-name=S%sR%sE%s\n" %(seed, R, EPS))
            filey.writelines("#SBATCH --output=%s/qsd_%s.out\n" %(out_dir,seed))
            filey.writelines("#SBATCH --error=%s/qsd_%s.err\n" %(out_dir,seed))
            filey.writelines("#SBATCH --time=2-00:00\n")
            filey.writelines("#SBATCH --mem=%s\n" %(memory))
            filey.writelines("module load singularity\n")
            filey.writelines("module load system\n")
            filey.writelines("module load singularity/2.4\n")

            if drive:
              filey.writelines("singularity run --bind %s:/data qsd..img --output_dir /data "
                               "--seed %s --save2pkl --regime '%s' --num_systems 2 "
                               "--delta_t %s --duration %s --downsample %s --sdeint_method_name '%s' "
                               "--R %s --eps %s --noise_amp 1. --drive_second_system True"
                               "\n" %(output_dir,seed,REGIME,delta_t,duration,downsample,method,R,EPS))
            else:
              filey.writelines("singularity run --bind %s:/data qsd..img --output_dir /data "
                               "--seed %s --save2pkl --regime '%s' --num_systems 2 "
                               "--delta_t %s --duration %s --downsample %s --sdeint_method_name '%s' "
                               "--R %s --eps %s --noise_amp 1."
                               "\n"%(output_dir,seed,REGIME,delta_t,duration,downsample,method,R,EPS))
            filey.close()
            os.system("sbatch -p %s %s" %(partition,filey_loc))
