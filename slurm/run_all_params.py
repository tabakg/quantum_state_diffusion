#!/usr/bin/python env

'''
Diffusion Maps: Submit jobs on SLURM

'''

import os

# Variables to run jobs
## basedir = os.path.abspath(os.getcwd())
output_dir='/scratch/users/tabakg/qsd_output'
traj_folder='/scratch/users/tabakg/qsd_output/trajectory_data'

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

NUM_SEEDS=8
REGIME='kerr_bistable'
ntraj=1
delta_t=1e-5
Nfock_a=50
Nfock_j=2
duration=30.0 ## try running longer ones? say 100. Make sure appropriate amount is being cut off
downsample=1000
num_systems=2
noise_amp=1.0 ## matters for two systems only
trans_phase=1.0 ## matters for two systems only

SEEDs=range(1, NUM_SEEDS + 1)
Rs=0.,0.6,0.8,1.0 ## matters for two systems only
EPSs=0.,0.3333333,0.5,1.0 ## matters for two systems only
DRIVES=True,False ##Driving second system? ## matters for two systems only

METHODS={(R,EPS):['itoSRI2'] for R,EPS in zip(Rs,EPSs)}
METHODS[(0.,0.)] += ['itoEuler','itoImplicitEuler']

for seed in SEEDs:
  for R, EPS in zip(Rs, EPSs):
    for drive in DRIVES:
      for method in METHODS[(R,EPS)]:
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
        file_loc = os.path.join(traj_folder,file_name)
        file_exists = os.path.isfile(file_loc)

        print("OVERWRITE is %s and file %s existence is %s" %(OVERWRITE,file_name,file_exists))
        print("If overwriting or file does not exist, going to process new seed.")

        if OVERWRITE or not file_exists:
          print "Processing seed %s" %(seed)
          # Write job to file
          filey = ".job/qsd_%s.job" %(seed)
          filey = open(filey,"w")
          filey.writelines("#!/bin/bash\n")
          filey.writelines("#SBATCH --job-name=making_%s\n" %(file_name))
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
          os.system("sbatch -p %s .job/qsd_%s.job" %(partition,seed))
