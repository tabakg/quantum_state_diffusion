import os
import numpy as np

# Variables to run jobs
## basedir = os.path.abspath(os.getcwd())
output_dir='/scratch/users/tabakg/qsd_output/trajectory_data'
dev_dir='/scratch/users/tabakg/qsd_dev'

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

drives = np.arange(33.25, 37.5, 0.75)
regimes = ['kerr_bistableB'+str(d) for d in drives]

NUM_SEEDS=16
ntraj=1
delta_t=1e-5
Nfock_a=100
Nfock_j=2
duration=100.0
downsample=1000
num_systems=1
noise_amp=1.0
trans_phase=1.0

SEEDs=range(1, NUM_SEEDS + 1)
R=0.
EPS=0.
drive=True

SDE_METHODS=['itoSRI2']

for REGIME in regimes:
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

                script_name = os.path.join(dev_dir, "make_quantum_trajectory.py")
                filey.writelines("python %s --output_dir '%s' "
                                   "--seed %s --save2pkl --regime '%s' --num_systems %s "
                                   "--delta_t %s --duration %s --downsample %s --sdeint_method_name '%s' "
                                   "--R %s --eps %s --noise_amp %s --drive_second_system %s"
                                   "\n" %(script_name, output_dir,seed,REGIME,num_systems,delta_t,duration,downsample,method,R,EPS,noise_amp,drive))

                filey.close()
                os.system("sbatch -p %s %s" %(partition,filey_loc))
