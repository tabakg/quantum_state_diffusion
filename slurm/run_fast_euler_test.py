import os
import numpy as np

# Variables to run jobs
json_spec_dir="/scratch/users/tabakg/qsd_output/json_spec/"
output_dir="/scratch/users/tabakg/qsd_output/fast_out/"
dev_dir="/scratch/users/tabakg/qsd_dev"
fast_sim_dir="/scratch/users/tabakg/qsd_dev/fast_euler"

# Variables for each job
memory = 4000
partition = 'normal'

# Create subdirectories for job, error, and output files
job_dir = os.path.join(json_spec_dir, ".job")
out_dir = os.path.join(json_spec_dir, ".out")
for new_dir in [output_dir,job_dir,out_dir]:
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

OVERWRITE=False

REGIME = 'kerr_bistableA21.75'
# delta_ts = [3e-7, 1e-6, 3e-6, 1e-5]
delta_ts = [1e-5]
num_systems_arr=[2]

NUM_SEEDS=1
ntraj=1
Nfock_a=30
Nfock_j=2
duration=0.2
downsample=100
noise_amp=1.0
trans_phase=1.0
lambd=0.999

SEEDs=range(1, NUM_SEEDS + 1)
# Rs=[0.3,0.6,0.8,1.0]
Rs = [1.0]
# min_EPSs = [0. if r ==0. else (1-np.sqrt(1-r**2))/r for r in Rs]
# EPSs = {R: np.linspace(m,2*m,5) for R,m in zip(Rs,min_EPSs)}
EPSs = {1.0 : [1.0]}
DRIVES=False, ##Driving second system?

sdeint_method_name = 'itoImplicitEuler'

for num_systems in num_systems_arr:
    for delta_t in delta_ts:
        for seed in SEEDs:
            for R in Rs:
                for EPS in EPSs[R]:
                    for drive_second_system in DRIVES:
                        param_str = ("%s_"*15)[:-1] %(seed,
                                                     ntraj,
                                                     delta_t,
                                                     Nfock_a,
                                                     Nfock_j,
                                                     duration,
                                                     downsample,
                                                     sdeint_method_name,
                                                     num_systems,
                                                     R,
                                                     EPS,
                                                     noise_amp,
                                                     lambd,
                                                     trans_phase,
                                                     drive_second_system)
                        json_spec_name = "json_spec_" + param_str + ".json"
                        json_spec_loc = os.path.join(json_spec_dir, json_spec_name)
                        json_spec_exists = os.path.isfile(json_spec_loc)

                        psis_name = "psis_" + param_str + ".json"
                        expects_name = "expects_" + param_str + ".json"
                        psis_loc = os.path.join(output_dir, psis_name)
                        expects_loc = os.path.join(output_dir, expects_name)

                        print("OVERWRITE is %s and file %s existence is %s" %(OVERWRITE, json_spec_name, json_spec_exists))
                        print("If overwriting or file does not exist, going to process new seed.")

                        if OVERWRITE or not json_spec_exists:
                            print ("Processing seed %s" %(seed))

                            ## Write job to file
                            filey_loc = os.path.join(job_dir, "%s.job" %(seed))
                            filey = open(filey_loc, "w")
                            filey.writelines("#!/bin/bash\n")
                            filey.writelines("#SBATCH --job-name=QSD\n")
                            filey.writelines("#SBATCH --output=%s/qsd_%s.out\n" %(out_dir,seed))
                            filey.writelines("#SBATCH --error=%s/qsd_%s.err\n" %(out_dir,seed))
                            filey.writelines("#SBATCH --time=2-00:00\n")
                            filey.writelines("#SBATCH --mem=%s\n" %(memory))

                            script_name = os.path.join(dev_dir, "generate_num_model.py")
                            fast_sim_name = os.path.join(fast_sim_dir, "fast_sim")

                            ## Python script
                            to_file = ("python %s --output_dir '%s' --Nfock_a %s"
                                      "--seed %s --regime '%s' --num_systems %s "
                                      "--delta_t %s --duration %s --downsample %s --sdeint_method_name '%s' "
                                      "--R %s --eps %s --noise_amp %s --lambda %s"
                                      %(script_name, json_spec_dir, Nfock_a,
                                       seed, REGIME, num_systems,
                                       delta_t, duration, downsample, sdeint_method_name,
                                       R, EPS, noise_amp, lambd))
                            if drive_second_system:
                                to_file = to_file + " --drive_second_system %s" % drive_second_system
                            to_file += "\n"
                            filey.writelines(to_file)

                            ## C++ code
                            filey.writelines("%s %s %s %s" %(fast_sim_name, json_spec_loc, psis_loc, expects_loc))

                            filey.close()

                            ## run batch script
                            os.system("sbatch -p %s %s" %(partition, filey_loc))
