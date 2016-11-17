# Solving quantum state diffusion (QSD) numerically.

The script [quantum_state_diffusion.py](quantum_state_diffusion.py) can be used to run QSD simulations.
I am using a (slightly) modified version of a package called sdeint. The only
modification I made is to normalize the trajectories for numerical stability.

My version can be found on [https://github.com/tabakg/sdeint](https://github.com/tabakg/sdeint)

There are two notebooks currently, one for the Kerr system and the second
for the absorptive bi-stability. Please compare the results to those obtained
using quantum jump trajectories (found on [this repo](https://github.com/tabakg/diffusion_maps)).

# Running

You have several options for running the simulation, including container-based and local environments:

- Docker 
- Singularity
- Virtual Environment

Depending on your familiarity with containers, the first two are recommended to handle software dependencies. Complete instructions are included below.


## Docker
The development environment is Dockerized, meaning that you can run the simulation with a Docker image. First, you need to [install Docker](http://54.71.194.30:4111/engine/installation). The base command to see help for how to run:


      docker run tabakg/quantum_state_diffusion --help


will show you the following (after a message about the font cache):


	usage: make_quantum_trajectory.py [-h] [--ntraj NTRAJ] [--duration DURATION]
		                          [--delta_t DELTAT] [--Nfock_a NFOCKA]
		                          [--Nfock_j NFOCKJ] [--downsample DOWNSAMPLE]
		                          [--verbose] [--output_dir OUTDIR]
		                          [--save2pkl] [--save2mat]

	generating trajectories using quantum state diffusion

	optional arguments:
	  -h, --help            show this help message and exit
	  --ntraj NTRAJ         Parameter Ntraj
	  --duration DURATION   Duration in ()
	  --delta_t DELTAT      Parameter delta_t
	  --Nfock_a NFOCKA      Parameter N_focka
	  --Nfock_j NFOCKJ      Parameter N_fockj
	  --downsample DOWNSAMPLE
		                How much to downsample results
	  --verbose             Turn on verbose logging (debug and info)
	  --output_dir OUTDIR   Output folder. If not defined, will use PWD.
	  --save2pkl            Save pickle file to --output_dir
	  --save2mat            Save .mat file to --output_dir


### Run and save to local machine

Note that the `--verbose` option can be added for better debugging. By default, no data is saved. To save, you will need to 1) specify the output directory to the `/data` folder in the container using the `output_dir` argument and 2) map some directory on your local machine to this `/data` folder.  We can do that like this:


           # on your local machine, let's say we want to save to Desktop
           docker run -v /home/vanessa/Desktop:/data \
                         tabakg/quantum_state_diffusion --output_dir /data --save2pkl
           
           
The above will produce the following output:

	INFO:root:Parameter duration set to 10
	INFO:root:Parameter delta_t set to 0.02
	INFO:root:Parameter Nfock_j set to 2
	INFO:root:Parameter Nfock_a set to 50
	INFO:root:Parameter Ntraj set to 10
	INFO:root:Downsample set to 100
	INFO:root:Regime is set to absorptive_bistable
	Run time:   1.2244760990142822  seconds.
	INFO:root:Saving pickle file...
	INFO:root:Saving result to /data/QSD_absorptive_bistable.pkl
	INFO:root:Data saved to pickle file /data/QSD_absorptive_bistable


The final output will be in the mapped folder - in the example above, this would be my Desktop at `/home/vanessa/Desktop/QSD_absorptive_bistable.pkl`


### Run inside container
You may want to inspect the data using the same environment it was generated from, in which case you would want to shell into the container. To do this, you can run:


      docker run -it --entrypoint=/bin/bash tabakg/quantum_state_diffusion


if you type `ls` you will see that we are sitting in the `/code` directory that contains the core python files. This means that we can run the analysis equivalently:


	/code# python make_quantum_trajectory.py --output_dir /data --save2pkl
	INFO:root:Parameter duration set to 10
	INFO:root:Parameter Ntraj set to 10
	INFO:root:Parameter Nfock_a set to 50
	INFO:root:Parameter Nfock_j set to 2
	INFO:root:Parameter delta_t set to 0.02
	INFO:root:Downsample set to 100
	INFO:root:Regime is set to absorptive_bistable
	Run time:   1.1898915767669678  seconds.
	INFO:root:Saving pickle file...
	INFO:root:Saving result to /data/QSD_absorptive_bistable.pkl
	INFO:root:Data saved to pickle file /data/QSD_absorptive_bistable


and the data is inside the container with us! Great.

	root@4420ae9e385d:/code# ls /data
	QSD_absorptive_bistable.pkl
      

### Customize the Docker image
If you don't want to use the image from Docker Hub (for example, if you want to make changes first) you can also build the image locally. You can build the image by doing the following:


      git clone https://www.github.com/tabakg/quantum_state_diffusion
      cd quantum_state_diffusion
      docker build -t tabakg/quantum_state_diffusion .


Note the `.` at the end of the command to specify the present working directory.


## Singularity
Singularity is a container that is HPC friendly, meaning that it can be run on a cluster environment. We have provided a Singularity file that can bootstrap the Docker image to build the image.


**being written**


# Local Installation:

Installation requires Python 3.

In addition to standard libraries (numpy, sympy, scipy, pickle)

In addition to the modified version of sdeint found on
https://github.com/tabakg/sdeint (mentioned above), please install
QNET (https://pypi.python.org/pypi/QNET). QNET is on pip, and can be installed
simply with:

    pip install QNET.

I am also using a package called multiprocess, which can be installed with

    pip install multiprocess
