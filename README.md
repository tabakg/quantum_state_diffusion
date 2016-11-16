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

## Clone the repo
For all installations, you first need to clone the repo to your local machine:

      git clone https://www.github.com/tabakg/quantum_state_diffusion
      cd quantum_state_diffusion


## Docker
The development environment is Dockerized, meaning that you can run the simulation with a Docker image. First, you need to [install Docker](http://54.71.194.30:4111/engine/installation). To run by pulling the image from Docker Hub:


      docker run tabakg/quantum_state_diffusion 


If you don't want to use the image from Docker Hub (for example, if you want to make changes first) you can also build the image locally. You can build the image by doing the following:


      docker build -t tabakg/quantum_state_diffusion .


To interactive run a shell into the image, you can do:


      docker run -it tabakg/quantum_state_diffusion bash


For either of the above, if you want to interactively shell into the container, you can do:


      docker exec -it tabakg/quantum_state_diffusion bash


Note the `.` at the end of the command to specify the present working directory.


## Singularity



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
