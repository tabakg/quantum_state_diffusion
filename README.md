# Solving quantum state diffusion (QSD) numerically.

The script quantum_state_diffusion.py can be used to run QSD simulations.
I am using a (slightly) modified version of a package called sdeint. The only
modification I made is to normalize the trajectories for numerical stability.

My version can be found on https://github.com/tabakg/sdeint

There are two notebooks currently, one for the Kerr system and the second
for the absorptive bi-stability. Please compare the results to those obtained
using quantum jump trajectories (found on https://github.com/tabakg/diffusion_maps).

# Requirements:

Installation requires Python 3.

In addition to standard libraries (numpy, sympy, scipy, pickle)

In addition to the modified version of sdeint found on
https://github.com/tabakg/sdeint (mentioned above), please install
QNET (https://pypi.python.org/pypi/QNET). QNET is on pip, and can be installed
simply with:

    pip install QNET.

I am also using a package called multiprocess, which can be installed with

    pip install multiprocess
