# Solving quantum state diffusion (QSD) numerically.

The script quantum_state_diffusion.py can be used to run QSD simulations.
I am using a (slightly) modified version of a package called sdeint. My version
can be found on https://github.com/tabakg/sdeint

The only modification I made is to normalize the trajectories for numerical
stability

There are two notebooks currently, one for the Kerr system and the second
for the absorptive bi-stability. Please compare the results to those obtained
using quantum jump trajectories (found on https://github.com/tabakg/diffusion_maps).
