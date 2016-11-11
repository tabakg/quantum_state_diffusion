# Generating trajectories using quantum state diffusion. We will be primairly
# interested in the absorptive bistability (Jaynes Cummings model)
# I store trajectory files as *.pkl files or *.mat files. This way I can easily
# load them into another notebook, or load the trajectories to matlab.
# See also diffusion_maps/make_quantum_trajectories_absorptive_bistability
# Requires Python 3.

## QHDL package
from qnet.algebra.operator_algebra import *
from qnet.algebra.circuit_algebra import *
import qnet.algebra.state_algebra as sa

from sympy import sqrt

## SDE integrator
import sdeint

## quantum state diffusion solver
from quantum_state_diffusion import qsd_solve

## numerical libraries
import numpy as np
import numpy.linalg as la
from scipy import sparse

## plottinsg
import matplotlib.pyplot as plt

## pickle
import pickle

from save2matfile_or_pkl import save2matfile_or_pkl

## General parameters

params = {}

ntraj = params['Ntraj'] = 10
duration = params['duration'] = 10000
delta_t = params['delta_t'] = 2e-2
Nfock_a = params['Nfock_a'] = 50
Nfock_j = params['Nfock_j'] = 2

## how much to downsample results
downsample = 100

## seeds for simulation
seed = [i for i in range(ntraj)]

## Names of files
Regime = "absorptive_bistable"
file_name = './trajectory_data/QSD_' + Regime

### Which file formats to save trajectory data.
### Name of the file to save. The extension will be .mat for matlab and .pkl for pickle.

## matlab file.
save_mat = False
## pkl file (can be loaded in python) in the same format as above.
save_pkl = True

# ## Make Operators

a = Destroy(1)
ad = a.dag()

sm = LocalSigma(2, 1,0)/sqrt(2)
sp = sm.dag()
sz = sp*sm - sm*sp

j = Jminus(2)
jp = j.dag()
jz = Jz(2)

jx = (jp + j) / 2.
jy = (jp - j) / 2.

# ## Make SLH Model

k,g0,g = symbols("kappa, g0,gamma", positive=True)
DD, TT = symbols("Delta, Theta", real=True)
W = symbols("Omega")

L = [sqrt(k)*a,
     sqrt(g)*j]
H = -I*g0*(a*jp - ad * j) + DD*jz + TT*ad*a
S = identity_matrix(2)

slh = SLH(S, L, H).coherent_input(W,0)
slh

## Numerical parameters

a.space.dimension = Nfock_a
j.space.dimension = Nfock_j

# default are for absorptive bistability
def make_nparams(Cn=10.5, kn=.12, yn=11.3, DDn=0, TTn=0., J = 0.5):
    g0n = np.sqrt(2.*kn*Cn)
    Wn = yn*kn/np.sqrt(2)/g0n

    nparams = {
        W: Wn/np.sqrt(2*kn),
        k: 2*kn,
        g: 2./np.sqrt(2*J),
        g0: -g0n/np.sqrt(2*J),
        DD: DDn,
        TT: TTn,
    }
    xrs = np.linspace(0, 10)
    yrs = 2*Cn*xrs/(1+xrs**2) + xrs
    plt.plot(yrs, xrs)
    plt.vlines([yn], *plt.ylim())
    return nparams

if Regime == "absorptive_bistable":
    nparams = make_nparams()
else:
    raise ValueError("Unknown regime, or not implemented yet.")

Hq, Lqs = slh.substitute(nparams).HL_to_qutip()

## Observables

obs = (a, j, jz, a*a, a.dag()*a, a*jp, jp, jx, jy)
obsq = [o.to_qutip(full_space=slh.space) for o in obs]

tspan = np.arange(0,duration,delta_t)

psi0 = qutip.tensor(qutip.basis(Nfock_a,0),qutip.basis(Nfock_j,0)).data
H = Hq.data
Ls = [Lq.data for Lq in Lqs]
obsq = [ob.data for ob in obsq]

### Run simulation
D = qsd_solve(H, psi0, tspan, Ls, sdeint.itoEuler, obsq = obsq, ntraj = ntraj,
              seed = seed, normalize_state = True)

### include time in results
D.update({'tspan':tspan})

### downsample
D_downsampled = {'psis' : D['psis'][:,::downsample],
                 'obsq_expects' : D['obsq_expects'][:,::downsample],
                 'seeds' : D['seeds'],
                 'tspan' : D['tspan'][::downsample] }

### Save results
save2matfile_or_pkl(D_downsampled, file_name, obs, params = {},
                    save_mat = save_mat, save_pkl = save_pkl)
