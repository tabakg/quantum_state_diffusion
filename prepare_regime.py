from scipy.io import savemat
import numpy as np
import pickle
import logging
import qutip

from qnet.algebra.operator_algebra import *
from qnet.algebra.circuit_algebra import *
import qnet.algebra.state_algebra as sa

# default are for absorptive bistability
def make_nparams_JC(W,k,g,g0,DD,TT,Cn=10.5,
                 kn=.12, yn=11.3, DDn=0, TTn=0., J = 0.5):


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
    return nparams

def make_system_JC(Nfock_a, Nfock_j):

    ## Make Operators
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

    ## Make SLH Model
    k,g0,g = symbols("kappa, g0,gamma", positive=True)
    DD, TT = symbols("Delta, Theta", real=True)
    W = symbols("Omega")

    L = [sqrt(k)*a,
         sqrt(g)*j]
    H = -I*g0*(a*jp - ad * j) + DD*jz + TT*ad*a
    S = identity_matrix(2)

    slh = SLH(S, L, H).coherent_input(W,0)

    ## Numerical parameters
    a.space.dimension = Nfock_a
    j.space.dimension = Nfock_j

    nparams = make_nparams_JC(W=W,k=k,g=g,g0=g0,DD=DD,TT=TT)

    Hq, Lqs = slh.substitute(nparams).HL_to_qutip()

    ## Observables
    obs = (a, j, jz, a*a, a.dag()*a, a*jp, jp, jx, jy)
    obsq = [o.to_qutip(full_space=slh.space) for o in obs]

    psi0 = qutip.tensor(qutip.basis(Nfock_a,0),qutip.basis(Nfock_j,0)).data
    H = Hq.data
    Ls = [Lq.data for Lq in Lqs]
    obsq_data = [ob.data for ob in obsq]
    return H, psi0, Ls, obsq_data
