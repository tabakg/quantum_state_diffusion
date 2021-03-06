from scipy.io import savemat
import numpy as np
import pickle
import logging
import qutip
from scipy import sparse

from qnet.algebra.operator_algebra import *
from qnet.algebra.circuit_algebra import *
from qnet.circuit_components.displace_cc import Displace
import qnet.algebra.state_algebra as sa

################################################################################
######## One system
################################################################################

######## Jaynes-Cummings System
################################################################################

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
    ## TODO!! when drive is not present...

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
    return H, psi0, Ls, obsq_data, obs

######## Kerr System
######## Bistable regimes based on the examples on pg. 6 in https://arxiv.org/pdf/1402.5983.pdf
######## When adding a new regime, be sure to import in make_quantum_trajectory.py
######## And update when regime is checked.
######## Also make sure to update in hybrid_qsd.py for functions obs_to_ls_...
######## Also, hybrid_qsd.py check the regime for determining the second system.
################################################################################

def make_system_kerr_bistable(Nfock, drive=True):
    if drive:
        params_dict = {"alpha0" : 21.75, "chi" : -10, "Delta" : 100., "kappa_1" : 25, "kappa_2" : 25}
    else:
        params_dict = {"alpha0" : 0.0, "chi" : -10, "Delta" : 100., "kappa_1" : 25, "kappa_2" : 25}
    return make_system_kerr(Nfock, params_dict)

## deprecated regime
# def make_system_kerr_bistable_regime2(Nfock, drive=True):
#     if drive:
#         params_dict = {"alpha0" : 33.25, "chi" : -1.5, "Delta" : 60., "kappa_1" : 25, "kappa_2" : 25}
#     else:
#         params_dict = {"alpha0" : 0., "chi" : -1.5, "Delta" : 60., "kappa_1" : 25, "kappa_2" : 25}
#     return make_system_kerr(Nfock, params_dict)

def make_system_kerr_bistable_regime_chose_drive(Nfock, which_kerr, drive):
    """Most custom options.

    User can pick between bistable kerr A and B (which_kerr).

    """
    assert which_kerr in ['A', 'B']
    if which_kerr == 'A':
        params_dict = {"alpha0" : drive, "chi" : -10, "Delta" : 100., "kappa_1" : 25, "kappa_2" : 25}
    else:
        params_dict = {"alpha0" : drive, "chi" : -1.5, "Delta" : 60., "kappa_1" : 25, "kappa_2" : 25}
    return make_system_kerr(Nfock, params_dict)

def make_system_kerr_qubit(Nfock, drive=True):
    if drive:
        params_dict = {"alpha0" : 10.0, "chi" : -100, "Delta" : 0., "kappa_1" : 0.5, "kappa_2" : 0}
    else:
        params_dict = {"alpha0" : 0.0, "chi" : -100, "Delta" : 0., "kappa_1" : 0.5, "kappa_2" : 0}
    return make_system_kerr(Nfock, params_dict)

def make_system_kerr(Nfock, params_dict):

    # Define Kerr parameters
    chi = symbols("chi", real=True, positive=True)
    Delta = symbols("Delta", real=True)
    kappa_1, kappa_2 = symbols("kappa_1, kappa_2", real=True, positive=True)
    alpha0 = symbols("alpha_0")

    params = {alpha0: params_dict["alpha0"],
              chi: params_dict["chi"],
              Delta: params_dict["Delta"],
              kappa_1: params_dict["kappa_1"],
              kappa_2: params_dict["kappa_2"],
              }

    # Construct Kerr SLH
    a_k = Destroy("k")
    S = -identity_matrix(2)
    L = [sqrt(kappa_1)*a_k, sqrt(kappa_2)*a_k]
    H = Delta*a_k.dag()*a_k + chi/2*a_k.dag()*a_k.dag()*a_k*a_k
    KERR = SLH(S, L, H).toSLH()

    # Add coherent drive
    SYS = KERR << Displace(alpha=alpha0)+cid(1)
    SYS = SYS.toSLH()

    # SYS_no_drive = KERR.toSLH()

    SYS_num = SYS.substitute(params)
    # SYS_num_no_drive = SYS_no_drive.substitute(params)

    SYS_num.space.dimension = Nfock
    # SYS_num_no_drive.space.dimension = Nfock

    Hq, Lqs = SYS_num.HL_to_qutip()

    ## Observables
    obs = [a_k.dag()*a_k, a_k+a_k.dag(), (a_k-a_k.dag())/1j]
    obsq = [o.to_qutip(full_space = SYS_num.space) for o in obs]

    psi0 = qutip.tensor(qutip.basis(Nfock,0)).data
    H = Hq.data
    Ls = [Lq.data for Lq in Lqs]
    obsq_data = [ob.data for ob in obsq]
    return H, psi0, Ls, obsq_data, obs

################################################################################
######## Two systems
################################################################################

######## Jaynes-Cummings System
################################################################################

#### (NOT IMPLEMENTED YET!)
#### TODO: implement this

######## Kerr System
################################################################################

def make_system_kerr_bistable_regime_chose_drive_two_systems(Nfock, which_kerr, custom_drive, drive_second_system=False):

    assert which_kerr in ['A', 'B']
    if which_kerr == 'A':
        params_dict = {"alpha0" : custom_drive, "chi" : -10, "Delta" : 100., "kappa_1" : 25, "kappa_2" : 25}
    else:
        params_dict = {"alpha0" : custom_drive, "chi" : -1.5, "Delta" : 60., "kappa_1" : 25, "kappa_2" : 25}

    return make_system_kerr_two_systems(Nfock, params_dict, drive_second_system=drive_second_system)

def make_system_kerr_bistable_two_systems(Nfock, drive_second_system=False):
    params_dict = {"alpha0" : 21.75, "chi" : -10, "Delta" : 100., "kappa_1" : 25, "kappa_2" : 25}
    return make_system_kerr_two_systems(Nfock, params_dict, drive_second_system=drive_second_system)

def make_system_kerr_qubit_two_systems(Nfock, drive_second_system=False):
    params_dict = {"alpha0" : 10.0, "chi" : -100, "Delta" : 0., "kappa_1" : 0.5, "kappa_2" : 0}
    return make_system_kerr_two_systems(Nfock, params_dict, drive_second_system=drive_second_system)

def make_system_empty_then_kerr(Nfock, which_kerr, custom_drive,
                                drive_second_system=False, S_mult=-1.):

    assert which_kerr in ['A', 'B']
    if which_kerr == 'A':
        params_dict = {"alpha0" : custom_drive, "chi" : -10, "Delta" : 100., "kappa_1" : 25, "kappa_2" : 25}
    else:
        params_dict = {"alpha0" : custom_drive, "chi" : -1.5, "Delta" : 60., "kappa_1" : 25, "kappa_2" : 25}

    # Define Kerr parameters for system 2:
    chi = symbols("chi", real=True, positive=True)
    Delta = symbols("Delta", real=True)
    kappa_1, kappa_2 = symbols("kappa_1, kappa_2", real=True, positive=True)
    alpha0 = symbols("alpha_0")

    params = {alpha0: params_dict["alpha0"],
              chi: params_dict["chi"],
              Delta: params_dict["Delta"],
              kappa_1: params_dict["kappa_1"],
              kappa_2: params_dict["kappa_2"],
              }

    ## Empty system 1:
    a_e = Destroy("e") ## fictitious operator for empty system 1
    S_empty = identity_matrix(2) * S_mult
    L_empty = [0., 0.]
    H_empty = a_e.dag()*a_e ## Need to feed in operator, will become zero for dim == 1...
    EMPTY = SLH(S_empty, L_empty, H_empty).toSLH()

    # Add coherent drive
    SYS_EMPTY = EMPTY << Displace(alpha=alpha0)+cid(1)

    # Construct H_num and L_num for a driven system
    SYS_EMPTY = SYS_EMPTY.toSLH()
    SYS_EMPTY_num = SYS_EMPTY.substitute(params)
    SYS_EMPTY_num.space.dimension = 1
    H1, L1s = SYS_EMPTY_num.HL_to_qutip()

    # Construct Kerr SLH
    a_k = Destroy("k")
    S = identity_matrix(2) * S_mult
    L = [sqrt(kappa_1)*a_k, sqrt(kappa_2)*a_k]
    H = Delta*a_k.dag()*a_k + chi/2*a_k.dag()*a_k.dag()*a_k*a_k
    KERR = SLH(S, L, H).toSLH()

    # Construct H_num and L_num for a driven system
    SYS = KERR.toSLH()
    SYS_num = SYS.substitute(params)
    SYS_num.space.dimension = Nfock
    H2, L2s = SYS_num.HL_to_qutip()
    obs = [a_k.dag()*a_k, a_k+a_k.dag(), (a_k-a_k.dag())/1j]
    obsq = [o.to_qutip(full_space = SYS_num.space) for o in obs]
    psi0 = qutip.tensor(qutip.basis(Nfock,0)).data
    obsq_data = [ob.data for ob in obsq]

    return H1.data, H2.data, psi0, [L.data for L in L1s], [L.data for L in L2s], obsq_data, obs

def make_system_kerr_two_systems(Nfock, params_dict, drive_second_system=False, S_mult=-1.):

    # Define Kerr parameters
    chi = symbols("chi", real=True, positive=True)
    Delta = symbols("Delta", real=True)
    kappa_1, kappa_2 = symbols("kappa_1, kappa_2", real=True, positive=True)
    alpha0 = symbols("alpha_0")

    params = {alpha0: params_dict["alpha0"],
              chi: params_dict["chi"],
              Delta: params_dict["Delta"],
              kappa_1: params_dict["kappa_1"],
              kappa_2: params_dict["kappa_2"],
              }

    # Construct Kerr SLH
    a_k = Destroy("k")
    S = identity_matrix(2) * S_mult
    L = [sqrt(kappa_1)*a_k, sqrt(kappa_2)*a_k]
    H = Delta*a_k.dag()*a_k + chi/2*a_k.dag()*a_k.dag()*a_k*a_k
    KERR = SLH(S, L, H).toSLH()

    # Add coherent drive
    SYS = KERR << Displace(alpha=alpha0)+cid(1)

    # Construct H_num and L_num for a driven system
    SYS = SYS.toSLH()
    SYS_num = SYS.substitute(params)
    SYS_num.space.dimension = Nfock
    H_num, L_num = SYS_num.HL_to_qutip()

    # The first system is always driven
    H1, L1s = H_num, L_num

    ## Make the second H_num and L_num. It may or may not be driven.
    if drive_second_system:
        H2, L2s = H_num, L_num
    else:
        # H_num and L_num for non-driven system
        SYS_no_drive = KERR.toSLH()
        SYS_num_no_drive = SYS_no_drive.substitute(params)
        SYS_num_no_drive.space.dimension = Nfock
        H_no_drive_num, L_no_drive_num = SYS_num_no_drive.HL_to_qutip()
        H2, L2s = H_no_drive_num, L_no_drive_num

    ## Observables
    obs = [a_k.dag()*a_k, a_k+a_k.dag(), (a_k-a_k.dag())/1j]
    obsq = [o.to_qutip(full_space = SYS_num.space) for o in obs]

    I = np.eye(Nfock)

    ## Extend the operators to the whole space.
    obsq_data_kron = ([sparse.csr_matrix(np.kron(ob.data.todense(), I)) for ob in obsq]
     + [sparse.csr_matrix(np.kron(I, ob.data.todense())) for ob in obsq])

    obs_two_systems = np.concatenate([[a_k.dag()*a_k, a_k+a_k.dag(), (a_k-a_k.dag())/1j]
        for a_k in [Destroy('1'), Destroy('2')]])

    psi0 = sparse.csr_matrix(([1] + [0]*(Nfock**2-1)),dtype=np.complex128).T

    return H1.data, H2.data, psi0, [L.data for L in L1s], [L.data for L in L2s], obsq_data_kron, obs_two_systems
