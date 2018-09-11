'''
Generate numerical model

Author: Gil Tabak
Date: July 30, 2018

This script generates a numerical model that can be used to simulate a system.
The inputs are purposely similar to qutip functions like mcsolve to make
integration easier. The functions here return JSON formatted lists.

'''

import numpy as np
import numpy.linalg as la
from scipy import sparse
import json
import os
from utils import preprocess_operators

out_dir="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications"
json_file_name="tmp_file.json"
json_file_dir=os.path.join(out_dir, json_file_name)

def split_complex(lst):
    return {"real": [el.real for el in lst],
            "imag": [el.imag for el in lst]}

def sparse_op_to_json(P):
    P_diags = sparse.dia_matrix(P)
    offsets = [int(el) for el in P_diags.offsets]
    data = [list(arr) for arr in (P_diags.data)]

    ## filter out entries with very small norm
    sums = [sum(abs(np.array(el))) for el in data]
    which_sums = [True if s > 1e-10 else False for s in sums]
    offsets = [offsets[i] for i,s in enumerate(which_sums) if s]
    data = [data[i] for i,s in enumerate(which_sums) if s]
    data = [split_complex(lst) for lst in data]

    return {"offsets": offsets, "data": data}


def gen_num_system(H,
                   psi0,
                   duration,
                   delta_t,
                   Ls,
                   sdeint_method,
                   obsq=None,
                   downsample=1,
                   ntraj=1,
                   seed=1):
    '''Given the inputs for one system, generate and save a numerical json file.

    Args:
        H: NxN csr matrix, dtype = complex128
            Hamiltonian.
        psi0: Nx1 csr matrix, dtype = complex128
            input state.
        duration: float.
            Duration of simulation
        delta_t: float.
            duration of a single timestep.
        Ls: list of NxN csr matrices, dtype = complex128
            System-environment interaction terms (Lindblad terms).
        sdeint_method (Optional) SDE solver method:
            Which SDE solver to use. Default is sdeint.itoSRI2.
        obsq (optional): list of NxN csr matrices, dtype = complex128
            Observables for which to generate trajectory information.
            Default value is None (no observables).
        downsample: optional, integer to indicate how frequently to save values.
        ntraj (optional): int
            number of trajectories.
        seed (optional): int
            Seed for random noise.
        implicit_type (optional): string
            Type of implicit solver to use if the solver is implicit.
    '''

    ## Check dimensions of inputs. These should be consistent with qutip Qobj.data.
    N = psi0.shape[0]
    if psi0.shape[1] != 1:
        raise ValueError("psi0 should have dimensions Nx1.")
    a,b = H.shape
    if a != N or b != N:
        raise ValueError("H should have dimensions NxN (same size as psi0).")
    for L in Ls:
        a,b = L.shape
        if a != N or b != N:
            raise ValueError("Every L should have dimensions NxN (same size as psi0).")

    ## Determine seeds for the SDEs
    if type(seed) is list or type(seed) is tuple:
        assert len(seed) == ntraj
        seeds = seed
    elif type(seed) is int or seed is None:
        np.random.seed(seed)
        seeds = [np.random.randint(4294967295) for _ in range(ntraj)]
    else:
        raise ValueError("Unknown seed type.")

    ## H_eff is the effective term appearing in the equation of motion
    ## NOT the effective Hamiltonian
    H_eff = -1j*H - 0.5*sum(L.H*L for L in Ls)

    H_eff_json = sparse_op_to_json(H_eff)
    if Ls:
        Ls_json = [sparse_op_to_json(L) for L in Ls]
    else:
        Ls_json = []
    if obsq:
        obsq_json = [sparse_op_to_json(ob) for ob in obsq]
    else:
        obsq_json = []

    psi0_list = split_complex(list(np.asarray(psi0.todense()).T[0]))

    data = {"num_systems": 1,
            "H_eff": H_eff_json,
            "Ls": Ls_json,
            "psi0": psi0_list,
            "duration": duration,
            "delta_t": delta_t,
            "sdeint_method": sdeint_method,
            "obsq": obsq_json,
            "downsample": downsample,
            "ntraj": ntraj,
            "seeds": seeds}

    with open(json_file_dir, 'w') as outfile:
        json.dump(data, outfile)


def gen_num_system_two_systems(H1,
                               H2,
                               psi0,
                               duration,
                               delta_t,
                               L1s,
                               L2s,
                               R,
                               eps,
                               n,
                               lambd,
                               sdeint_method,
                               trans_phase=None,
                               obsq=None,
                               downsample=1,
                               ops_on_whole_space = False,
                               ntraj=1,
                               seed=1):

    '''Given the inputs for two systems, writes the numerical model to json.

    Args:
        H1: N1xN1 csr matrix, dtype = complex128
            Hamiltonian for system 1.
        H2: N2xN2 csr matrix, dtype = complex128
            Hamiltonian for system 2.
        psi0: Nx1 csr matrix, dtype = complex128
            input state.
        duration: float.
            Duration of simulation
        delta_t: float.
            duration of a single timestep.
        L1s: list of N1xN1 csr matrices, dtype = complex128
            System-environment interaction terms (Lindblad terms) for system 1.
        L2s: list of N2xN2 csr matrices, dtype = complex128
            System-environment interaction terms (Lindblad terms) for system 2.
        R: float
            reflectivity used to separate the classical versus coherent
            transmission
        eps: float
            The multiplier by which the classical state displaces the coherent
            state
        n: float
            Scalar to multiply the measurement feedback noise
        lambd: float
            Kalman coefficient for classical transmission filtering.
        sdeint_method (Optional) SDE solver method:
            Which SDE solver to use. Default is sdeint.itoSRI2.
        obsq (optional): list of NxN csr matrices, dtype = complex128
            Observables for which to generate trajectory information.
            Default value is None (no observables).
        downsample: optional, integer to indicate how frequently to save values.
        ops_on_whole_space (optional): Boolean
            whether the Given L and H operators have been defined on the whole
            space or individual subspaces.
        ntraj (optional): int
            number of trajectories.
        seed (optional): int
            Seed for random noise.
    '''

    ## Check dimensions of inputs. These should be consistent with qutip Qobj.data.
    N = psi0.shape[0]
    if psi0.shape[1] != 1:
        raise ValueError("psi0 should have dimensions Nx1.")

    ## Determine seeds for the SDEs
    if type(seed) is list or type(seed) is tuple:
        assert len(seed) == ntraj
        seeds = seed
    elif type(seed) is int or seed is None:
        np.random.seed(seed)
        seeds = [np.random.randint(4294967295) for _ in range(ntraj)]
    else:
        raise ValueError("Unknown seed type.")

    H1, H2, L1s, L2s = preprocess_operators(H1, H2, L1s, L2s, ops_on_whole_space)

    H_eff = -1j*H1 - 0.5*sum(L.H*L for L in L1s) -1j*H2 - 0.5*sum(L.H*L for L in L2s)
    H_eff_json = sparse_op_to_json(H_eff)

    Ls = [L1s[0], L2s[0]] + L1s[1:] + L2s[1:]
    Ls_json = [sparse_op_to_json(L) for L in Ls]

    L2_dag = L2s[0].H
    L2_dag_json = sparse_op_to_json(L2_dag)

    L2_dag_L1 = L2s[0].H  * L1s[0]
    L2_dag_L1_json = sparse_op_to_json(L2_dag_L1)

    if obsq:
        obsq_json = [sparse_op_to_json(ob) for ob in obsq]
    else:
        obsq_json = []

    psi0_list = split_complex(list(np.asarray(psi0.todense()).T[0]))

    T = np.sqrt(1 - R**2)

    if trans_phase is not None:
        eps *= trans_phase
        T *= trans_phase

    data = {"num_systems": 2,
            "H_eff": H_eff_json,
            "Ls": Ls_json,
            "L2_dag": L2_dag_json,
            "L2_dag_L1": L2_dag_L1_json,
            "psi0": psi0_list,
            "duration": duration,
            "delta_t": delta_t,
            "sdeint_method": sdeint_method,
            "obsq": obsq_json,
            "downsample": downsample,
            "ntraj": ntraj,
            "seeds": seeds,
            "R": R,
            "T": T,
            "eps": eps,
            "n": n,
            "lambda": lambd}

    with open(json_file_dir, 'w') as outfile:
        json.dump(data, outfile)

def make_one_system_example():

    from prepare_regime import make_system_kerr_bistable_regime_chose_drive

    ## generic parameters
    dim = 50
    drive = 21.75
    duration = 0.2
    delta_t = 1e-5
    sdeint_method = "itoImplicitEuler"
    downsample=100
    ntraj = 1
    seed = 1

    H, psi0, Ls, obsq_data, obs = make_system_kerr_bistable_regime_chose_drive(dim, 'A', drive)
    gen_num_system(H,
                   psi0,
                   duration,
                   delta_t,
                   Ls,
                   "itoImplicitEuler",
                   obsq=obsq_data,
                   downsample=downsample,
                   ntraj=ntraj,
                   seed=seed)

def make_two_system_example():
    from prepare_regime import make_system_kerr_bistable_regime_chose_drive_two_systems

    ## generic parameters
    dim = 10
    drive = 21.75
    duration = 0.2
    delta_t = 1e-5
    sdeint_method = "itoImplicitEuler"
    downsample=100
    ntraj = 1
    seed = 1

    ## two-system specific parameters
    R = 0.0
    eps = 0.5
    n = 1.0
    lambd = 0.9999

    H1, H2, psi0, L1s, L2s, obsq_data_kron, _ = make_system_kerr_bistable_regime_chose_drive_two_systems(dim, 'A', drive, drive_second_system=False)
    gen_num_system_two_systems(H1,
                               H2,
                               psi0,
                               duration,
                               delta_t,
                               L1s,
                               L2s,
                               R,
                               eps,
                               n,
                               lambd,
                               sdeint_method,
                               trans_phase=None,
                               obsq=obsq_data_kron,
                               downsample=downsample,
                               ops_on_whole_space = False,
                               ntraj=ntraj,
                               seed=seed)


if __name__ == "__main__":

    # make_one_system_example()
    make_two_system_example()
