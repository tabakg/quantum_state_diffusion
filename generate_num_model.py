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
import sys
import argparse
from utils import preprocess_operators
import logging

from quantum_state_diffusion import (
    qsd_solve,
    qsd_solve_two_systems
)

from utils import (
    print_params
)

from prepare_regime import (
    make_system_JC,
    make_system_kerr_bistable,
    make_system_kerr_bistable_regime_chose_drive,
    make_system_kerr_qubit,
    ## make_system_JC_two_systems, ## Not yet implemented
    make_system_kerr_bistable_two_systems,
    make_system_kerr_qubit_two_systems,
    make_system_empty_then_kerr,
    make_system_kerr_bistable_regime_chose_drive_two_systems,
)

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


def gen_num_system(json_file_dir,
                   H,
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
        json_file_dir: string, where to output json file.
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


def gen_num_system_two_systems(json_file_dir,
                               H1,
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
        json_file_dir: string, where to output json file.
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

def make_one_system_example(json_file_dir):

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
    gen_num_system(json_file_dir,
                   H,
                   psi0,
                   duration,
                   delta_t,
                   Ls,
                   "itoImplicitEuler",
                   obsq=obsq_data,
                   downsample=downsample,
                   ntraj=ntraj,
                   seed=seed)

def make_two_system_example(json_file_dir):

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
    gen_num_system_two_systems(json_file_dir,
                               H1,
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
                               ops_on_whole_space=False,
                               ntraj=ntraj,
                               seed=seed)


def get_parser():
    '''get_parser returns the arg parse object, for use by an external application (and this script)
    '''
    parser = argparse.ArgumentParser(
    description="generating trajectories using quantum state diffusion")


    ################################################################################
    # General Simulation Parameters
    ################################################################################

    # Seed
    parser.add_argument("--seed",
                        dest='seed',
                        help="Seed to set for the simulation.",
                        type=int,
                        default=1)

    # Number of trajectories
    parser.add_argument("--ntraj",
                        dest='ntraj',
                        help="number of trajectories, should be kept at 1 if run via slurm",
                        type=int,
                        default=1)

    # Duration
    parser.add_argument("--duration",
                        dest='duration',
                        help="Duration (iterations = duration / divided by delta_t)",
                        type=float,
                        default=10)

    # Delta T
    parser.add_argument("--delta_t",
                        dest='delta_t',
                        help="Parameter delta_t",
                        type=float,
                        default=2e-3)

    # How much to downsample results
    parser.add_argument("--downsample",
                        dest='downsample',
                        help="How much to downsample results",
                        type=int,
                        default=1)

    # Simulation method
    parser.add_argument("--sdeint_method_name",
                        dest='sdeint_method_name',
                        help="Which simulation method to use from sdeint packge.",
                        type=str,
                        default="")

    ################################################################################
    # System-specific parameters
    ################################################################################

    # regime
    parser.add_argument("--regime",
                        dest='regime',
                        help="Type of system or regime."
                             "Can be 'absorptive_bistable', 'kerr_bistable', or 'kerr_qubit'",
                        type=str,
                        default='absorptive_bistable')

    # num_systems
    parser.add_argument("--num_systems",
                        dest='num_systems',
                        help="Number of system in the network. Can currently be 1 or 2",
                        type=int,
                        default=1)

    # Nfock_a
    parser.add_argument("--Nfock_a",
                        dest='Nfock_a',
                        help="Number of fock states in each cavity",
                        type=int,
                        default=50)

    # Nfock_j
    parser.add_argument("--Nfock_j",
                        dest='Nfock_j',
                        help="Dimensionality of atom states"
                             "Used only if using a Jaynes-Cummings model",
                        type=int,
                        default=2)

    ################################################################################
    # Parameters that apply only for the two-system case
    ################################################################################

    # R
    parser.add_argument("--R",
                        dest='R',
                        help="Reflectivity of the beamsplitter in the two-system case.",
                        type=float,
                        default=0.)

    # eps
    parser.add_argument("--eps",
                        dest='eps',
                        help="Amplification of the classical signal when using partially classical transmission.",
                        type=float,
                        default=0.)

    # noise_amp
    parser.add_argument("--noise_amp",
                        dest='noise_amp',
                        help="Artificial amplification of the measurement-feedback noise."
                             "This is a non-physical term that is useful for understanding the effects of noise.",
                        type=float,
                        default=1.)

    # lambda
    parser.add_argument("--lambda",
                        dest='lambd',
                        help="Kalman filtering parameter for classical transmission.",
                        type=float,
                        default=0.)

    # trans_phase
    parser.add_argument("--trans_phase",
                        dest='trans_phase',
                        help="Additional phase term added between the two systems.",
                        type=float,
                        default=1.)

    # drive_second_system
    parser.add_argument("--drive_second_system",
                        dest='drive_second_system',
                        help="Whether the second system is independently driven.",
                        type=bool,
                        default=False)

    ################################################################################
    # Output Variables
    ################################################################################

    parser.add_argument("--output_dir",
                        dest='output_dir',
                        type=str,
                        help="Output file location (full path).",
                        default="/scratch/users/tabakg/qsd_output/json_spec/")


    # Does the user want to quiet output?
    parser.add_argument("--quiet",
                        dest='quiet',
                        action="store_true",
                        help="Turn off logging (debug and info)",
                        default=False)
    return parser

def main():
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        print("Unable to get parser, exiting now...")
        sys.exit(0)

    ############################################################################
    #### Sample output directory and file name, tested locally.
    # out_dir="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications"
    # json_file_name="tmp_file.json"
    # json_file_dir=os.path.join(out_dir, json_file_name)

    # make_one_system_example(json_file_dir)
    # make_two_system_example(json_file_dir)
    ############################################################################

    ############################################################################
    #### Set up commands from parser
    #### Sample call from command line
    # python /scratch/users/tabakg/qsd_dev/generate_num_model.py --output_dir '/scratch/users/tabakg/qsd_output/json_spec/' --Nfock_a 30 \
    # --seed 1 --regime 'kerr_bistableA21.75' --num_systems 2 --delta_t 1e-05 --duration 0.2 --downsample 100 \
    # --sdeint_method_name 'itoImplicitEuler' --R 1.0 --eps 1.0 --noise_amp 1.0 --lambda 0.999
    ############################################################################

    params = dict()
    ntraj = params['Ntraj'] = args.ntraj
    seed = params['seed'] = args.seed
    duration = params['duration'] = args.duration
    delta_t = params['delta_t'] = args.delta_t
    Nfock_a = params['Nfock_a'] = args.Nfock_a
    Nfock_j = params['Nfock_j'] = args.Nfock_j
    downsample = params['downsample'] = args.downsample
    Regime = params['regime'] = args.regime
    num_systems = params['num_systems'] = args.num_systems
    drive_second_system = params['drive_second_system'] = args.drive_second_system

    if args.sdeint_method_name == "":
        logging.info("sdeint_method_name not set. Using itoEuler as a default.")
        sdeint_method_name = params['sdeint_method_name'] = "itoEuler"
    else:
        sdeint_method_name = params['sdeint_method_name'] = args.sdeint_method_name

    R = params['R'] = args.R
    eps = params['eps'] = args.eps
    noise_amp = params['noise_amp'] = args.noise_amp
    lambd = params['lambd'] = args.lambd
    trans_phase = params['trans_phase'] = args.trans_phase

    # Does the user want to print verbose output?
    quiet = args.quiet

    if not quiet:
        print_params(params=params)

    #### output directory and file name, generated from inputs

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
                                 eps,
                                 noise_amp,
                                 lambd,
                                 trans_phase,
                                 drive_second_system)

    json_file_name = "json_spec_" + param_str + ".json"

    json_file_dir=os.path.join(args.output_dir, json_file_name)
    print("output directory is ", json_file_dir)

    tspan = np.arange(0,duration,delta_t)

    if num_systems == 1:

        if Regime == "absorptive_bistable":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_JC(Nfock_a, Nfock_j)
        elif Regime == "kerr_bistable":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_kerr_bistable(Nfock_a)
        elif Regime[:len("kerr_bistable")] == "kerr_bistable": ##inputs in this case are e.g. kerr_bistableA33.25_...
            which_kerr = Regime[len("kerr_bistable")] ## e.g. A in kerr_bistableA33.25_
            custom_drive = float(Regime[len("kerr_bistableA"):]) ## e.g. 33.25 in kerr_bistableA33.25
            logging.info("Regime is set to %s, with custom drive %s" %(Regime, custom_drive))
            H, psi0, Ls, obsq_data, obs_names = make_system_kerr_bistable_regime_chose_drive(Nfock_a, which_kerr, custom_drive)
        elif Regime == "kerr_qubit":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_kerr_qubit(Nfock_a)
        else:
            logging.error("Unknown regime, %s, or not implemented yet.", Regime)
            raise ValueError("Unknown regime, or not implemented yet.")

        gen_num_system(json_file_dir,
                       H,
                       psi0,
                       duration,
                       delta_t,
                       Ls,
                       sdeint_method_name,
                       obsq=obsq_data,
                       downsample=downsample,
                       ntraj=ntraj,
                       seed=seed)

    elif num_systems == 2:

        if Regime == "absorptive_bistable":
            logging.info("Regime is set to %s", Regime)
            H1, H2, psi0, L1s, L2s, obsq_data, obs_names = make_system_JC_two_systems(Nfock_a, Nfock_j, drive_second_system)
        elif Regime == "kerr_bistable":
            logging.info("Regime is set to %s", Regime)
            H1, H2, psi0, L1s, L2s, obsq_data, obs_names = make_system_kerr_bistable_two_systems(Nfock_a, drive_second_system)
        elif Regime == "kerr_qubit":
            logging.info("Regime is set to %s", Regime)
            H1, H2, psi0, L1s, L2s, obsq_data, obs_names = make_system_kerr_qubit_two_systems(Nfock_a, drive_second_system)
        elif Regime[:len("empty_then_kerr")] == 'empty_then_kerr': ##e.g. empty_then_kerrA33.25
            which_kerr = Regime[len("empty_then_kerr")] ## e.g. A in empty_then_kerrA33.25_
            custom_drive = float(Regime[len("empty_then_kerrA"):]) ## e.g. 33.25 in empty_then_kerrA33.25
            logging.info("Regime is set to %s, with custom drive %s" %(Regime, custom_drive))
            H1, H2, psi0, L1s, L2s, obsq_data, obs_names = make_system_empty_then_kerr(Nfock_a, which_kerr, custom_drive)
        elif Regime[:len("kerr_bistable")] == "kerr_bistable": ##inputs in this case are e.g. kerr_bistableA33.25_...
            which_kerr = Regime[len("kerr_bistable")] ## e.g. A in kerr_bistableA33.25_
            custom_drive = float(Regime[len("kerr_bistableA"):]) ## e.g. 33.25 in kerr_bistableA33.25
            logging.info("Regime is set to %s, with custom drive %s" %(Regime, custom_drive))
            H1, H2, psi0, L1s, L2s, obsq_data, obs_names = make_system_kerr_bistable_regime_chose_drive_two_systems(Nfock_a, which_kerr, custom_drive)
        else:
            logging.error("Unknown regime, %s, or not implemented yet.", Regime)
            raise ValueError("Unknown regime, or not implemented yet.")

        gen_num_system_two_systems(json_file_dir,
                                   H1,
                                   H2,
                                   psi0,
                                   duration,
                                   delta_t,
                                   L1s,
                                   L2s,
                                   R,
                                   eps,
                                   noise_amp,
                                   lambd,
                                   sdeint_method_name,
                                   trans_phase=None,
                                   obsq=obsq_data,
                                   downsample=downsample,
                                   ops_on_whole_space=False,
                                   ntraj=ntraj,
                                   seed=seed)

    else: ## num_systems not equal to 1 or 2
        logging.error("Unknown num_systems, %s, or not implemented yet.", num_systems)
        raise ValueError("Unknown num_systems, or not implemented yet.")

if __name__ == '__main__':
    main()
