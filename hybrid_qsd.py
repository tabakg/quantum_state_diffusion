import argparse
import logging
import os
import pickle
import sys

import math
import numpy as np
import numpy.linalg as la
import sdeint
from scipy import sparse
from time import time

diffusion_maps = '/Users/gil/Desktop/test/diffusion_map'
markov_models = '/Users/gil/Desktop/test/markov_models'
qsd_dev = '/Users/gil/Google Drive/repos/quantum_state_diffusion'

sys.path.append(qsd_dev)
sys.path.append(markov_models)
sys.path.append(diffusion_maps)

from utils import get_params
import markov_model

from prepare_regime import (
    make_system_JC,
    make_system_kerr_bistable,
    make_system_kerr_qubit,
    ## make_system_JC_two_systems, ## Not yet implemented
    make_system_kerr_bistable_two_systems,
    make_system_kerr_qubit_two_systems,
)

from quantum_state_diffusion import (
    complex_to_real_vector,
    complex_to_real_matrix,
    individual_term,
    classical_trans,
    insert_conj,
    dim_check)

SDEINT_METHODS = {"itoEuler": sdeint.itoEuler,
                  "itoSRI2": sdeint.itoSRI2,
                  ## "itoMilstein": itoMilstein.itoSRI2,
                  "numItoMilstein": sdeint.numItoMilstein,
                  "itoImplicitEuler": sdeint.itoImplicitEuler,
                  "itoQuasiImplicitEuler": sdeint.itoQuasiImplicitEuler}
IMPLICIT_METHODS=["itoImplicitEuler"]

logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)

def get_parser():
    '''get_parser returns the arg parse object, for use by an external application (and this script)
    '''
    parser = argparse.ArgumentParser(
    description="Generating diffusion maps of trajectories.")

    ################################################################################
    # General Simulation Parameters
    ################################################################################

    parser.add_argument("--seed",
                        dest='seed',
                        help="Seed for generating trajectory of system 1 and "
                             "stochastic terms of system 2.",
                        type=int,
                        default=1)

    parser.add_argument("--slow_down",
                        dest='slow_down',
                        help="How much to slow down the generated trajectories. "
                             "Roughly corresponds to the inverse of the sampling "
                             "rate to generate the Markov model.",
                        type=int,
                        default=1000)

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
                        default=30)

    # Delta T
    parser.add_argument("--delta_t",
                        dest='deltat',
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
                        help="Type of system or regime for system 2. "
                             "Can be 'absorptive_bistable', 'kerr_bistable', or 'kerr_qubit'. "
                             "Default would use the same as system 1.",
                        type=str,
                        default=None)

    # Nfock_a
    parser.add_argument("--Nfock_a",
                        dest='nfocka',
                        help="Number of fock states in each cavity. "
                             "Default is same as system 1",
                        type=int,
                        default=None)

    # Nfock_j
    parser.add_argument("--Nfock_j",
                        dest='nfockj',
                        help="Dimensionality of atom states"
                             "Used only if using a Jaynes-Cummings model"
                             "Default is same as system 1",
                        type=int,
                        default=None)

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

    parser.add_argument("--markov_file",
                        dest='markov_file',
                        type=str,
                        help="Input file to use. Should be an output of "
                             "markov_model_builder containing the proper "
                             "pickled data",
                        default=None)

    parser.add_argument("--diffusion_file",
                        dest='diffusion_file',
                        type=str,
                        help="diffusion map file.",
                        default=None)

    parser.add_argument("--output_dir",
                        dest='outdir',
                        type=str,
                        help="Output folder",
                        default=None)

    # Save to pickle?
    parser.add_argument("--save2pkl",
                        dest='save2pkl',
                        action="store_true",
                        help="Save pickle file to --output_dir",
                        default=False)

    # Save to mat?
    parser.add_argument("--save2mat",
                        dest='save2mat',
                        action="store_true",
                        help="Save .mat file to --output_dir",
                        default=False)
    return parser

## TODO: make little functions like this for other regimes...
def obs_to_ls_kerr_bistable(obs):
    L_params = {"kappa_1" : 25, "kappa_2" : 25}
    x, p = obs[1:3] ## assume specific structure in obs to be [_, a_k+a_k.dag(), (a_k-a_k.dag())/1j, ...]
    a_k = (x+1j*p)/2.
    L = [np.sqrt(L_params['kappa_1'])*a_k,
         np.sqrt(L_params['kappa_2'])*a_k]
    return L

class hybrid_drift_diffusion_holder(object):
    '''
    This object is used to generate the equation of motion of two systems,
    where transmission from system 1 to system 2 occurs via both a classical
    as well as a 'quantum' channel. Here this is assumed to be done by splitting
    the output of system 1 with a beamsplitter, and making a heterodyne
    measurement on one of the outputs. However, instead of a full quantum
    simulation for both systems, we use a reduced model for system 1. The
    reduced model should be able to produce trajectories of the observables
    l^{(i)}_1 = <psi_1, L^{(i)}_1, psi_1>. Here psi_1 is the state of system 1
    in its Hilbert space, and each L^{(i)}_1 is an entry in the vector of
    Lindblad operators associated with system 1.

    Each l^{(i)}_1 is a complex-valued entry, which contains both x and p
    information about the state of system 1.

    Since we do not have access to the full operator L1, in the formulate for
    'quantum' transmission we replace L1 by l1. This then differs from classical
    transmission only in the measurement feedback term.

    For system 2 only:

    We include a way to update L*psi and l = <psi,L,psi> when t changes.
    This makes the code somewhat more efficient since these values are used
    both for the drift f and the diffusion G terms, and don't have to be
    recomputed each time.

    Each psi used as an input should be a complex-valued array of length (d,)

    The outputs generated f and G are complex-valued arrays of dimensions
    (d,) and (d,m), respectively.

    d: complex-valued dimension of the space
    m: number of complex-valued noise terms.
    '''
    def __init__(self, l1s_reduced, H2, L2s, R, eps, n, tspan, trans_phase=None):

        self.t_old = min(tspan) - 1.

        assert 0 <= R <= 1
        self.R = R
        self.T = np.sqrt(1 - R**2)
        self.eps = eps
        self.n = n

        if trans_phase is not None:
            self.eps *= trans_phase
            self.T *= trans_phase

        self.H2 = H2
        self.L2s = L2s

        dim_check(H2, L2s)

        for ls in l1s_reduced:
            if len(tspan) > len(ls):
                raise ValueError("An item in l1s_reduced provided (with len=%s) not long enough for tspan (len=%s)"
                                 %(len(ls), len(tspan))
                                )
        self.l1s_reduced = l1s_reduced
        self.l1s_index = -1

        self.l1s = None
        self.L2psis = None
        self.l2s = None

    def update_Lpsis_and_ls(self, psi, t):
        '''Updates Lpsis and ls.

        If t is different than t_old, update Lpsis, ls, and t_old.
        Otherwise, do nothing.

        Args:
            psi0: N2x1 csr matrix, dtype = complex128
                input state.
            t: dtype = float
        '''
        if t != self.t_old:
            self.L2psis = [L2.dot(psi) for L2 in self.L2s]
            self.l1s_index += 1
            self.l1s = [arr[self.l1s_index] for arr in self.l1s_reduced]
            self.l2s = [L2psi.dot(psi.conj()) for L2psi in self.L2psis]
            self.t_old = t

    def f_normalized(self, psi, t):
        '''Computes drift f.

        Assumes two systems. Computes an individual term for each,
        as well as a classical transmission and a quantum transmission term.

        Args:
            psi0: N2x1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: array of shape (d,)
            to define the deterministic part of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)

        return (individual_term(self.H2, self.L2s, self.l2s, self.L2psis, psi)
            + (self.eps * self.R + self.T) * classical_trans(
                self.L2s[0], self.l1s[0], self.l2s[0], self.L2psis[0], psi))

    def G_normalized(self, psi, t):
        '''Computes diffusion G.

        Because of the measurement feedback, we require an additional
        noise term dW_2^*. So instead of the noise terms we would expect
        with a total of N ports, [dW_1, ..., dW_m]^T, the appropriate noise
        to use for this term are [dW_1, dW_2, dW_2^*, dW_3, ..., dW_m]^T
        (note: There are a total of m+1 noise terms!). If we decompose
        this term into real in imaginary components, the noise terms to
        use will have form:

            [Re(dW_1), Re(dW_2), +Re(dW_2), Re(dW_3), ..., Re(dW_m),
             Im(dW_1), Im(dW_2), -Im(dW_2), Im(dW_3), ..., Im(dW_m)]^T.

        Args:
            psi0: N2x1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: returning an array of shape (d, m)
            to define the noise coefficients of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)
        L2_cent = [L2psi - l2*psi for L2psi,l2 in zip(self.L2psis, self.l2s)]
        N2 = self.n * self.eps * (-self.L2s[0].H.dot(psi) )
        N2_conj = (self.n * self.eps * (self.L2s[0].dot(psi)))
        return np.vstack([N2, N2_conj] + L2_cent[1:]).T

def qsd_solve_hybrid_system(H2,
                            psi0,
                            tspan,
                            l1s_reduced,
                            L2s,
                            R,
                            eps,
                            n,
                            sdeint_method,
                            trans_phase=None,
                            obsq=None,
                            normalize_state=True,
                            downsample=1,
                            multiprocessing = False,
                            ntraj=1,
                            processes=8,
                            seed=1,
                            implicit_type = None):
    '''
    Args:
        H2: N2xN2 csr matrix, dtype = complex128
            Hamiltonian for system 2.
        psi0: N2x1 csr matrix, dtype = complex128
            input state.
        tspan: numpy array, dtype = float
            Time series of some length T.
        l1s_reduced: a list of arrays, each of which is the time series
            for one of expectations of each L1.
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
        sdeint_method (Optional) SDE solver method:
            Which SDE solver to use. Default is sdeint.itoSRI2.
        obsq (optional): list of N2xN2 csr matrices, dtype = complex128
            Observables for which to generate trajectory information.
            Default value is None (no observables).
        normalize_state (optional): Boolean
            Whether to numerically normalize the equation at each step.
        downsample: optional, integer to indicate how frequently to save values.
        multiprocessing (optional): Boolean
            Whether or not to use multiprocessing
        ntraj (optional): int
            number of trajectories.
        processes (optional): int
            number of processes. If processes == 1, don't use multiprocessing.
        seed (optional): int
            Seed for random noise.

    Returns:
        A dictionary with the following keys and values:
            ['psis'] -> np.array with shape = (ntraj,T,N2) and dtype = complex128
            ['obsq_expects'] -> np.array with shape = (ntraj,T,len(obsq)) and dtype = complex128

    '''

    ## Check dimensions of inputs. These should be consistent with qutip Qobj.data.
    N = psi0.shape[0]
    if psi0.shape[1] != 1:
        raise ValueError("psi0 should have dimensions N2x1.")

    ## Determine seeds for the SDEs
    if type(seed) is list or type(seed) is tuple:
        assert len(seed) == ntraj
        seeds = seed
    elif type(seed) is int or seed is None:
        np.random.seed(seed)
        seeds = [np.random.randint(4294967295) for _ in range(ntraj)]
    else:
        raise ValueError("Unknown seed type.")

    T_init = time()
    psi0_arr = np.asarray(psi0.todense()).T[0]
    x0 = np.concatenate([psi0_arr.real, psi0_arr.imag])
    drift_diffusion = hybrid_drift_diffusion_holder(l1s_reduced, H2, L2s, R, eps, n, tspan, trans_phase=None)

    f = complex_to_real_vector(drift_diffusion.f_normalized)
    G = complex_to_real_matrix(drift_diffusion.G_normalized)


    def SDE_helper(args, s):
        '''Let's make different wiener increments for each trajectory'''
        m = 2 * len(L2s)
        N = len(tspan)-1
        h = (tspan[N-1] - tspan[0])/(N - 1)
        np.random.seed(s)
        dW = np.random.normal(0.0, np.sqrt(h), (N, m)) / np.sqrt(2.)
        dW_with_conj = insert_conj(dW, port=1)
        if sdeint_method is sdeint.itoQuasiImplicitEuler:
            implicit_ports = [1,2,int(m/2+1),int(m/2)+2]
            out = sdeint_method(*args, dW=dW_with_conj,
                normalized=normalize_state, downsample=downsample, implicit_ports=implicit_ports)
            return out
        if implicit_type is None:
            out = sdeint_method(*args, dW=dW_with_conj,
                normalized=normalize_state, downsample=downsample)
            return out
        else:
            try:
                out = sdeint_method(*args, dW=dW_with_conj,
                    normalized=normalize_state, downsample=downsample,
                    implicit_type=implicit_type)
            except TypeError:
                print ("Not an implicit method. implicit_type argument ignored.")
                out = sdeint_method(*args, dW=dW_with_conj,
                    normalized=normalize_state, downsample=downsample)
        return out


    ## simulation parameters
    params = [[f, G, x0, tspan]] * ntraj

    if multiprocessing:
        pool = Pool(processes=processes,)
        outputs = pool.map(lambda z: SDE_helper(z[0], z[1]),
            zip(params, seeds))
    else:
        outputs = [
            SDE_helper(p, s) for p, s in zip(params, seeds)
            ]

    try:
        xs = np.array([ o["trajectory"] for o in outputs])
    except KeyError:
        print ("Warning: trajectory not returned by SDE method!")
    try:
        norms = np.array([o["norms"] for o in outputs])
    except KeyError:
        print ("Warning: norms not returned by SDE method!")
        norms = None

    print ("done running simulation!")

    psis = xs[:,:,:int(len(x0)/2)] + 1j * xs[:,:,int(len(x0)/2):]

    # Obtaining expectations of observables
    obsq_expects = (np.asarray([[ np.asarray([ob.dot(psi).dot(psi.conj())
                        for ob in obsq])
                            for psi in psis[i] ] for i in range(ntraj)])
                                if not obsq is None else None)

    T_fin = time()
    print ("Run time:  ", T_fin - T_init, " seconds.")
    return {"psis":psis, "obsq_expects":obsq_expects, "seeds":seeds, "norms": norms}

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    markov_file = args.markov_file ## input file full path
    diffusion_file = args.diffusion_file ## input file full path
    output_dir = args.output_dir
    slow_down = args.slow_down
    ntraj = args.ntraj
    seed = args.seed
    duration = args.duration
    delta_t = args.deltat
    Nfock_a = args.nfocka
    Nfock_j = args.nfockj
    downsample = args.downsample
    Regime = args.regime
    drive_second_system = args.drive_second_system

    if args.sdeint_method_name == "":
        logging.info("sdeint_method_name not set. Using itoEuler as a default.")
        sdeint_method_name = "itoEuler"
    else:
        sdeint_method_name = args.sdeint_method_name

    R = args.R
    eps = args.eps
    noise_amp = args.noise_amp
    trans_phase = args.trans_phase

    if slow_down is None:
        slow_down = downsample
    mod = markov_model.markov_model_builder()
    mod.load(markov_file)
    reduced_traj_len = math.ceil(duration/(delta_t*slow_down))
    obs = mod.generate_obs_traj(reduced_traj_len, seed, slow_down=slow_down)
    pkl_file = open(diffusion_file, 'rb')
    data1 = pickle.load(pkl_file)
    traj_list = data1['traj_list']
    sys1_params = get_params(traj_list[0])
    sys1_regime = sys1_params['regime']
    sys1_Nfock_a = sys1_params['Nfock_a']
    sys1_Nfock_j = sys1_params['Nfock_j']

    if Regime is None:
        Regime = sys1_regime
    if Nfock_a is None:
        Nfock_a = sys1_Nfock_a
    if Nfock_j is None:
        Nfock_j = sys1_Nfock_j

    if Regime == 'kerr_bistable':
        l1s_reduced = obs_to_ls_kerr_bistable(obs)
    else:
        raise KeyError("Not implemented: Making the observables for regime %s found in file %s.")

    # Saving options
    save_mat = args.save2mat
    save_pkl = args.save2pkl

    if save_mat == False and save_pkl == False:
        logging.warning("Both pickle and mat save are disabled, no data will be saved.")
        logging.warning("You can modify this with args --save2pkl and --save2mat")

    tspan = np.arange(0,duration,delta_t)

    if Regime == "absorptive_bistable":
        logging.info("Regime is set to %s", Regime)
        H, psi0, Ls, obsq_data, obs_names = make_system_JC(Nfock_a, Nfock_j)
    elif Regime == "kerr_bistable":
        logging.info("Regime is set to %s", Regime)
        H, psi0, Ls, obsq_data, obs_names = make_system_kerr_bistable(Nfock_a, drive=drive_second_system)
    elif Regime == "kerr_qubit":
        logging.info("Regime is set to %s", Regime)
        H, psi0, Ls, obsq_data, obs_names = make_system_kerr_qubit(Nfock_a, drive=drive_second_system)
    else:
        logging.error("Unknown regime, %s, or not implemented yet.", Regime)
        raise ValueError("Unknown regime, or not implemented yet.")

    if sdeint_method_name in SDEINT_METHODS:
        sdeint_method = SDEINT_METHODS[sdeint_method_name]

        ## For now let's use the full implicit method for implicit methods.
        ## The value implicit_type can be made one of:
        ## "implicit", "semi_implicit_drift", or "semi_implicit_diffusion".
        if sdeint_method_name in IMPLICIT_METHODS:
            implicit_type = "implicit"
    else:
        logging.error("Unknown sdeint_method_name, %s, or not implemented yet.", sdeint_method_name)
        raise ValueError("Unknown sdeint_method_name, or not implemented yet.")

    D = qsd_solve_hybrid_system(H,
                                psi0,
                                tspan,
                                l1s_reduced,
                                Ls,
                                R=R,
                                eps=eps,
                                n=noise_amp,
                                sdeint_method=sdeint_method,
                                trans_phase=trans_phase,
                                obsq=obsq_data,
                                normalize_state=True,
                                downsample=downsample,
                                multiprocessing = False,
                                ntraj=ntraj,
                                processes=8,
                                seed=1,
                                implicit_type = None)

    ### include time in results
    D.update({'tspan':tspan})

    ### Save results
    if save_mat:
        logging.info("Saving mat file...")
        save2mat(data=D,
                 file_name=os.path.join(output_dir, hash),
                 obs=obs_names,
                 params=params)
    if save_pkl:
        logging.info("Saving pickle file...")
        save2pkl(data=D,
                 file_name=os.path.join(output_dir, hash),
                 obs=obs_names,
                 params=params)

## TODO: params, hash name, add stuff to run_hybrid_qsd
