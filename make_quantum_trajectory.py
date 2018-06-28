#!/usr/bin/env python

'''

Make Quantum Trajectory
Author: Gil Tabak
Date: Nov 3, 2016

Generating trajectories using quantum state diffusion. We will be primairly
interested in the absorptive bistability (Jaynes Cummings model)
I store trajectory files as *.pkl files or *.mat files. This way I can easily
load them into another notebook, or load the trajectories to matlab.
Requires Python 3.

'''

from quantum_state_diffusion import (
    qsd_solve,
    qsd_solve_two_systems
)

from utils import (
    save2mat,
    save2pkl,
    print_params
)

from prepare_regime import (
    make_system_JC,
    make_system_kerr_bistable,
    make_system_kerr_bistable_regime2,
    make_system_kerr_bistable_regime3,
    make_system_kerr_bistable_regime4,
    make_system_kerr_bistable_regime5,
    make_system_kerr_bistable_regime6,
    make_system_kerr_bistable_regime7,
    make_system_kerr_bistable_regime_chose_drive,
    make_system_kerr_qubit,
    ## make_system_JC_two_systems, ## Not yet implemented
    make_system_kerr_bistable_two_systems,
    make_system_kerr_qubit_two_systems,
    make_system_empty_then_kerr,
)

import sdeint
import argparse
import numpy as np
import numpy.linalg as la
import logging
import os
import pickle
from scipy import sparse
from sympy import sqrt
import sys

SDEINT_METHODS = {"itoEuler": sdeint.itoEuler,
                  "itoSRI2": sdeint.itoSRI2,
                  ## "itoMilstein": itoMilstein.itoSRI2,
                  "numItoMilstein": sdeint.numItoMilstein,
                  "itoImplicitEuler": sdeint.itoImplicitEuler,
                  "itoQuasiImplicitEuler": sdeint.itoQuasiImplicitEuler}
IMPLICIT_METHODS=["itoImplicitEuler"]

# Log everything to stdout
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)

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
                        dest='nfocka',
                        help="Number of fock states in each cavity",
                        type=int,
                        default=50)

    # Nfock_j
    parser.add_argument("--Nfock_j",
                        dest='nfockj',
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


    # Does the user want to quiet output?
    parser.add_argument("--quiet",
                        dest='quiet',
                        action="store_true",
                        help="Turn off logging (debug and info)",
                        default=False)

    # Does the user want to quiet output?
    parser.add_argument("--output_dir",
                        dest='outdir',
                        type=str,
                        help="Output folder. If not defined, will use place in a directory /trajectory_data.",
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


def main():
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        sys.exit(0)

    # Set up commands from parser
    params = dict()
    ntraj = params['Ntraj'] = args.ntraj
    seed = params['seed'] = args.seed
    duration = params['duration'] = args.duration
    delta_t = params['delta_t'] = args.deltat
    Nfock_a = params['Nfock_a'] = args.nfocka
    Nfock_j = params['Nfock_j'] = args.nfockj
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
    trans_phase = params['trans_phase'] = args.trans_phase

    # Does the user want to print verbose output?
    quiet = args.quiet

    if not quiet:
        print_params(params=params)

    # How much to downsample results
    logging.info("Downsample set to %s", downsample)

    ## Names of files and output
    if args.outdir is None:
        outdir = os.getcwd()
    else:
        outdir = args.outdir

    try:
        os.stat(outdir)
    except:
        os.mkdir(outdir)

    param_str = ("%s-"*14)[:-1] %(seed,
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
                                trans_phase,
                                drive_second_system)
    file_name = '%s/QSD_%s_%s' %(outdir,Regime,param_str)

    # Saving options
    save_mat = args.save2mat
    save_pkl = args.save2pkl

    if save_mat == False and save_pkl == False:
        logging.warning("Both pickle and mat save are disabled, no data will be saved.")
        logging.warning("You can modify this with args --save2pkl and --save2mat")

    implicit_type = None

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

    tspan = np.arange(0,duration,delta_t)

    obsq_data = None
    if num_systems == 1:

        if Regime == "absorptive_bistable":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_JC(Nfock_a, Nfock_j)
        elif Regime == "kerr_bistable":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_kerr_bistable(Nfock_a)
        elif Regime == "kerr_bistable2":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_kerr_bistable_regime2(Nfock_a)
        elif Regime == "kerr_bistable3":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_kerr_bistable_regime3(Nfock_a)
        elif Regime == "kerr_bistable4":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_kerr_bistable_regime4(Nfock_a)
        elif Regime == "kerr_bistable5":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_kerr_bistable_regime5(Nfock_a)
        elif Regime == "kerr_bistable6":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_kerr_bistable_regime6(Nfock_a)
        elif Regime == "kerr_bistable7":
            logging.info("Regime is set to %s", Regime)
            H, psi0, Ls, obsq_data, obs_names = make_system_kerr_bistable_regime7(Nfock_a)
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

        ### Run simulation for one system
        D = qsd_solve(H=H,
                      psi0=psi0,
                      tspan=tspan,
                      Ls=Ls,
                      sdeint_method=sdeint_method,
                      obsq = obsq_data,
                      ntraj = ntraj,
                      seed = seed,
                      normalize_state=True,
                      downsample=downsample,
                      implicit_type=implicit_type,
                      )
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
        else:
            logging.error("Unknown regime, %s, or not implemented yet.", Regime)
            raise ValueError("Unknown regime, or not implemented yet.")

        ### Run simulation for one system
        D = qsd_solve_two_systems(H1,
                                  H2,
                                  psi0,
                                  tspan,
                                  L1s,
                                  L2s,
                                  R=R,
                                  eps=eps,
                                  n=noise_amp,
                                  sdeint_method=sdeint_method,
                                  trans_phase=trans_phase,
                                  obsq=obsq_data,
                                  normalize_state=True,
                                  downsample=downsample,
                                  ops_on_whole_space = False, ## assume the given operators only operate on their own subspace
                                  multiprocessing = False, ## disable multiprocessing for now
                                  ntraj=ntraj,
                                  seed=seed,
                                  implicit_type = implicit_type,
                                  )
    else:
        logging.error("Unknown num_systems, %s, or not implemented yet.", num_systems)
        raise ValueError("Unknown num_systems, or not implemented yet.")

    ### include time in results
    D.update({'tspan':tspan})

    ### downsample
    D_downsampled = {'psis' : D['psis'],
                     'obsq_expects' : D['obsq_expects'],
                     'seeds' : D['seeds'],
                     'tspan' : D['tspan'][::downsample] }

    ### Save results
    if save_mat:
        logging.info("Saving mat file...")
        save2mat(data=D_downsampled,
                 file_name=file_name,
                 obs=obs_names,
                 params=params)
    if save_pkl:
        logging.info("Saving pickle file...")
        save2pkl(data=D_downsampled,
                 file_name=file_name,
                 obs=obs_names,
                 params=params)


if __name__ == '__main__':
    main()
