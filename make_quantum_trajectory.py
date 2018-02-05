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
    make_system_kerr_qubit
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

# Log everything to stdout
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)

def get_parser():
    '''get_parser returns the arg parse object, for use by an external application (and this script)
    '''
    parser = argparse.ArgumentParser(
    description="generating trajectories using quantum state diffusion")


    ################################################################################
    # Simulation Parameters
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
                        type=int,
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
                        help="Output folder. If not defined, will use PWD.",
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

    # Does the user want to print verbose output?
    quiet = args.quiet

    if not quiet:
        print_params(params=params)

    # How much to downsample results
    logging.info("Downsample set to %s",downsample)

    ## Names of files and output
    if args.outdir is None:
        outdir = os.getcwd()
    else:
        outdir = args.outdir

    ## make a folder for trajectory data
    directory_name = "/trajectory_data"
    traj_folder = (outdir + directory_name)
    try:
        os.stat(traj_folder)
    except:
        os.mkdir(traj_folder)

    param_str = "%s-%s-%s-%s-%s-%s" %(seed,ntraj,delta_t,Nfock_a,Nfock_j,duration)
    file_name = '%s/QSD_%s_%s' %(traj_folder,Regime,param_str)

    # Saving options
    save_mat = args.save2mat
    save_pkl = args.save2pkl

    if save_mat == False and save_pkl == False:
        logging.warning("Both pickle and mat save are disabled, no data will be saved.")
        logging.warning("You can modify this with args --save2pkl and --save2mat")

    if Regime == "absorptive_bistable":
        logging.info("Regime is set to %s", Regime)
        H, psi0, Ls, obsq_data, obs = make_system_JC(Nfock_a, Nfock_j)
    elif Regime == "kerr_bistable":
        H, psi0, Ls, obsq_data, obs = make_system_kerr_bistable(Nfock_a)
    elif Regime == "kerr_qubit":
        H, psi0, Ls, obsq_data, obs = make_system_kerr_qubit(Nfock_a)
    else:
        logging.error("Unknown regime, %s, or not implemented yet.", Regime)
        raise ValueError("Unknown regime, or not implemented yet.")

    tspan = np.arange(0,duration,delta_t)

    ### Run simulation
    D = qsd_solve(H=H,
                  psi0=psi0,
                  tspan=tspan,
                  Ls=Ls,
                  sdeint_method=sdeint.itoEuler,
                  obsq = obsq_data,
                  ntraj = ntraj,
                  seed = seed,
                  normalize_state=True,
                  downsample=downsample
                  )

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
                 obs=obs,
                 params=params)
    if save_pkl:
        logging.info("Saving pickle file...")
        save2pkl(data=D_downsampled,
                 file_name=file_name,
                 obs=obs,
                 params=params)


if __name__ == '__main__':
    main()
