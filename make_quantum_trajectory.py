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

from qnet.algebra.operator_algebra import *
from qnet.algebra.circuit_algebra import *
import qnet.algebra.state_algebra as sa
from quantum_state_diffusion import qsd_solve

from utils import (
    make_nparams, 
    save2mat,
    save2pkl,
    print_params
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
                        help="Duration in ()", 
                        type=int, 
                        default=10)

    # Delta T
    parser.add_argument("--delta_t", 
                        dest='deltat', 
                        help="Parameter delta_t", 
                        type=float, 
                        default=2e-3)

    # Nfock_a
    parser.add_argument("--Nfock_a", 
                        dest='nfocka', 
                        help="Parameter N_focka", 
                        type=int, 
                        default=50)

    # Nfock_j
    parser.add_argument("--Nfock_j", 
                        dest='nfockj', 
                        help="Parameter N_fockj", 
                        type=int, 
                        default=2)

    # How much to downsample results
    parser.add_argument("--downsample", 
                        dest='downsample', 
                        help="How much to downsample results", 
                        type=int, 
                        default=100)


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
    
    # Does the user want to print verbose output?
    quiet = args.quiet

    if not quiet:
        print_params(params=params)

    # How much to downsample results
    logging.info("Downsample set to %s",downsample)

    ## Names of files and output
    Regime = "absorptive_bistable"
    param_str = "%s-%s-%s-%s-%s" %(ntraj,delta_t,Nfock_a,Nfock_j,duration)
    outdir = ""
    if args.outdir != None:
        outdir = args.outdir
    file_name = '%s/QSD_%s_%s' %(outdir,Regime,param_str) 

    # Saving options
    save_mat = args.save2mat
    save_pkl = args.save2pkl

    if save_mat == False and save_pkl == False:
        logging.warning("Both pickle and mat save are disabled, no data will be saved.")
        logging.warning("You can modify this with args --save2pkl and --save2mat")

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

    if Regime == "absorptive_bistable":
        logging.info("Regime is set to %s", Regime)
        nparams = make_nparams(W=W,k=k,g=g,g0=g0,DD=DD,TT=TT)
    else:
        logging.error("Unknown regime, %s, or not implemented yet.", Regime)
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
    D = qsd_solve(H=H, 
                  psi0=psi0, 
                  tspan=tspan, 
                  Ls=Ls, 
                  sdeint_method=sdeint.itoEuler, 
                  obsq = obsq, 
                  ntraj = ntraj,
                  seed = seed, 
                  normalize_state=True)

    ### include time in results
    D.update({'tspan':tspan})

    ### downsample
    D_downsampled = {'psis' : D['psis'][:,::downsample],
                     'obsq_expects' : D['obsq_expects'][:,::downsample],
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
