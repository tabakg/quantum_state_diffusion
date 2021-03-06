import argparse
import numpy as np
import numpy.linalg as la
import logging
import os
import pickle
import sys
sys.path.append('/scratch/users/tabakg/qsd_dev')
from utils import load_trajectory
from utils import save
from utils import sorted_eigs

# Log everything to stdout
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)

def get_parser():
    '''get_parser returns the arg parse object, for use by an external application (and this script)
    '''
    parser = argparse.ArgumentParser(
    description="Generating diffusion maps of trajectories.")


    ################################################################################
    # General Simulation Parameters
    ################################################################################

    parser.add_argument("--traj",
                        dest='traj',
                        help="complete path to trajectory file, or a list of "
                             "complete paths to trajectory files separated by commas.",
                        type=str,
                        default=None)

    parser.add_argument("--eps",
                        dest='eps',
                        help="epsilon parameter in diffusion maps",
                        type=float,
                        default=0.5)

    parser.add_argument("--alpha",
                        dest='alpha',
                        help="alpha parameter in diffusion maps.",
                        type=float,
                        default=0.5)

    parser.add_argument("--eig_lower_bound",
                        dest='eig_lower_bound',
                        help="Lower bound of generated eigenvalue",
                        type=int,
                        default=0)

    parser.add_argument("--eig_upper_bound",
                        dest='eig_upper_bound',
                        help="upper bound of generated eigenvalue",
                        type=int,
                        default=6)

    parser.add_argument("--sample_size",
                        dest='sample_size',
                        help="number of points to use. If 0, use all.",
                        type=int,
                        default=10000)

    # Does the user want to quiet output?
    parser.add_argument("--quiet",
                        dest='quiet',
                        action="store_true",
                        help="Turn off logging (debug and info)",
                        default=False)

    parser.add_argument("--output_dir",
                        dest='outdir',
                        type=str,
                        help="Output folder.",
                        default=None)

    parser.add_argument("--output_name",
                        dest='output_name',
                        type=str,
                        help="Name of output file.",
                        default=None)

    return parser

def inner_to_FS(val):
    return np.arccos(np.sqrt(val)) if val < 1 else 0.

def converter(a, f):
    a = a.reshape(-1)
    for i, v in enumerate(a):
        a[i] = f(v)

def FS_metric(u, v):
    l = u.shape[-1]
    if v.shape[-1] != l:
        raise ValueError("The lengths of the inputs should be the same.")
    if l%2 != 0:
        raise ValueError("The lengths of the inputs must be even.")

    n = int(l/2)

    inner = ( (np.dot(u[:,:n],v.T[:n,:]) + np.dot(u[:,n:],v.T[n:,:]))**2
            + (np.dot(u[:,:n],v.T[n:,:]) - np.dot(u[:,n:],v.T[:n,:]))**2  )

    converter(inner, inner_to_FS)
    return inner


def run_diffusion_map_dense(distance_matrix,
                            eps=0.5,
                            alpha=0.5,
                            eig_lower_bound=None,
                            eig_upper_bound=None):
    '''
    Computes the eigenvealues and eigenvectors for diffusion maps
    given a dense input.

    Args:
        distance_matrix (numpy.ndarray): a kxk square input representing mutual distances
            between k points.
        eps (double): diffusion map parameter for K = exp( -distance_matrix ** 2 / (2 * eps) ).

    Returns:
        eigenvales (np.ndarray): a length k array of eigenvalues.
            eigenvectors (numpy.ndarray): a kxk array representing eigenvectors (descending order).
    '''
    K = np.exp(-distance_matrix**2/ (2. * eps) )
    d_K = np.squeeze(np.asarray(K.sum(axis = 1)))
    d_K_inv = np.power(d_K,-1)
    d_K_inv = np.nan_to_num(d_K_inv)
    L = d_K_inv*(d_K_inv*K).T
    d_L = np.squeeze(np.asarray(L.sum(axis = 1)))
    d_L_inv = np.power(d_L,-alpha)
    M = d_L_inv*(d_L_inv*L).T
    eigs = la.eigh(M)
    if eig_lower_bound is None:
        eig_lower_bound = 0
    if eig_upper_bound is None:
        eig_upper_bound = len(eigs[0])
    return (eigs[0][::-1][eig_lower_bound:eig_upper_bound],
            eigs[1].T[::-1].T[:,eig_lower_bound:eig_upper_bound])

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Set up commands from parser
    params = dict()
    traj_list = params['traj_list'] = traj_list = [item for item in args.traj.split(',')]
    eps = params['eps'] = args.eps
    alpha = params['alpha'] = args.alpha
    eig_lower_bound = params['eig_lower_bound'] = args.eig_lower_bound
    eig_upper_bound = params['eig_upper_bound'] = args.eig_upper_bound
    sample_size = params['sample_size'] = args.sample_size
    output = args.output_name

    # Does the user want to print verbose output?
    quiet = args.quiet

    ## TODO: import print_params, etc...
    # if not quiet:
    #     print_params(params=params)

    ## Names of files and output
    if args.outdir is None:
        outdir = os.getcwd()
    else:
        outdir = args.outdir

    ## Memory efficient

    diffusion_coords_dict = {'expects': [], 'times': [], 'traj_list': traj_list}
    psis = []
    num_successful = 0
    for traj in traj_list:
        try:
            loaded = load_trajectory(traj)
            num_successful += 1
        except pickle.UnpicklingError:
            logging.info("Could not open trajectory %s" %traj)

        ## Concatenate the psis and expects across trajectories
        psis_current_traj = np.concatenate(loaded['psis'])
        expects_current_traj = np.concatenate(loaded['expects'])
        assert psis_current_traj.shape[0] == expects_current_traj.shape[0]

        ## Find downsample factor to avoid using too much memory
        ## This assumes we want a total of sample_size points, and
        ## the number of points per trajectory is the same.
        every_other_n = int(psis_current_traj.shape[0] * len(traj_list) / (sample_size))

        ## If the fraction is too small (too few points) just sample every point.
        if every_other_n == 0:
            every_other_n = 1

        ## Downsample psis, expects, and times, and add to dict
        psis.append(psis_current_traj[::every_other_n])
        diffusion_coords_dict['expects'].append(expects_current_traj[::every_other_n])
        diffusion_coords_dict['times'].append(loaded['times'][::every_other_n])

    ## Consolidate expects and times for consistency
    sampled_psis = np.concatenate(psis)
    diffusion_coords_dict['times'] = np.concatenate(diffusion_coords_dict['times'])
    diffusion_coords_dict['expects'] = np.concatenate(diffusion_coords_dict['expects'])

    ## Output messages
    logging.info("Successfully loaded %s/%s trajectories." %(len(traj_list), num_successful))
    logging.info("Total number of points is %s" % psis_current_traj.shape[0])

    psis_doubled = np.concatenate([sampled_psis.real.T,sampled_psis.imag.T]).T ## convert to (real, imag) format

    distance_matrix = FS_metric(psis_doubled, psis_doubled)
    vals, vecs = run_diffusion_map_dense(distance_matrix,eps=eps,
                                        alpha=alpha,
                                        eig_lower_bound=eig_lower_bound,
                                        eig_upper_bound=eig_upper_bound)

    ## Sort eigen-pairs, dropping the trivial eigenvalue.
    vals_tmp, vecs_tmp = vals[1:], vecs[:,1:]
    sorted_vals, sorted_vecs = sorted_eigs(vals_tmp, vecs_tmp)

    diffusion_coords = {"vals" : sorted_vals, "vecs" : sorted_vecs}
    diffusion_coords_dict.update(diffusion_coords)
    save(output, diffusion_coords_dict)

if __name__ == '__main__':
    main()
