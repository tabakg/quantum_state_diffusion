import argparse
import numpy as np
import numpy.linalg as la
import logging
import os
import pickle
import sys
import hashlib

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
                        help="Output folder. If not defined, will use place in a directory /trajectory_data.",
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

def load_trajectory(traj):
    pkl_file = open(traj, 'rb')
    pkl_dict = pickle.load(pkl_file)
    pkl_file.close()
    return pkl_dict

def save_diffusion_coordinates(output, diff_coords):

    pkl_file = open(output, 'wb')
    pickle.dump(diff_coords,pkl_file,protocol=0)
    pkl_file.close()

    return traj_data


def run_diffusion_map_dense(distance_matrix,eps = 0.5, alpha = 0.5, eig_lower_bound = None, eig_upper_bound = None):
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

    ## make a folder for trajectory data
    directory_name = "/diffusion_maps_data"
    diffusion_maps_folder = (outdir + directory_name)
    try:
        os.stat(diffusion_maps_folder)
    except:
        os.mkdir(diffusion_maps_folder)

    def make_hash():
        """We make a name using a hash because there could be multiple
        trajectories in traj_list feeding into a single set of diffusion maps"""
        hash_code = hashlib.sha256(args.traj.encode('utf-8'))
        return hash_code.hexdigest()

    hash_code = make_hash()
    output = '%s/diffusion_map_%s.pkl' %(diffusion_maps_folder, hash_code)

    pkl_dict = {traj: load_trajectory(traj) for traj in traj_list}

    diffusion_coords_dict = {}
    psis = np.concatenate(np.concatenate([pkl_dict[traj]['psis'] for traj in traj_list]))
    diffusion_coords_dict['expects'] = np.concatenate(np.concatenate([pkl_dict[traj]['expects'] for traj in traj_list]))
    diffusion_coords_dict['times'] = np.concatenate([pkl_dict[traj]['times'] for traj in traj_list])
    diffusion_coords_dict['traj_list'] = traj_list

    if sample_size == 0:
        sampled_psis = psis
    else:
        every_other_n = int(psis.shape[-1] / sample_size)
        if every_other_n == 0:
            sampled_psis = psis
        else:
            sampled_psis = psis[::every_other_n]

    psis_doubled = np.concatenate([psis.real.T,psis.imag.T]).T ## convert to (real, imag) format

    distance_matrix = FS_metric(psis_doubled, psis_doubled)
    vals, vecs = run_diffusion_map_dense(distance_matrix,eps=eps,
                                        alpha=alpha,
                                        eig_lower_bound=eig_lower_bound,
                                        eig_upper_bound=eig_upper_bound)
    diffusion_coords = {"vals":vals, "vecs":vecs}
    diffusion_coords_dict.update(diffusion_coords)
    save_diffusion_coordinates(output, diffusion_coords_dict)

if __name__ == '__main__':
    main()