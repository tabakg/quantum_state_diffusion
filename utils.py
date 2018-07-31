from scipy.io import savemat
import pickle
import logging
import os
import hashlib
import numpy as np
from scipy import sparse

def print_params(params):
    '''print params will print a dictioary of parameters to the screen for the user
    :param params: the dictionary of parameters
    '''
    for key,value in params.items():
        logging.info("Parameter %s set to %s",key,value)


def save2mat(data, file_name, obs, params=None):
    ''' uses scipy.io savemat to save a .mat file of the data
    :param data: the data dictionary object
    :param file_name: the file name (with extension) to save to
    :param obs:
    :param params: extra params to add to the save object
    '''
    logging.info("Saving result to %s.mat", file_name)
    mdict = prepare_save(data,file_name,obs,params)
    savemat("%s.mat" %file_name, mdict)
    logging.info("Data saved to .mat file %s",file_name)


def save2pkl(data,file_name,obs, params=None):
    ''' save2pkl saves to a pickle file
    :param data: the data dictionary object
    :param file_name: the file name (with extension) to save to
    :param obs:
    :param params: extra params to add to the save object
    '''
    logging.info("Saving result to %s.pkl", file_name)
    mdict = prepare_save(data,file_name,obs,params)
    output = open("%s.pkl" %file_name, 'wb')
    pickle.dump(mdict,output,protocol=0)
    output.close()
    logging.info("Data saved to pickle file %s", file_name)


def prepare_save(data, file_name, obs, params=None):
    """prepare_save: takes an mcdata object and the observables and stores the states,
    expectations, times, observable labels (str and latex), random seeds,
    number of trajectories as:

        {
            "psis": psis,                          # shape (ntraj, ntimes, dim_psi) complex128
            "expects": expects,                    # shape (ntraj, ntimes, num_observables) complex128
            "times": times,                        # shape (ntimes) float64
            "observable_str": observable_str,      # shape (num_observables) string
            "observable_latex": observable_latex,  # shape (num_observables) string
            "seeds": seeds,                        # shape (ntraj) int
        }

    Additional parameters can be passed using the params argument, as a python dictionary.
    """

    ntraj = data['psis'].shape[0]
    assert ntraj == data['obsq_expects'].shape[0]
    assert ntraj >= 1
    psis = data['psis']
    expects = data['obsq_expects']
    times = data['tspan']
    observable_str = [str(o) for o in obs]
    observable_latex = [o._repr_latex_() for o in obs]
    seeds = data['seeds']
    mdict = {
            "psis": psis,
            "expects": expects,
            "times": times,
            "observable_str": observable_str,
            "observable_latex": observable_latex,
            "seeds": seeds,
    }
    if params != None:
        mdict.update(params) ## other paramters (optional)

    return mdict

def load_trajectory(traj):
    """Generic load"""
    pkl_file = open(traj, 'rb')
    pkl_dict = pickle.load(pkl_file)
    pkl_file.close()
    return pkl_dict

def save(file, pkl_dict):
    """Generic save"""
    pkl_file = open(file, 'wb')
    pickle.dump(pkl_dict, pkl_file, protocol=0)
    pkl_file.close()

def get_params(traj):
    """Get parameters from trajectory file name.

    """
    params={}
    things = traj[4:].split('-')
    first = things[0].split('_')
    params['seed'] = int(first[-1])
    params['regime'] = str('_'.join(first[:-1])).split('/')[-1]
    params['ntraj'] = int(things[1])
    try:
        (delta_t1,delta_t2,Nfock_a,Nfock_j,
            duration,downsample,method,num_systems,
            R,EPS,noise_amp,trans_phase,drive) = things[2:]
        params['delta_t'] = float("".join([delta_t1,'-',delta_t2]))
    except:
        (delta_t,Nfock_a,Nfock_j,
            duration,downsample,method,num_systems,
            R,EPS,noise_amp,trans_phase,drive) = things[2:]
        params['delta_t'] = float(delta_t)

    drive = drive[:-4] ## drop the .pkl extension...

    params['Nfock_j'] = int(Nfock_j)
    params['duration'] = float(duration)
    params['downsample'] = int(downsample)
    params['method'] = str(method)
    params['num_systems'] = int(num_systems)
    params['R'] = float(R)
    params['EPS'] =float(EPS)
    params['noise_amp'] =float(noise_amp)
    params['trans_phase'] =float(trans_phase)
    params['drive'] = True if drive == "True" else False
    return params

## Which values to use to distinguish groups of files
bools = {'seed': False,
         'regime': True,
         'ntraj': True,
         'delta_t': True,
         'Nfock_j': True,
         'duration': True,
         'downsample': True,
         'method': True,
         'num_systems': True,
         'R': True,
         'EPS': True,
         'noise_amp': True,
         'trans_phase': True,
         'drive': True}

def files_by_params(files, bools, max_seed=None, duration=None):
    """
    Return a list of lists, each one having the unique files with distinct params determined by params_bool
    """
    params_each_file = {f: get_params(f) for f in files}
    if max_seed:
        params_each_file = {f:p for f,p in params_each_file.items() if p['seed'] <= max_seed}
    if duration:
        params_each_file = {f:p for f,p in params_each_file.items() if p['duration'] == duration}
    relevant_params_each_file = {f : tuple(sorted((k,v) for k,v in p.items() if bools[k]))
        for f, p in params_each_file.items()}
    all_possible_params = set(relevant_params_each_file.values())
    groups = {p : [f for f in relevant_params_each_file if relevant_params_each_file[f] == p]
        for p in all_possible_params}
    return list(groups.values())


def make_hash(traj):
    """We make a name using a hash because there could be multiple
    trajectories in traj_list feeding into a single set of diffusion maps"""
    hash_code = hashlib.sha256(traj.encode('utf-8'))
    return hash_code.hexdigest()


def dim_check(H, Ls):
    """Make sure the dimensions match.

    Args:
        H (possibly sparse matrix)
        Ls (list of possibly sparse matrices)

    Ensures that H and each L in Ls is a square
    matrix of the same size.
    """
    N = H.shape[0]
    assert H.shape == (N, N)
    assert all(L.shape == (N, N) for L in Ls)
    return N


def preprocess_operators(H1, H2, L1s, L2s, ops_on_whole_space):
    """Make sure operators are defined over the entire space.

    First, check that all dimensions match. Then return either the given
    operators, or extend them so that they are defined on the Kronecker product
    Hilbert space.

    Args:
        H1: square matrix
            Hamiltonian matrix of system 1
        H2: square matrix
            Hamiltonian matrix of system 2
        L1s: list of square matrices
            Lindblad matrices of system 1
        L2s: list of square matrices
            Lindblad matrices of system 2
        ops_on_whole_space: boolean
            whether the given operators are defined on the whole space

    Returns: H1, H2, L1s, L2s
        Operators extended to whole space if necessary
    """
    N1 = dim_check(H1, L1s)
    N2 = dim_check(H2, L2s)

    if ops_on_whole_space:
        assert(N1 == N2)
        return H1, H2, L1s, L2s
    else:
        I1 = np.eye(N1)
        I2 = np.eye(N2)
        H1 = sparse.csr_matrix(np.kron(H1.todense(), I2))
        H2 = sparse.csr_matrix(np.kron(I1, H2.todense()))
        L1s = [sparse.csr_matrix(np.kron(L1.todense(), I2)) for L1 in L1s]
        L2s = [sparse.csr_matrix(np.kron(I1, L2.todense())) for L2 in L2s]
        return H1, H2, L1s, L2s
