from scipy.io import savemat
import pickle
import logging
import os
import hashlib
import numpy as np
from scipy import sparse
import json

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
    params['sdeint_method_name'] = str(method)
    params['num_systems'] = int(num_systems)
    params['R'] = float(R)
    params['eps'] =float(EPS)
    params['noise_amp'] =float(noise_amp)
    params['trans_phase'] =float(trans_phase)
    params['drive_second_system'] = True if drive == "True" else False
    return params

## Which values to use to distinguish groups of files
bools = {'seed': False,
         'regime': True,
         'ntraj': True,
         'delta_t': True,
         'Nfock_a': True,
         'Nfock_j': True,
         'duration': True,
         'downsample': True,
         'sdeint_method_name': True,
         'num_systems': True,
         'R': True,
         'eps': True,
         'lambd': True,
         'noise_amp': True,
         'trans_phase': True,
         'drive_second_system': True}


def files_by_params(files, bools, max_seed=None, duration=None):
    """
    Return a list of lists, each one having the unique files with distinct params determined by params_bool
    """

    ## Ensure all files have the same extension
    extension = files[0].split('.')[-1]
    assert all(f.split('.')[-1] == extension for f in files)

    params_each_file = {}
    for f in files:
        name = f.split('/')[-1]
        try:
            params_each_file[f] = get_params(name)
        except: ## couldn't use the first one (for original pickle files)
            try:
                params_each_file[f] = get_params_json(name) ## works for JSON or pickle from JSON.
            except:
                raise ValueError("could not get parameters from file %s." %name)

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

def get_params_json(file):
    """Get parameters from Json file.

    This is different from the older get_params -- the convention for the JSON
    files is to use underscores instead of dashes to simplify parsing.

    Args:
        file (string): name of file
    Returns:
        params (dict): maps each parameter to the value indicated by the file
        name.
    """
    p = {}
    [_, seed, ntraj, delta_t, Nfock_a,
    Nfock_j, duration, downsample,
    sdeint_method_name, num_systems,
    R, eps, noise_amp, lambd, trans_phase, drive_second_system] = file.split('_')
    p['seed'] = int(seed)
    p['ntraj'] = int(ntraj)
    p['delta_t'] = float(delta_t)
    p['Nfock_a'] = int(Nfock_a)
    p['Nfock_j'] = int(Nfock_j)
    p['duration'] = float(duration)
    p['downsample'] = int(downsample)
    p['sdeint_method_name'] = sdeint_method_name
    p['num_systems'] = int(num_systems)
    p['R'] = float(R)
    p['eps'] = float(eps)
    p['noise_amp'] = float(noise_amp)
    p['lambd'] = float(lambd)
    p['trans_phase'] = float(trans_phase)
    p['drive_second_system'] = bool(drive_second_system)
    return p

def get_by_params_json(files, params):
    """Get all files with specified parameters.

    Args:
        files (list of strings): List of files
        params (dict): dictionary with specific parameters whose values should
        be chosen.
    Returns:
        selected_files (list of strings): subset of the files from the input
        which have the correct parameters.
    """
    return [f for f in files if all(get_params_json(f)[p] == params[p] for p in params)]

def load_json_seq(file_path):
    """Loads data from JSON file.

    Args:
        file_path (string): name of JSON file with expectation values.

    Returns:
        data (numpy array): Numpy array with loaded expectation values.
    """

    with open(file_path) as json_data:
        trajectories = json.load(json_data)
    return np.array([[np.array(p['real']) + 1j*np.array(p['imag'])
                      for p in traj]
                          for traj in trajectories], dtype=complex)


def sorted_eigs(e_vals, e_vecs):
    '''
    Then sort the eigenvectors and eigenvalues
    s.t. the eigenvalues monotonically decrease.

    K is the number of eigenvalues/eigenvectors.

    Args:
        e_vals ([K,] ndarray): input eigenvalues
        e_vecs ([dim, K] ndarray): corresponding eigenvectors

    Returns:
        e_vals ([K,] ndarray): Sorted output eigenvalues
        e_vecs ([dim, K] ndarray): Corresponding sorted eigenvectors
    '''
    l = zip(e_vals, e_vecs.T)
    l = sorted(l,key = lambda z: -z[0])
    return np.asarray([el[0] for el in l]), np.asarray([el[1] for el in l]).T
