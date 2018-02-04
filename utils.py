from scipy.io import savemat
import pickle
import logging


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
