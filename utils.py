from scipy.io import savemat
#from IPython.display import FileLink, display
import numpy as np
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
    data = prepare_save(data,obs,params=params)
    savemat(file_name, mdict)
    logging.info("Data saved to .mat file %s",file_name)


def save2pkl(data, file_name, obs, params=None):
    ''' save2pkl saves to a pickle file
    :param data: the data dictionary object
    :param file_name: the file name (with extension) to save to
    :param obs:
    :param params: extra params to add to the save object
    '''
    output = open(file_name + ".pkl", 'wb')
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


# default are for absorptive bistability
def make_nparams(Cn=10.5, kn=.12, yn=11.3, DDn=0, TTn=0., J = 0.5):
    g0n = np.sqrt(2.*kn*Cn)
    Wn = yn*kn/np.sqrt(2)/g0n

    nparams = {
        W: Wn/np.sqrt(2*kn),
        k: 2*kn,
        g: 2./np.sqrt(2*J),
        g0: -g0n/np.sqrt(2*J),
        DD: DDn,
        TT: TTn,
    }
    xrs = np.linspace(0, 10)
    yrs = 2*Cn*xrs/(1+xrs**2) + xrs
    return nparams

