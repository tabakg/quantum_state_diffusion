'''
Quantum State Diffusion

Author: Gil Tabak
Date: Nov 3, 2016

This script uses the library sdeint to perform quantum state diffusion trajectories.
The inputs are purposely similar to qubit functions like mcsolve to make
integration easier later.

'''

import numpy as np
import numpy.linalg as la
import sdeint
from scipy import sparse
from time import time
from multiprocess import Pool

################################################################################
#### Conversion between real and complex-valued terms
################################################################################

def complex_to_real_vector(f):
    """
    Generates a vector-valued function taking and returning
    real values instead of complex values.

    Args:
        f: function sending (psi, t) -> psi_out
            Here psi and psi_out are complex valued arrays of dimension (d,)
    Returns:
        tilde_f : function sending (x, t) -> x_out
            Here x and x_out are real valued arrays of dimension (2d,)
    """
    def tilde_f(x, t):
        psi = x[:int(len(x)/2)] + 1j*x[int(len(x)/2):]
        f_val = f(psi, t)
        return np.concatenate([f_val.real, f_val.imag])
    return tilde_f


def complex_to_real_matrix(G):
    """
    Generates a matrix-valued function taking and returning
    real values instead of complex values.

    Args:
        G: function sending (psi, t) -> M
            Here psi is a complex-valued array of dimension (d,)
            M is a complex-value array of dimension (d,m)
    Returns:
        tilde_G : function sending (x, t) -> M_out
            Here x and is a real valued array of dimension (2d,)
            M is a real-value array of dimension (2d,2m)
    """
    def tilde_G(x, t):
        psi = x[:int(len(x)/2)] + 1j*x[int(len(x)/2):]
        G_val = G(psi, t)
        return np.vstack([np.hstack([G_val.real, -G_val.imag]),
                          np.hstack([G_val.imag, G_val.real])])
    return tilde_G

################################################################################
#### One system below
################################################################################


class drift_diffusion_holder(object):
    '''
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
    def __init__(self, H, Ls, tspan, normalized_equation=True):
        self.t_old = min(tspan) - 1.
        self.H = H
        self.Ls = Ls
        self.Lpsis = None
        self.ls = None
        if normalized_equation:
            self.f = self.f_normalized
            self.G = self.G_normalized
        else:
            self.f = self.f_non_normalized
            self.G = self.G_non_normalized

    def update_Lpsis_and_ls(self, psi, t):
        '''Updates Lpsis and ls.

        If t is different than t_old, update Lpsis, ls, and t_old.
        Otherwise, do nothing.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float
        '''
        if t != self.t_old:
            self.Lpsis = [L.dot(psi) for L in self.Ls]
            self.ls = [Lpsi.dot(psi.conj()) for Lpsi in self.Lpsis]
            self.t_old = t

    def f_normalized(self, psi, t):
        '''Computes drift f.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: array of shape (d,)
            to define the deterministic part of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)
        return (-1j * self.H.dot(psi)
                - sum([ 0.5*(L.H.dot(Lpsi) + np.conj(l)*l*psi)
                - np.conj(l)*(Lpsi)
                    for L,l,Lpsi in zip(self.Ls, self.ls, self.Lpsis)]))

    def G_normalized(self, psi, t):
        '''Computes diffusion G.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: returning an array of shape (d, m)
            to define the noise coefficients of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)
        return np.vstack([Lpsi - l*psi
            for Lpsi, l in zip(self.Lpsis, self.ls)]).T

    def f_non_normalized(self, psi, t):
        '''Computes drift f.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: array of shape (d,)
            to define the deterministic part of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)
        return (-1j * self.H.dot(psi)
                - sum([ 0.5*(L.H.dot(Lpsi)) - np.conj(l)*(Lpsi)
                    for L,l,Lpsi in zip(self.Ls, self.ls, self.Lpsis)]))

    def G_non_normalized(self, psi, t):
        '''Computes diffusion G.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: returning an array of shape (d, m)
            to define the noise coefficients of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)
        return np.vstack(self.Lpsis).T

def qsd_solve(H,
              psi0,
              tspan,
              Ls,
              sdeint_method,
              obsq=None,
              normalized_equation=True,
              normalize_state=True,
              downsample=1,
              multiprocessing = False,
              ntraj=1,
              processes=8,
              seed=1,
              implicit_type=None,
              ):
    '''
    Args:
        H: NxN csr matrix, dtype = complex128
            Hamiltonian.
        psi0: Nx1 csr matrix, dtype = complex128
            input state.
        tspan: numpy array, dtype = float
            Time series of some length T.
        Ls: list of NxN csr matrices, dtype = complex128
            System-environment interaction terms (Lindblad terms).
        sdeint_method (Optional) SDE solver method:
            Which SDE solver to use. Default is sdeint.itoSRI2.
        obsq (optional): list of NxN csr matrices, dtype = complex128
            Observables for which to generate trajectory information.
            Default value is None (no observables).
        normalized_equation (optional): Boolean
            Use the normalized quantum state diffusion equations. (TODO: case False)
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
        implicit_type (optional): string
            Type of implicit solver to use if the solver is implicit.

    Returns:
        A dictionary with the following keys and values:
            ['psis'] -> np.array with shape = (ntraj,T,N) and dtype = complex128
            ['obsq_expects'] -> np.array with shape = (ntraj,T,len(obsq)) and dtype = complex128

    '''

    ## Check dimensions of inputs. These should be consistent with qutip Qobj.data.
    N = psi0.shape[0]
    if psi0.shape[1] != 1:
        raise ValueError("psi0 should have dimensions Nx1.")
    a,b = H.shape
    if a != N or b != N:
        raise ValueError("H should have dimensions NxN (same size as psi0).")
    for L in Ls:
        a,b = L.shape
        if a != N or b != N:
            raise ValueError("Every L should have dimensions NxN (same size as psi0).")
    ## Determine seeds for the SDEs
    if type(seed) is list or type(seed) is tuple:
        assert len(seed) == ntraj
        seeds = seed
    elif type(seed) is int or seed is None:
        np.random.seed(seed)
        seeds = [np.random.randint(1000000) for _ in range(ntraj)]
    else:
        raise ValueError("Unknown seed type.")

    T_init = time()
    psi0_arr = np.asarray(psi0.todense()).T[0]
    x0 = np.concatenate([psi0_arr.real, psi0_arr.imag])
    drift_diffusion = drift_diffusion_holder(H, Ls, tspan)

    f = complex_to_real_vector(drift_diffusion.f)
    G = complex_to_real_matrix(drift_diffusion.G)

    def SDE_helper(args, s):
        '''Let's make different wiener increments for each trajectory'''
        m = 2 * len(Ls)
        N = len(tspan)-1
        h = (tspan[N-1] - tspan[0])/(N - 1)
        np.random.seed(s)
        dW = np.random.normal(0.0, np.sqrt(h), (N, m)) / np.sqrt(2.)
        if implicit_type is None:
            out = sdeint_method(*args, dW=dW, normalized=normalize_state,
                downsample=downsample)
        try:
            out = sdeint_method(*args, dW=dW, normalized=normalize_state,
                implicit_type=implicit_type, downsample=downsample)
        except TypeError:
            print ("Not an implicit method. implicit_type argument ignored.")
            out = sdeint_method(*args, dW=dW, normalized=normalize_state,
                downsample=downsample)
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

################################################################################
#### Two systems below
################################################################################

def individual_term(H, Ls, ls, Lpsis, psi):
    """Generates the drift term for each individual component.

    Args:
        H: square matrix
            Hamiltonian matrix
        Ls: list of square matrices
            Lindblad matrices
        ls: list of complex values
            expectations of Lindblad matrices
        Lpsis: list of complex-valued vectors
            L matrices dotted with psi
        psi: complex-values vector
            the state psi

    Returns: complex-valued vector
        term in SDE for individual component
    """
    return (-1j * H.dot(psi)
            - sum([0.5 * (L.H.dot(Lpsi) + np.conj(l)*l*psi) - np.conj(l)*(Lpsi)
                   for L, l, Lpsi in zip(Ls, ls, Lpsis)]))

def classical_trans(L2, l1, l2, L2psi, psi):
    """classical transmission term for SDE.

    Assumes heterodyne detection of system 1 via operator L1
    to system 2 with operator L2.

    Args:
        L2: square matrix
            Lindblad operator
        l1: complex value
            expectation of l1
        l2: complex value
            expectation of l2
        L2psi: complex-valued vector
            L2 dotted with psi
        psi: complex-values vector
            the state psi

    Returns: complex-valued vector
        term in SDE for classical transmission

    """
    return (-L2.H.dot(psi)*l1 + L2psi*np.conj(l1)
            + 0.5 * (l1*np.conj(l2) - np.conj(l1)*l2)*psi)

def quantum_trans(L2, l1, l2, L1psi, L2psi, psi):
    """classical transmission term for SDE.

    Assumes heterodyne detection of system 1 via operator L1
    to system 2 with operator L2.

    Args:
        L2: square matrix
            Lindblad operator
        l1: complex value
            expectation of l1
        l2: complex value
            expectation of l2
        L1psi: complex-valued vector
            L2 dotted with psi
        L2psi: complex-valued vector
            L2 dotted with psi
        psi: complex-values vector
            the state psi

    Returns: complex-valued vector
        term in SDE for coherent transmission

    """
    return (-L2.H.dot(L1psi) + L2psi*np.conj(l1) + L1psi*np.conj(l2)
            - 0.5*(l1*np.conj(l2) + np.conj(l1)*l2)*psi)

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

class drift_diffusion_two_systems_holder(object):
    '''
    This object is used to generate the equation of motion of two systems,
    where transmission from system 1 to system 2 occurs via both a classical
    as well as a quantum channel. Here this is assumed to be done by splitting
    the output of system 1 with a beamsplitter, and making a heterodyne
    measurement on one of the outputs. The second output is then displaced by
    the measured state.

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
    def __init__(self, H1, H2, L1s, L2s, R, eps, n, tspan, trans_phase=None,
                 ops_on_whole_space = False):
        self.t_old = min(tspan) - 1.

        assert 0 <= R <= 1
        self.R = R
        self.T = np.sqrt(1 - R**2)
        self.eps = eps
        self.n = n

        if trans_phase is not None:
            self.eps *= trans_phase
            self.T *= trans_phase

        self.H1, self.H2, self.L1s, self.L2s = preprocess_operators(
            H1, H2, L1s, L2s, ops_on_whole_space)

        print ("done preparing operators...")

        self.L1psis = None
        self.L2psis = None
        self.l1s = None
        self.l2s = None

    def update_Lpsis_and_ls(self, psi, t):
        '''Updates Lpsis and ls.

        If t is different than t_old, update Lpsis, ls, and t_old.
        Otherwise, do nothing.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float
        '''
        if t != self.t_old:
            self.L1psis = [L1.dot(psi) for L1 in self.L1s]
            self.L2psis = [L2.dot(psi) for L2 in self.L2s]
            self.l1s = [L1psi.dot(psi.conj()) for L1psi in self.L1psis]
            self.l2s = [L2psi.dot(psi.conj()) for L2psi in self.L2psis]
            self.t_old = t

    def f_normalized(self, psi, t):
        '''Computes drift f.

        Assumes two systems. Computes an individual term for each,
        as well as a classical transmission and a quantum transmission term.

        Args:
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: array of shape (d,)
            to define the deterministic part of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)

        return (individual_term(self.H1, self.L1s, self.l1s, self.L1psis, psi)
            + individual_term(self.H2, self.L2s, self.l2s, self.L2psis, psi)
            + self.eps * self.R * classical_trans(
                self.L2s[0], self.l1s[0], self.l2s[0], self.L2psis[0], psi)
            + self.T * quantum_trans(
                self.L2s[0], self.l1s[0], self.l2s[0], self.L1psis[0], self.L2psis[0], psi)
               )

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
            psi0: Nx1 csr matrix, dtype = complex128
                input state.
            t: dtype = float

        Returns: returning an array of shape (d, m)
            to define the noise coefficients of the system.

        '''
        self.update_Lpsis_and_ls(psi, t)
        L1_cent = [L1psi - l1*psi for L1psi,l1 in zip(self.L1psis, self.l1s)]
        L2_cent = [L2psi - l2*psi for L2psi,l2 in zip(self.L2psis, self.l2s)]
        N1 = self.T*L1_cent[0] + L2_cent[0]
        N2 = self.n * self.eps * (-self.L2s[0].H.dot(psi) )
        N2_conj = (self.n * self.eps * (self.L2s[0].dot(psi) )
                + self.R * L1_cent[0])
        return np.vstack([N1, N2, N2_conj] + L1_cent[1:] + L2_cent[1:]).T

def insert_conj(dW, port=1):
    """Insert a conjugate noise term to channel `port'.

    the dW is an Nxm term of real-valued noise, where N
    is the t-span and m is twice number of physical channels.
    When measurement feedback occurs, we must sometimes
    include conjugate noise terms (e.g. both dW_j and dW_j^*).
    This can be done by including an extra channel with redundant
    information, which is done here.

    Args:
        dW: Nxm array
            Noise for the real and imaginary components of channels
        port: int
            which port should have its conjugate inserted.
            Note: the conjugate port is inserted immediately after
            the original port.

    """
    dW_real, dW_imag = np.split(dW.T, 2)
    return np.concatenate([np.concatenate([dW_real[:(1+port)],
                                    [dW_real[port]],
                                    dW_real[(1+port):]]),
                    np.concatenate([dW_imag[:(1+port)],
                                    [-dW_imag[port]],
                                    dW_imag[(1+port):]])]).T

def qsd_solve_two_systems(H1,
                          H2,
                          psi0,
                          tspan,
                          L1s,
                          L2s,
                          R,
                          eps,
                          n,
                          sdeint_method,
                          trans_phase=None,
                          obsq=None,
                          normalize_state=True,
                          downsample=1,
                          ops_on_whole_space = False,
                          multiprocessing = False,
                          ntraj=1,
                          processes=8,
                          seed=1,
                          implicit_type = None
                          ):
    '''
    Args:
        H1: N1xN1 csr matrix, dtype = complex128
            Hamiltonian for system 1.
        H2: N2xN2 csr matrix, dtype = complex128
            Hamiltonian for system 2.
        psi0: Nx1 csr matrix, dtype = complex128
            input state.
        tspan: numpy array, dtype = float
            Time series of some length T.
        L1s: list of N1xN1 csr matrices, dtype = complex128
            System-environment interaction terms (Lindblad terms) for system 1.
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
        obsq (optional): list of NxN csr matrices, dtype = complex128
            Observables for which to generate trajectory information.
            Default value is None (no observables).
        normalize_state (optional): Boolean
            Whether to numerically normalize the equation at each step.
        downsample: optional, integer to indicate how frequently to save values.
        ops_on_whole_space (optional): Boolean
            whether the Given L and H operators have been defined on the whole
            space or individual subspaces.
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
            ['psis'] -> np.array with shape = (ntraj,T,N) and dtype = complex128
            ['obsq_expects'] -> np.array with shape = (ntraj,T,len(obsq)) and dtype = complex128

    '''

    ## Check dimensions of inputs. These should be consistent with qutip Qobj.data.
    N = psi0.shape[0]
    if psi0.shape[1] != 1:
        raise ValueError("psi0 should have dimensions Nx1.")

    ## Determine seeds for the SDEs
    if type(seed) is list or type(seed) is tuple:
        assert len(seed) == ntraj
        seeds = seed
    elif type(seed) is int or seed is None:
        np.random.seed(seed)
        seeds = [np.random.randint(1000000) for _ in range(ntraj)]
    else:
        raise ValueError("Unknown seed type.")

    T_init = time()
    psi0_arr = np.asarray(psi0.todense()).T[0]
    x0 = np.concatenate([psi0_arr.real, psi0_arr.imag])
    drift_diffusion = drift_diffusion_two_systems_holder(
        H1, H2, L1s, L2s, R, eps, n, tspan, trans_phase=trans_phase,
        ops_on_whole_space=ops_on_whole_space)

    f = complex_to_real_vector(drift_diffusion.f_normalized)
    G = complex_to_real_matrix(drift_diffusion.G_normalized)


    def SDE_helper(args, s):
        '''Let's make different wiener increments for each trajectory'''
        m = 2 * (len(L1s) + len(L2s))
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

    psi0 = sparse.csr_matrix(([0,0,0,0,0,0,0,1.]),dtype=np.complex128).T
    H = sparse.csr_matrix(np.eye(8),dtype=np.complex128)
    Ls = [sparse.csr_matrix( np.diag([np.sqrt(i) for i in range(1,8)],k=1),dtype=np.complex128)]
    tspan = np.linspace(0, 10.0, 3000)
    obsq = [sparse.csr_matrix(np.diag([i for i in range(4)]*2), dtype=np.complex128)]

    ntraj = 50

    D = qsd_solve(H, psi0, tspan, Ls, sdeint.itoSRI2, obsq = obsq, ntraj = ntraj, normalized_equation=False, normalize_state=True)

    psis = D["psis"]
    obsq_expects = D["obsq_expects"]

    print ("Last point of traj 0: ", psis[0][-1])
    print ("Norm of last point in traj 0: ", la.norm(psis[0][-1]))  ## should be close to 1...

    for i in range(ntraj):
        plt.plot(tspan, obsq_expects[i,:,0].real,  linewidth=0.3)

    ave_traj = np.average(np.array([obsq_expects[i,:,0].real
        for i in range(ntraj)]), axis=0)
    plt.plot(tspan, ave_traj, color='k',  linewidth=2.0)
    plt.show()
