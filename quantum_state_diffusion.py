'''
Gil Tabak

Nov 3, 2016

This notebook uses the library sdeint to perform quantum state diffusion trajectories.
The inputs are purposely similar to qubit functions like mcsolve to make
integration easier later.

'''

import numpy as np
import sdeint
from scipy import sparse
import numpy.linalg as la
from time import time
import matplotlib as mil
mil.use('TkAgg')
import matplotlib.pyplot as plt


def qsd_solve(H, psi0, tspan, Ls, sdeint_method, obsq = None, normalized = True, ntraj=1,):
    '''
    Args:
        H: NxN csr matrix, dtype = complex128
            Hamiltonian.
        psi0: Nx1 csr matrix, dtype = complex128
            input state.
        tspan: numpy array, dtype = float
            Time series.
        Ls: list of NxN csr matrices, dtype = complex128
            System-environment interaction terms (Lindblad terms).
        sdeint_method (Optional) SDE solver method:
            Which SDE solver to use. Default is sdeint.itoSRI2.
        obsq (optional): list of NxN csr matrices, dtype = complex128
            Observables for which to generate trajectory information
        normalized (optional): Boolean
            Use the normalized quantum state diffusion equations. (TODO: case False)
        ntraj (optional): int
            number of trajectories. (TODO)

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

    '''
    We include a way to update L*psi and l = <psi,L,psi> when t changes.
    This makes the code somewhat more efficient since these values are used
    both for the drift f and the diffusion G terms.
    '''
    global t_old
    global Lpsis
    global ls
    t_old = min(tspan) - 1.
    def update_Lpsis_and_ls(psi,t):
        global t_old
        global Lpsis
        global ls
        if t != t_old:
            Lpsis = [L.dot(psi) for L in Ls]
            ls = [Lpsi.dot(psi.conj()) for Lpsi in Lpsis]
            t_old = t

    if normalized: ## We'll include an option for non-normalized equations later...
        def f(psi,t):
            update_Lpsis_and_ls(psi,t)
            return (-1j * H.dot(psi)
                    - sum([ 0.5*(L.H.dot(Lpsi) + np.conj(l)*l*psi)
                              - np.conj(l)*(Lpsi) for L,l,Lpsi in zip(Ls,ls,Lpsis)]) )
        def G(psi,t):
            update_Lpsis_and_ls(psi,t)
            complex_noise = np.vstack([Lpsi - l*psi for Lpsi,l in zip(Lpsis,ls)]) / np.sqrt(2.)
            return np.vstack([complex_noise.real, 1j*complex_noise.imag]).T
    else:
        raise ValueError("Case normalized == False is not implemented.")

    psi0_arr = np.asarray(psi0.todense()).T[0]
    psis = sdeint_method(f,G,psi0_arr,tspan)

    ## maybe there is a more efficient way to do this...
    obsq_expects = [ np.asarray([ob.dot(psi).dot(psi.conj()) for psi in psis])
                        for ob in obsq ]

    return psis, obsq_expects

if __name__ == "__main__":

    psi0 = sparse.csr_matrix(([0,0,0,0,0,0,0,1.]),dtype=np.complex128).T
    H = sparse.csr_matrix(np.eye(8),dtype=np.complex128)
    Ls = [sparse.csr_matrix( np.diag([np.sqrt(i) for i in range(1,8)],k=1),dtype=np.complex128)]
    tspan = np.linspace(0,0.5,5000)
    obsq = [sparse.csr_matrix(np.diag([i for i in range(4)]*2),dtype=np.complex128)]

    T_init = time()
    psis,obsq_expects = qsd_solve(H, psi0, tspan, Ls, sdeint.itoSRI2, obsq = obsq)
    T_fin = time()

    print ("time to run:  ", T_fin - T_init, " seconds.")
    print ("Last point: ",psis[-1])
    print ("Norm of last point: ",la.norm(psis[-1]))  ## should be close to 1...
    plt.plot(tspan,obsq_expects[0])
    plt.show()
