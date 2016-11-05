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
            Observables for which to generate trajectory information (TODO)
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
    result = sdeint_method(f,G,psi0_arr,tspan)

    return result

if __name__ == "__main__":

    psi0 = sparse.csr_matrix(([0,0,0,0,0,0,0,1.]),dtype=np.complex128).T
    H = sparse.csr_matrix(np.diag([1,1,1,1,1,1,1,1]),dtype=np.complex128)
    Ls = [sparse.csr_matrix( [[0,1,0,0,0,0,0,0],
                                  [0,0,np.sqrt(2),0,0,0,0,0],
                                  [0,0,0,np.sqrt(3),0,0,0,0],
                                  [0,0,0,0,np.sqrt(4),0,0,0],
                                  [0,0,0,0,0,np.sqrt(5),0,0],
                                  [0,0,0,0,0,0,np.sqrt(6),0],
                                  [0,0,0,0,0,0,0,np.sqrt(7)],
                                  [0,0,0,0,0,0,0,0]],dtype=np.complex128)]
    tspan = np.linspace(0,0.5,5000)

    T_init = time()
    output = qsd_solve(H, psi0, tspan, Ls, sdeint.itoSRI2)
    T_fin = time()

    print ("time to run:  ", T_fin - T_init, " seconds.")

    print ("Last point: ",output[-1])
    print ("Norm of last point: ",la.norm(output[-1]))  ## should be close to 1...


###### Deprecated, but could still be useful

    # def doubled_to_complex(psi2):
    #     '''
    #     Convertes from:
    #         <(2N)xM sparse matrix of type '<class 'numpy.float64'>'
    #             in Compressed Sparse Row format>
    #     to:
    #         <NxM sparse matrix of type '<class 'numpy.complex128'>'
    #     	    in Compressed Sparse Row format>
    #
    #     This function ASSUMES that the row dimension of the input is 2N
    #     (specifically, even).
    #     '''
    #     return psi2[:N,:] + 1j*psi2[N:,:]
    #
    # def complex_to_doubled(psi):
    #     '''
    #     Convertes from:
    #         <NxM sparse matrix of type '<class 'numpy.complex128'>'
	#            in Compressed Sparse Row format>
    #     to:
    #         <(2N)xM sparse matrix of type '<class 'numpy.float64'>'
    #            in Compressed Sparse Row format>
    #
    #     CSR format is chosen for faster splitting along the rows.
    #     '''
    #     return sparse.vstack([psi.real,psi.imag],format = 'csr')
    #
    # def generate_sdeint_functions(complex_func):
    #     def arr_func(psi_arr,t):
    #         psi_as_doubled_sparse_matix = sparse.csr_matrix(psi_arr).T
    #         psi_as_complex_sparse_matix = doubled_to_complex(psi_as_doubled_sparse_matix)
    #         complex_func_val = complex_func(psi_as_complex_sparse_matix,t)
    #         doubled_func_val = complex_to_doubled(complex_func_val)
    #         dense_func_val = doubled_func_val.todense()
    #         arr_func_val = np.asarray(dense_func_val).T[0]
    #         return arr_func_val
    #     return arr_func
