'''
This module has our implementations of BayesCG under the Krylov Prior
'''

import numpy as np
from scipy import linalg
from utilities import mv_normal

def bayescg_k(A, b, x, post_rank=5, max_it=None, tol=1e-6, l_tol=1e-32,\
             samples = None, reorth=False, NormA=None, xTrue=None):
    ''' Computes solution to Ax = b with BayesCG with Krylov Prior

    This program iteratively solves a symmetric positive definite system 
    of linear equations with a with the Bayesian Conjugate Gradient method
    under the Krylov prior, a probabilistic numerical method. This 
    implementation is based on the Conjugate Gradient method.
            
    Parameters
    ----------
    A : function
        Function that computes the matvec of A
    b : numpy array
        Right hand side vector from equation Ax = b
    x : numpy array
        Initial guess for x in equation Ax = b
    post_rank : int, optional, default is 5
        Rank of Krylov Posterior Covariance
    max_it : int, optional, default is size of A
        Maximum amount of iterations to run
    tol : float, optional, default is 1e-6
        Convergence tolerance
    l_tol : float, optional, default is 1e-32
        Minimum norm of residuals used for creating posterior covariance
        This determines maximum Krylov space dimension
    samples : int or None, optional, default is None
        Number of S-statistic samples to compute each iteration
    reorth : bool, optional, default is False
        Whether to reorthogonalize 
    NormA : float or None, optional, default is None 
        2-norm of matrix A
        If supplied, residual computed as (||r||/(||A|| ||x_m||))
        If not, residual is (||r||/||b||)
    xTrue : numpy array or None, optional, default is None
        True solution
        If supplied, more convergence information is returned

    Returns
    -------
    x : numpy array
        Posterior mean
    V : numpy array
        Factor of posterior covariance
    Phi : numpy array
        Factor of posterior covariance
    info : dict
        Dictionary containing convergence information
        Dictionary keys always returned:
            'res' : Residual history
            'sExp' : Expected value of S statistic at each iteration
            'sSD' : Standard deviation of S statistic at each iteration
        Additional key if samples is not None
            'samples' : (samples by m numpy array) Samples from S statistic
        Additional keys if xTrue is supplied
            'err' : Error history
            'actual_res' : Actual residual, b-Ax, history 
                (as opposed to recursively computed residual)
    '''
    
    #
    #Define the variables
    #
    
    #Size of the system
    N = len(x)
    
    #Default Maximum Iterations
    if max_it is None:
        max_it = N
    
    #Residual and first search direction
    r = b - A(x)
    
    # Setting Up Posterior Covariance
    V = np.zeros((N,post_rank+1))
    V[:,0] = r

    # CG Recursion values
    # Values for post_rank iterations are stored. Newest first, oldest last
    gamma = np.zeros(post_rank+1)
    vIP = np.copy(gamma)
    rIP = np.copy(gamma)
    rNorm = np.copy(gamma)

    rIP[0] = np.inner(r,r)
    rNorm[0] = np.sqrt(rIP[0])

    # Setting up convergence history
    res = np.zeros(max_it+1)
    if (NormA is None) or (xTrue is None):
        bNorm = linalg.norm(b)
        res[0] = rNorm[0]/bNorm
    if xTrue is not None:
        xNorm = linalg.norm(xTrue)
        err = np.zeros(max_it+1)
        err[0] = np.inner(x-xTrue,A(x-xTrue))
        if NormA is not None:
            xNormANorm = xNorm*NormA
            res[0] = rNorm[0]/xNormANorm
        res2 = np.copy(res)

    # Expected S Statistic values
    sExp = np.zeros(max_it+1)
    sSD = np.zeros(max_it+1)

    # Setting up reorthogonalization
    r_hist = np.zeros((N,max_it+post_rank+1))
    r_hist[:,0] = r/rNorm[0]

    # Setting up sampling
    if samples is not None:
        S = np.zeros((samples,max_it+1))

    xit = 0
    countdown = None
    
    #
    # Iterating Through CG
    #
    # This program stores information from post_rank+1 CG iterations
    # The oldest recursion values are used to compute posterior mean iterates
    # The remaining are used to compute the posterior covariance factors
    #
    
    for i in range(max_it+post_rank):

        # No matrix vector products are computed once max Krylov dim is reached
        if countdown is None:
            Av = A(V[:,0])
            vIP[0] = np.inner(V[:,0],Av)
        else:
            Av = np.zeros((N,1))
            vIP[0] = 1

        # Step length
        gamma[0] = rIP[0]/vIP[0]

        # S statistic computed for iteration i - post-rank
        if i >= post_rank - 1:

            # S statistic expected value and standard deviation
            sExp[xit] = sum(gamma[:-1]*rIP[:-1])
            sSD[xit] = np.sqrt(2*sum((gamma[:-1]*rIP[:-1])**2))

            # S statistic samples if desired
            if samples is not None:
                Phi = np.sqrt(gamma[:-1]*rIP[:-1]/vIP[:-1])
                for j in range(samples):
                    x_sample = mv_normal(x,V[:,:-1]*Phi)
                    x_diff = x_sample - x
                    S[j,xit] = np.inner(x_diff,A(x_diff))

        # Stop when out of search directions, convergence, or max iterations
        if (countdown is 0) or (i == max_it+post_rank-1) \
           or (tol is not None and res[xit] < tol):
            break

        # Rotate CG recursion values by 1 (moves newest to second position etc)
        rIP = np.roll(rIP,1)
        rNorm = np.roll(rNorm,1)
        V = np.roll(V,1,1)
        gamma = np.roll(gamma,1)
        vIP = np.roll(vIP,1)

        # If not at max Krylov dim then new residuals are created
        if rIP[1] >= l_tol:
            r = r - gamma[1]*Av
            if reorth:
                r = r - r_hist[:,:i+1]@(r_hist[:,:i+1].T@r)
                r = r - r_hist[:,:i+1]@(r_hist[:,:i+1].T@r)
        else:
            # Max Krylov dimension. No more CG recursions.
            # Mean computed with remaining post_rank-1 unused search directions
            if countdown is None:
                print('Max Krylov dimension. No new search directions')
                countdown = post_rank - 1
            else:
                countdown = countdown - 1
            r = np.zeros(N)

        # Save newest residual value in first position
        rIP[0] = np.inner(r,r)
        rNorm[0] = np.sqrt(rIP[0])

        # New mean iterates computed after post_rank iterations
        if i >= post_rank - 1:
            # Compute Next Iterate
            xit = xit + 1
            x = x + gamma[-1]*V[:,-1]

            # Compute Residual Norm
            if xTrue is not None:
                err[xit] = np.inner(xTrue-x,A(xTrue-x))
                rTrueNorm = linalg.norm(b-A(x))
                if NormA is not None:
                    res[xit] = rNorm[-2]/xNormANorm
                    res2[xit] = rTrueNorm/xNormANorm
                else:
                    res2[xit] = rTrueNorm/bNorm
            if NormA is None:
                res[xit] = rNorm[-2]/bNorm
            elif xTrue is None:
                res[xit] = rNorm[-2]/linalg.norm(x)/NormA

        # Compute new search direction if max Krylov dim not reached
        if countdown is None:
            delta = rIP[0]/rIP[1]
            if reorth:
                r_hist[:,i+1] = r/rNorm[0]
        else:
            delta = 0

        V[:,0] = r + delta*V[:,1]
        
    #
    # Compute posterior covariance
    #

    V = V[:,-2::-1]*(np.sqrt(vIP[-2::-1])**-1)
    Phi = gamma[-2::-1]*rIP[-2::-1]

    # Return maximum avaliable posterior covariance
    if countdown is not None:
        countdown = max(countdown-1,1)
        V = V[:,:countdown]
        Phi = Phi[:countdown]

    #
    # Return the results
    #
    
    info = {'res':res[:xit+1]}

    if xTrue is not None:
        info['err'] = err[:xit+1]
        info['actual_res'] = res2[:xit+1]

    info['sExp'] = sExp[:xit+1]
    info['sSD'] = sSD[:xit+1]

    if samples is not None:
        info['samples'] = S[:,:xit+1]

    return x, V, Phi, info

