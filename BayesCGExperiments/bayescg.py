''' 
This module has our implementation of the Bayesian Conjugate Gradient Method
'''

import numpy as np
from scipy import linalg

def bayescg(A, b, x, Sig, max_it=None, tol=1e-6, \
            reorth=True, NormA=None, xTrue=None):
    ''' Computes solution to Ax = b with Bayesian Conjugate Gradient Method

    This program iteratively solves a symmetric positive definite system
    of linear equations with a with the Bayesian Conjugate Gradient method, 
    a probabilistic numerical method originally presented in [1]. 

    Parameters
    ----------
    A : function
        Function that computes matvec of A
    b : numpy array
        Right hand side vector from equation Ax = b
    x : numpy array
        Prior mean vector
    Sig : function
        Function that computes matvec of Sigma_0, the prior covariance 
    max_it : int, optional, default is size of A
        Maximum amount of iterations to run
    tol : float, optional, default is 1e-6
        Convergence tolerance
    reorth : bool, optional, default is True
        Whether to reorthogonalize, called "batch computed search 
        directions" in [1], however we reorthogonalize residuals instead of 
        search directions.
    NormA : float or None, optional, default is None
        2-Norm of matrix A
        If supplied, residual computed as (||r||/(||A|| ||x_m||))
        If not, residual is (||r||/||b||)
    xTrue : numpy array or None, optional, default is None
        True solution of linear system
        If supplied, more convergence information is returned

    Returns
    -------
    x : numpy array
        Posterior mean
    SigAs : (n by m) numpy array 
        Vectors Sigma_0As_i, 1<=i<=m, to compute posterior covariance
    info : dict
        Dictionary containing convergence information
        Dictionary keys always returned:
            'res' : Residual history
            'search_dir' : Search directions used to compute posterior
        Additional keys if xTrue is supplied
            'err' : A-Norm Error history [ || x_* - x_m || ]
            'trace' : Trace history [ trace(A Sigma_m) ]
            'actual_res' : Actual residual, b-Ax, history 
                (as opposed to recursively computed residual)

    References
    ----------
    [1] Cockayne, J; Oates, C. J.; Ipsen, I. C. F.; Girolami, M. "A Bayesian
    Conjugate Gradient Method (with Discussion)"
    DOI = 10.1214/19-BA1145
    '''
    
    #
    #Here we define the variables
    #
    
    #Size of the system
    N = len(x)
    
    #Default Maximum Iterations
    if max_it is None:
        max_it = N
    
    #Residual and first search direction
    r = np.zeros((N,max_it+1))
    r[:,0] = b-A(x)
    S = np.copy(r)
    
    #Inner products
    rIP = np.zeros(max_it+1)
    rIP[0] = np.inner(r[:,0],r[:,0])
    sIP = np.zeros(max_it)
    
    #Array holding matrix-vector products
    SigAs = np.zeros((N,max_it))
    
    # Convergence information
    # If xTrue is supplied, more information is computed
    rNorm = np.sqrt(rIP[0])
    Res2 = np.zeros(max_it+1)
    if (NormA is None) or (xTrue is None):
        bNorm = linalg.norm(b)
        Res = rNorm/bNorm
        Res2[0] = Res
    if xTrue is not None:
        xNorm = linalg.norm(xTrue)
        err_hist = np.zeros(max_it+1)
        err_hist[0] = np.inner(x-xTrue,A(x-xTrue))
        if NormA is not None:
            xNormANorm = linalg.norm(xTrue)*NormA
            Res = rNorm/xNormANorm
            Res2[0] = Res
        Res3 = np.copy(Res2)
        tr_hist = np.zeros(max_it+1)
        tr_hist[0] = np.trace(A(Sig(np.eye(N))))
    
    i = 0
    
    
    #
    #Iterating Through Bayesian Conjugate Gradient
    #
    
    while i < max_it and (tol is None or Res > tol):
        
        #Compute Matrix Vector Products
        As = A(S[:,i])
        SigAs[:,i] = Sig(As)
        ASigAs = A(SigAs[:,i])
        
        # Search Direction Inner Product
        sIP[i] = np.abs(np.inner(S[:,i],ASigAs))
        
        #Calculate next x
        alpha = rIP[i]/sIP[i]
        x = x+alpha*SigAs[:,i]
        
        #Calculate New Residual
        r[:,i+1] = r[:,i] - alpha*ASigAs
        
        if reorth:
            # Reorthogonalize Residual
            r[:,i+1] = r[:,i+1] - (rIP[:i+1]**-1)*r[:,:i+1]\
                @(r[:,:i+1].T@r[:,i+1])
            r[:,i+1] = r[:,i+1] - (rIP[:i+1]**-1)*r[:,:i+1]\
                @(r[:,:i+1].T@r[:,i+1])
        
                
        # Compute Residual Norms
        rIP[i+1] = np.inner(r[:,i+1],r[:,i+1])
        rNorm = np.sqrt(rIP[i+1])
        if xTrue is not None:
            err_hist[i+1] = np.inner(x-xTrue,A(x-xTrue))
            tr_hist[i+1] = tr_hist[i] - np.trace(A(np.outer( \
                SigAs[:,i],SigAs[:,i])))/sIP[i]
            rTrueNorm = linalg.norm(b-A(x))
            if NormA is not None:
                Res = rNorm/xNormANorm
                Res3[i+1] = rTrueNorm/xNormANorm
            else:
                Res3[i+1] = rTrueNorm/bNorm    
        if NormA is None:
            Res = rNorm/bNorm
        elif xTrue is None:
            Res = rNorm/NormA/linalg.norm(x)
        Res2[i+1] = np.copy(Res)
            
        #Calculate next search direction
        beta = rIP[i+1]/rIP[i]
        S[:,i+1] = r[:,i+1]+beta*S[:,i]
        
        i = i+1
    
    #
    #Return the results
    #

    info = {'res':Res2[:i+1]}
    info['search_dir'] = (sIP[:i]**(-1/2))*S[:,:i]

    if xTrue is not None:
        info['actual_res'] = Res3[:i+1]
        info['err'] = err_hist[:i+1]
        info['trace'] = tr_hist[:i+1]
    
    return x, (sIP[:i]**(-1/2))*SigAs[:,:i], info


def bayescg_post_cov(Sig,SigAs):
    ''' Computes BayesCG posterior covariance from output of bayescg

    This program computes the posterior covariance with a rank-m downdate
    if given the prior covariance matrix and the vectors SigmaAs_i, 1<=i<=m

    Parameters
    ----------
    Sig : function
        Function that computes matvec of prior covariance
    SigAs : (m by n) numpy array
        Matrix of vectors SigmaAs_i, 1<=i<=m, to compute posterior as rank-m
    downdate of prior. This matrix is returned by program bayescg

    Returns
    -------
    Posterior covariance matrix as a numpy array
    '''
    N = SigAs.shape[0]
    return Sig(np.eye(N)) - SigAs@(SigAs.T@np.eye(N))

