'''
Functions for generating random SPD matrices and sampling multivariate Gaussians
'''

import numpy as np
from scipy import linalg

def random_matrix_generator(N,k = None,dist = 1,diag = False):
    '''Creates a random SPD matrix three choices of eigenvalue distribution

    Parameters
    ----------
    N : int
        Matrix size
    k : int, optional, default is log10(N)+1
        Power of 10 of matrix condition number
    dist : {0, 1, 2}, optional, default is 1
        Eigenvalue distribution choice. Options:
        0 -- Random positive real
        1 -- Distribution that causes round off errors in CG [1]
        2 -- Geometric [2]
    diag : bool, optional, default is False
        Whether to return vectior of eigenvalues instead of matrix
        True is useful if you want a big matrix
            
        
    Returns
    -------
    A : numpy array
        Matrix A
    SqrtA : numpy array
        Square root of A
    InvA : numpy array
        Inverse of A
    InvSqrtA : numpy array
        Inverse square root of A
    MinS : float
        Smallest eigenvalue of A
    MaxS : float
        Largest eigenvalue of A

    References
    ----------
    [1] Strakos, Z. "On the real convergence rate of the conjugate gradient 
    method"
    DOI = 10.1016/0024-3795(91)90393-B

    [2] Greenbaum, A. "Estimating the Attainable Accuracy of Recursively 
    Computed Residual Methods"
    DOI = 10.1137/S0895479895284944
    '''
    
    if k is None:
        k = int(np.log10(N))+1
            
    #
    # Creating the matrix
    #

    SVD = {'S' : np.zeros(N)}

    if not diag:
        A = np.random.rand(N,N)
        SVD['U'],_ = linalg.qr(A)

    

    if dist == 1:
        # Eigenvalue distribution that causes round off errors
        rho = 0.9
        for i in range(0,N):
            SVD['S'][i] = 0.1+((i+1)-1)/(N-1)*(10**(k-1)-0.1)*(rho**(N-(i+1)))
    elif dist == 2:
        # Geometric eigenvalue distribution
        for i in range(N):
            SVD['S'][i] = (10**k)**((i)/(N-1)) # i-1+1 because of zero index
    else:
        # Random eigenvalue distribution
        SVD['S'] = np.sort(np.random.rand(N))[::-1]
        CondA = max(SVD['S'])/min(SVD['S'])
        ShiftRatio = 10**k/CondA
        for i in range(N-2,-1,-1):
            SVD['S'][i] = SVD['S'][N-1]+ShiftRatio*(SVD['S'][i]-SVD['S'][N-1])

    # Multiply by random orthogonal matrix if not wanting diagonal
    if not diag:
        A = SVD['U']@np.diag(SVD['S'])@SVD['U'].T
    else:
        A = SVD['S']
    
    #
    # Returning the values
    #
    
    return A, SVD


def mv_normal(Mean,SqrtCov):
    '''Samples from a multivariable normal distribution

    Parameters
    ----------
    Mean : numpy array
        Mean vector
    SqrtCov : numpy array
        Square Root or Cholesky Factor of Covariance matrix

    Returns
    -------
    x : numpy array
        Sample from normal distribution N(Mean,SqrtCov @ SqrtCov.T)
    '''
    N = SqrtCov.shape[1]
    return (SqrtCov @ np.random.randn(N)) + Mean

