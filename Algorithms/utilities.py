"""
Functions for generating random SPD matrices and sampling multivariate Gaussians
"""

import numpy as np
from scipy import linalg
from scipy.stats import chi2


def random_matrix_generator(N, k=None, dist=1, diag=False, eig_mult=1, imaginary=False):
    """Creates a random SPD matrix three choices of eigenvalue distribution

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
    eig_mult : int, optional, default is 1
        Multiplicity of eigenvalues
    imaginary : bool, optional, default is False
        Whether matrix generated has an imaginary part or not


    Returns
    -------
    A : numpy array
        Matrix A
    SVD : dictionary
        Key "S" is vector of singular values of A
        Key "U" is matrix of singular vectors of A

    References
    ----------
    [1] Strakos, Z. "On the real convergence rate of the conjugate gradient
    method"
    DOI = 10.1016/0024-3795(91)90393-B

    [2] Greenbaum, A. "Estimating the Attainable Accuracy of Recursively
    Computed Residual Methods"
    DOI = 10.1137/S0895479895284944
    """

    if k is None:
        k = int(np.log10(N)) + 1

    #
    # Creating the matrix
    #

    SVD = {"S": np.zeros(N)}

    if not diag:
        A = np.random.randn(N, N)
        if imaginary:
            A = A + 1j * np.random.randn(N, N)
        SVD["U"], _ = linalg.qr(A)

    if dist == 1:
        # Eigenvalue distribution that causes round off errors
        rho = 0.9
        for i in range(0, N):
            SVD["S"][i] = 0.1 + ((i + 1) - 1) / (N - 1) * (10 ** (k - 1) - 0.1) * (
                rho ** (N - (i + 1))
            )
    elif dist == 2:
        # Geometric eigenvalue distribution
        for i in range(N):
            SVD["S"][i] = (10**k) ** ((i) / (N - 1))  # i-1+1 because of zero index
    else:
        # Random eigenvalue distribution
        SVD["S"] = np.sort(np.random.rand(N))[::-1]
        CondA = max(SVD["S"]) / min(SVD["S"])
        ShiftRatio = 10**k / CondA
        for i in range(N - 2, -1, -1):
            SVD["S"][i] = SVD["S"][N - 1] + ShiftRatio * (SVD["S"][i] - SVD["S"][N - 1])

    # Increase eigenvalue multiplicity if desired
    if eig_mult != 1:
        for i in range(len(SVD["S"])):
            if i % eig_mult != 0:
                SVD["S"][i] = SVD["S"][i - i % eig_mult]

    # Multiply by random orthogonal matrix if not wanting diagonal
    if not diag:
        A = SVD["U"] @ np.diag(SVD["S"]) @ SVD["U"].conj().T
    else:
        A = SVD["S"]

    #
    # Returning the values
    #

    return A, SVD


def mv_normal(Mean, SqrtCov, N_samples=1):
    """Samples from a multivariable normal distribution

    Parameters
    ----------
    Mean : numpy array
        Mean vector
    SqrtCov : numpy array
        Square Root or Cholesky Factor of Covariance matrix
    N_samples : int (default is 1)
        Number of samples from the distribution to compute

    Returns
    -------
    x : numpy array
        Sample from normal distribution N(Mean,SqrtCov @ SqrtCov.conj().T)
        If N_samples > 1, then each row of x is a sample
    """
    N = SqrtCov.shape[1]
    if N_samples == 1:
        x = (SqrtCov @ np.random.randn(N)) + Mean
    else:
        x = np.zeros((N_samples, N))
        for i in range(N_samples):
            x[i] = (SqrtCov @ np.random.randn(N)) + Mean
    return x


def mvt(mean, sqrt_cov, dof):
    """Samples from a multivariable T distribution

    Parameters
    ----------
    mean : numpy array
        Mean vector of the distribution
    sqrt_cov : numpy array
        Square root or Cholesky factor of scale matrix
    dof : scalar
        Degrees of freedom of the distribution

    Returns
    -------
    x : numpy array
        Sample from multivariate T distribution
        MVT_dof(mean,SqrtCov @ SqrtCov.conj().T)
    """
    N = len(mean)
    x = mv_normal(np.zeros(N), sqrt_cov)
    u = chi2.rvs(dof)

    return x / np.sqrt(u / dof) + mean


def ecdf(x, data):
    """ecdf
    This function evaluates the empirical cumulative distribution function of
    the input "data" at the value "x"
    """
    N = len(data)
    ecdf_inner = lambda y: len(data[data <= y]) / N
    cdf_eval = map(ecdf_inner, x)
    return np.array(list(cdf_eval))


def ecdf_shift(x, data):
    """ecdf_shift
    This function evaluates the empirical cumulative distribution function of
    the input "data" at the value "x" minus epsilon for a very small epsilon
    """
    N = len(data)
    ecdf_inner = lambda y: len(data[data < y]) / N
    cdf_eval = map(ecdf_inner, x)
    return np.array(list(cdf_eval))
