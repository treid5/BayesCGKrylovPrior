""" 
This module has our implementation of the Bayesian Conjugate Gradient Method
"""

import numpy as np
from scipy import linalg
from .a_lanczos import *


def bayescg(
    A,
    b,
    x,
    Sig,
    max_it=None,
    tol=1e-6,
    delay=None,
    reorth=True,
    NormA=None,
    xTrue=None,
    SqrtSigTranspose=None,
):
    """Computes solution to Ax = b with Bayesian Conjugate Gradient Method

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
    delay : float or None, optional, default is None
        Number of previous iteration step sizes to add together to compute
        posterior covariance scale
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
    SqrtSigTranspose : function or None, optional, default is None
        Function that computes matrix vector product of transpose of
        factorization of prior covariance times a vector.
        If not None, then a factorization of the posterior is computed
        If not None, then Sig must be a function that computes a factorization
        of the prior times a vector.
        If not None, prior covariance must have form Sigma_0 = F F^T, where
        Sig = F and SqrtSigTranspose = F^T

    Returns
    -------
    x : numpy array
        Posterior mean
    SigAs_hist : (m by n) numpy array
        Vectors Sigma_0As_i, 1<=i<=m, to compute posterior covariance
    info : dict
        Dictionary containing convergence information
        Dictionary keys always returned:
            'res' : Residual history
            'search_dir' : Search directions used to compute posterior
        Additional keys if xTrue is supplied
            'err' : Error history
            'trace' : Trace history [ trace(A Sigma_m) ]
            'actual_res' : Actual residual, b-Ax, history
                (as opposed to recursively computed residual)

    References
    ----------
    [1] Cockayne, J; Oates, C. J.; Ipsen, I. C. F.; Girolami, M. "A Bayesian
    Conjugate Gradient Method (with Discussion)"
    DOI = 10.1214/19-BA1145
    """

    #
    # Here we define the variables
    #

    # Size of the system
    N = len(x)

    # Set data type. Type of A is probably same as type of Az, were z is random
    data_type = np.double

    # Default Maximum Iterations
    if max_it is None:
        max_it = N

    # Residual and first search direction
    r = np.zeros((N, max_it + 1), dtype=data_type)
    r[:, 0] = b - A(x)
    S = np.copy(r)

    # Inner products
    rIP = np.zeros(max_it + 1, dtype=data_type)
    rIP[0] = np.inner(r[:, 0].conj(), r[:, 0])
    sIP = np.zeros(max_it, dtype=data_type)

    # Array holding matrix-vector products
    SigAs_hist = np.zeros((N, max_it), dtype=data_type)

    # Convergence information
    # If xTrue is supplied, more information is computed
    rNorm = np.sqrt(rIP[0])
    Res2 = np.zeros(max_it + 1, dtype=data_type)
    if (NormA is None) or (xTrue is None):
        bNorm = linalg.norm(b)
        Res = rNorm / bNorm
        Res2[0] = Res
    if xTrue is not None:
        xNorm = linalg.norm(xTrue)
        err_hist = np.zeros(max_it + 1, dtype=data_type)
        err_hist[0] = np.inner((x - xTrue).conj(), A(x - xTrue))
        if NormA is not None:
            xNormANorm = linalg.norm(xTrue) * NormA
            Res = rNorm / xNormANorm
            Res2[0] = Res
        Res3 = np.copy(Res2)
        tr_hist = np.zeros(max_it + 1, dtype=data_type)
        tr_hist[0] = np.trace(A(Sig(np.eye(N))))

    i = 0

    #
    # Iterating Through Bayesian Conjugate Gradient
    #

    while i < max_it and (tol is None or Res > tol):
        # Compute Matrix Vector Products
        As = A(S[:, i])
        if SqrtSigTranspose is not None:
            SigAs_hist[:, i] = SqrtSigTranspose(As)
            SigAs = Sig(SigAs_hist[:, i])
        else:
            SigAs_hist[:, i] = Sig(As)
            SigAs = SigAs_hist[:, i]
        ASigAs = A(SigAs)

        # Search Direction Inner Product
        sIP[i] = np.abs(np.inner(S[:, i].conj(), ASigAs))

        # Calculate next x
        alpha = rIP[i] / sIP[i]
        x = x + alpha * SigAs

        # Calculate New Residual
        r[:, i + 1] = r[:, i] - alpha * ASigAs

        if reorth:
            # Reorthogonalize Residual
            r[:, i + 1] = r[:, i + 1] - (
                (rIP[: i + 1] ** -1)
                * r[:, : i + 1]
                @ (r[:, : i + 1].conj().T @ r[:, i + 1])
            )
            r[:, i + 1] = r[:, i + 1] - (
                (rIP[: i + 1] ** -1)
                * r[:, : i + 1]
                @ (r[:, : i + 1].conj().T @ r[:, i + 1])
            )

        # Compute Residual Norms
        rIP[i + 1] = np.inner(r[:, i + 1].conj(), r[:, i + 1])
        rNorm = np.sqrt(rIP[i + 1])
        if xTrue is not None:
            err_hist[i + 1] = np.inner((x - xTrue).conj(), A(x - xTrue))
            tr_hist[i + 1] = (
                tr_hist[i] - np.trace(A(np.outer(SigAs.conj(), SigAs))) / sIP[i]
            )

            rTrueNorm = linalg.norm(b - A(x))
            if NormA is not None:
                Res = rNorm / xNormANorm
                Res3[i + 1] = rTrueNorm / xNormANorm
            else:
                Res3[i + 1] = rTrueNorm / bNorm
        if NormA is None:
            Res = rNorm / bNorm
        elif xTrue is None:
            Res = rNorm / NormA / linalg.norm(x)
        Res2[i + 1] = np.copy(Res)

        # Calculate next search direction
        beta = rIP[i + 1] / rIP[i]
        S[:, i + 1] = r[:, i + 1] + beta * S[:, i]

        i = i + 1

    #
    # Return the results
    #

    info = {"res": Res2[: i + 1]}
    info["search_dir"] = (sIP[:i] ** (-1 / 2)) * S[:, :i]

    if xTrue is not None:
        info["actual_res"] = Res3[: i + 1]
        info["err"] = err_hist[: i + 1]
        info["trace"] = tr_hist[: i + 1]

    if delay is not None:
        delay = min(delay, i)
        post_scale = np.sum(rIP[i - delay : i] ** 2 / sIP[i - delay : i])
        info["scale"] = post_scale

    return x, (sIP[:i] ** (-1 / 2)) * SigAs_hist[:, :i], info


def bayescg_random_search(
    A,
    b,
    x,
    Sig,
    max_it=None,
    tol=1e-6,
    reorth=True,
    NormA=None,
    xTrue=None,
    SqrtSigTranspose=None,
    seed=None,
):
    """Solves Ax = b with BayesCG using a random initial search direction

    Informative Description

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
    NormA : float or None, optional, default is None
        2-Norm of matrix A
        If supplied, residual computed as (||r||/(||A|| ||x_m||))
        If not, residual is (||r||/||b||)
    xTrue : numpy array or None, optional, default is None
        True solution of linear system
        If supplied, more convergence information is returned
    SqrtSigTranspose : function or None, optional, default is None
        Function that computes matrix vector product of transpose of
        factorization of prior covariance times a vector.
        If not None, then a factorization of the posterior is computed
        If not None, then Sig must be a function that computes a factorization
        of the prior times a vector.

    Returns
    -------
    x : numpy array
        Posterior mean
    SigAs_hist : (m by n) numpy array
        Vectors Sigma_0As_i, 1<=i<=m, to compute posterior covariance
    info : dict
        Dictionary containing convergence information
        Dictionary keys always returned:
            'res' : Residual history
            'search_dir' : Search directions used to compute posterior
        Additional keys if xTrue is supplied
            'err' : Error history
            'actual_res' : Actual residual, b-Ax, history
                (as opposed to recursively computed residual)

    References
    ----------
    [1] Cockayne, J; Oates, C. J.; Ipsen, I. C. F.; Girolami, M. "A Bayesian
    Conjugate Gradient Method (with Discussion)"
    DOI = 10.1214/19-BA1145
    """

    #
    # Here we define the variables
    #

    # Set data type. Type of A is probably same as type of Az, were z is random
    data_type = np.double

    # Size of the system
    N = len(x)

    # Default Maximum Iterations
    if max_it is None:
        max_it = N

    # Residual and first search direction
    if seed is not None:
        np.random.seed(seed)
    r = b - A(x)
    inv_check = np.random.rand(N)
    if SqrtSigTranspose is None:
        if linalg.norm(inv_check - A(Sig(inv_check))) < 1e-12:
            ASigA_func = A
        else:
            ASigA_func = lambda w: A(Sig(A(w)))
    else:
        if linalg.norm(inv_check - A(Sig(SqrtSigTranspose(inv_check)))) < 1e-12:
            ASigA_func = A
        else:
            ASigA_func = lambda w: A(Sig(SqrtSigTranspose(A(w))))
    S = a_lanczos(ASigA_func, np.random.randn(N), N, 1e-15, reorth)
    max_it = min(max_it, S.shape[1])

    # Inner products
    rIP = np.inner(r.conj(), r)

    # Array holding matrix-vector products
    SigAs_hist = np.zeros((N, max_it), dtype=data_type)

    # Convergence information
    # If xTrue is supplied, more information is computed
    rNorm = np.sqrt(rIP)
    Res2 = np.zeros(max_it + 1, dtype=data_type)
    if (NormA is None) or (xTrue is None):
        bNorm = linalg.norm(b)
        Res = rNorm / bNorm
        Res2[0] = Res
    if xTrue is not None:
        xNorm = linalg.norm(xTrue)
        err_hist = np.zeros(max_it + 1, dtype=data_type)
        err_hist[0] = np.inner((x - xTrue).conj(), A(x - xTrue))
        if NormA is not None:
            xNormANorm = linalg.norm(xTrue) * NormA
            Res = rNorm / xNormANorm
            Res2[0] = Res
        Res3 = np.copy(Res2)

    i = 0

    #
    # Iterating Through Bayesian Conjugate Gradient
    #

    while i < max_it and (tol is None or Res > tol):
        # Compute Matrix Vector Products
        As = A(S[:, i])
        if SqrtSigTranspose is not None:
            SigAs_hist[:, i] = SqrtSigTranspose(As)
            SigAs = Sig(SigAs_hist[:, i])
        else:
            SigAs_hist[:, i] = Sig(As)
            SigAs = SigAs_hist[:, i]
        ASigAs = A(SigAs)

        # Calculate next x
        alpha = np.inner(r.conj(), S[:, i])
        x = x + alpha * SigAs

        # Calculate New Residual
        r = r - alpha * ASigAs

        # Compute Residual Norms
        rIP = np.inner(r.conj(), r)
        rNorm = np.sqrt(rIP)
        if xTrue is not None:
            err_hist[i + 1] = np.inner((x - xTrue).conj(), A(x - xTrue))
            rTrueNorm = linalg.norm(b - A(x))
            if NormA is not None:
                Res = rNorm / xNormANorm
                Res3[i + 1] = rTrueNorm / xNormANorm
            else:
                Res3[i + 1] = rTrueNorm / bNorm
        if NormA is None:
            Res = rNorm / bNorm
        elif xTrue is None:
            Res = rNorm / NormA / linalg.norm(x)
        Res2[i + 1] = np.copy(Res)

        i = i + 1

    #
    # Return the results
    #

    info = {"res": Res2[: i + 1]}

    if xTrue is not None:
        info["actual_res"] = Res3[: i + 1]
        info["err"] = err_hist[: i + 1]

    return x, SigAs_hist[:, :i], info


def bayescg_post_cov(Sig, SigAs, SqrtSig=False):
    """Computes BayesCG posterior covariance from output of bayescg

    This program computes the posterior covariance with a rank-m downdate
    if given the prior covariance matrix and the vectors SigmaAs_i, 1<=i<=m

    Parameters
    ----------
    Sig : function
        Function that computes matvec of prior covariance
    SigAs : (m by n) numpy array
        Matrix of vectors SigmaAs_i, 1<=i<=m, to compute posterior as rank-m
        downdate of prior. This matrix is returned by program bayescg.
    SqrtSig : boolean, optional, default is False
        Whether BayesCG was computed with SqrtSigTranspose being not None

    Returns
    -------
    Posterior covariance matrix as a numpy array
    """
    N = SigAs.shape[0]
    if SqrtSig:
        return Sig(np.eye(N) - SigAs @ SigAs.conj().T)
    else:
        return Sig(np.eye(N)) - SigAs @ (SigAs.conj().T @ np.eye(N))
