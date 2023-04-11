"""
This module has our implementation of CG with error estimates
"""

import numpy as np
from scipy import linalg


def cgqs(
    A,
    b,
    x,
    mu=None,
    tol=1e-6,
    max_it=None,
    delay=5,
    reorth=False,
    NormA=None,
    xTrue=None,
):
    """Iteratively computes solution to Ax = b with error estimates.

    This program iteratively solves a symmetric positive definite system
    of linear equations with a with the Conjugate Gradient method.
    Additionally, it computes the S statistic mean and standard deviation
    and a Gauss-Radau error estimate. If a lower bound to the
    smallest eigenvalue of A is provided, the Gauss-Radau error bound is
    computed with the CGQ algorithm [1]. If a bound is not provided, then
    the approximation to the Gauss-Radau bound is computed [2].

    Parameters
    ----------
    A : function
        Function that computes matvec of matrix A
    b : numpy array
        Vector b
    x : numpy array
        Initial guess for x
    mu : float or None, optional, Default is None
        Lower bound for smallest eigenvalue of A
        If mu is supplied the Gauss-Radau error bound [1] is computed
        If mu is None, the Gauss-Radau approximation [2] is computed
    tol : float or None, optional, Default is 1e-6
        Convergence tolerance.
        If None then program iterates until maximum iterations
    max_it : int, optional, default is size of A
        Maximum iteration count
    delay : int, optional, default is 5
        Delay for computing error estimates
    reorth : bool, optional, default is False
        Whether to reorthogonalize
    NormA : float or None, optional, default is None
        2-norm of A
        If supplied, residual is ||r||/(||A|| ||x_m||)
        If not, residual is ||r||/||b||
    xTrue : numpy array or None, optional, default is None
        True solution of linear system

    Returns
    -------
    x : numpy array
        Approximate solution
    info : dict
        Dictionary containing convergence information
        Dictionary keys always returned:
            'res' : Residual history
            'sExp' : Expected value of S statistic at each iteration
            'sSD' : Standard deviation of S statistic at each iteration
            'GRApprox' : Gauss Radau approximation from [2]
        Dictionary keys returned if mu is supplied:
            'GaussRadau' : Gauss Radau bound computed with CGQ [1]
        Additional keys if xTrue is supplied
            'err' : Error history
            'actual_res' : Actual residual, b-Ax, history
                (as opposed to recursively computed residual)

    References
    ----------
    [1] Meurant, G and Tichy, P. "On computing quadrature-based bounds
    for the A-norm of the error in conjugate gradients"
    DOI = 10.1007/s11075-012-9591-9

    [2] Meurant, G and Tichy, P. "Approximating the extreme Ritz values
    and upper bounds for the A-norm of the error in CG"
    DOI = 10.1007/s11075-018-0634-8
    """

    #
    # Here we define the variables
    #

    # Size of the system
    N = len(x)

    # Default Maximum Iterations
    if max_it is None:
        max_it = N
    else:
        max_it = max_it + delay

    # Residual and first search direction
    r = b - A(x)
    s = np.copy(r)

    # Residual Norm
    rIP = np.zeros(max_it + 1)
    rIP[0] = np.inner(r, r)
    rNorm = np.sqrt(rIP[0])

    if reorth:
        r_hist = np.zeros((N, max_it + 1))
        r_hist[:, 0] = r / rNorm

    # Calculate first search direction length
    As = A(s)
    sIP = np.abs(np.inner(s, As))
    gamma = np.zeros(max_it + 1)
    gamma[0] = rIP[0] / sIP

    # Gauss-Radau values
    g = np.zeros(max_it + 1)

    if mu is not None:
        # CGQ
        Emu = np.zeros(max_it + 1)
        Emu[0] = gamma[0] * rIP[0]
        gmu = np.zeros(max_it + 1)
        gmu[0] = rIP[0] / mu

    # G-R approximation by estimating Ritz value
    Emu_approx = np.zeros(max_it + 1)
    rho = gamma[0]
    tau = gamma[0]
    sigma = 0
    t = 0
    c = 0
    phi = np.zeros(max_it + 1)
    phi[0] = 1

    # S Stat Values
    SExp = np.zeros(max_it + 1)
    SSD = np.zeros(max_it + 1)

    # Convergence Values
    res = np.zeros(max_it + 1)
    if (NormA is None) or (xTrue is None):
        bNorm = linalg.norm(b)
        res[0] = rNorm / bNorm
    if xTrue is not None:
        xNorm = linalg.norm(xTrue)
        err_hist = np.zeros(max_it + 1)
        err_hist[0] = np.inner(x - xTrue, A(x - xTrue))
        if NormA is not None:
            xNormANorm = linalg.norm(xTrue) * NormA
            res[0] = rNorm / xNormANorm
        res2 = np.copy(res)

    i = 0

    #
    # Iterating Through Conjugate Gradient
    #

    while i < max_it or i < delay:
        x = x + gamma[i] * s

        # Calculate New Residual
        r = r - gamma[i] * As

        if reorth:
            # Reorthogonalize Residual
            r = r - r_hist[:, : i + 1] @ (r_hist[:, : i + 1].T @ r)
            r = r - r_hist[:, : i + 1] @ (r_hist[:, : i + 1].T @ r)

        # Compute Residual Norms
        rIP[i + 1] = np.inner(r, r)
        rNorm = np.sqrt(rIP[i + 1])
        if xTrue is not None:
            err_hist[i + 1] = np.inner(x - xTrue, A(x - xTrue))
            rTrueNorm = linalg.norm(b - A(x))
            if NormA is not None:
                res[i + 1] = rNorm / xNormANorm
                res2[i + 1] = rTrueNorm / xNormANorm
            else:
                res2[i + 1] = rTrueNorm / bNorm
        if NormA is None:
            res[i + 1] = rNorm / bNorm
        elif xTrue is None:
            res[i + 1] = rNorm / NormA / linalg.norm(x)

        # Store residual vector if reorthogonalizing
        if reorth:
            r_hist[:, i + 1] = r / rNorm

        # Calculate next search direction
        delta = rIP[i + 1] / rIP[i]
        s = r + delta * s

        # Calculate next search direction length
        As = A(s)
        sIP = np.inner(s, As)
        gamma[i + 1] = rIP[i + 1] / sIP

        # Update Gauss-Radau values
        g[i] = gamma[i] * rIP[i]
        if mu is not None:
            # CGQ
            Deltamu = gmu[i] - g[i]
            gmu[i + 1] = rIP[i + 1] * (Deltamu / (mu * Deltamu + rIP[i + 1]))

        # Ritz Value Estimate Variables
        sigma = -1.0 * np.sqrt(gamma[i + 1] * delta / gamma[i]) * (t * sigma + c * tau)
        tau = gamma[i + 1] * (delta * tau / gamma[i] + 1)
        chi = np.sqrt((rho - tau) ** 2 + 4 * sigma**2)
        c2 = 0.5 * (1 - (rho - tau) / chi)
        rho = rho + chi * c2
        t = np.sqrt(1 - c2)
        c = np.sqrt(np.abs(c)) * np.sign(sigma)
        phi[i + 1] = phi[i] / (phi[i] + delta)  # phi is Ritz value

        #  1 based indexing for Error Estimates
        i = i + 1

        # Update Error Bound and S Stat Values
        if i >= delay:
            ila = i - delay
            # S Statistic
            SExp[ila] = np.sum(g[ila:i])
            SSD[ila] = np.sqrt(2 * np.sum(g[ila:i] ** 2))
            if mu is not None:
                # CGQ
                Emu[ila] = SExp[ila] + gmu[i]

            # Gauss-Radau approximation
            Emu_approx[ila] = phi[ila] * rho * rIP[ila]

        # Evaluate convergence condition
        if tol is not None and i >= delay:
            if res[i] < tol:
                break

    #
    # Return the results
    #

    info = {"res": res[: ila + 1]}
    info["sExp"] = SExp[: ila + 1]
    info["sSD"] = SSD[: ila + 1]
    info["GRApprox"] = Emu_approx[: ila + 1]

    if mu is not None:
        info["GaussRadau"] = Emu[: ila + 1]

    if xTrue is not None:
        info["err"] = err_hist[: ila + 1]
        info["actual_res"] = res2[: ila + 1]

    return x, info
