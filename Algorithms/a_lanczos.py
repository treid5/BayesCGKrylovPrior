"""
Modified Lanczos that creates A-orthonormal bases and BayesCG based on it.
"""

import numpy as np


def a_lanczos(A, v0, max_it=None, tol=1e-10, reorth=True):
    """Creates A-orthonormal basis for Krylov space K(A,v0)

    This is a modification of the Lanczos method that creates an
    A-orthonormal basis instead of an orthonormal basis.

    Parameters
    ----------
    A : function
        Function that computes matvec of A
    v0: numpy array
        Initial vector for Krylov space
    max_it : int, optional, default is size of A
        Maximum dimension of Krylov space
    tol : float, optional, default is 1e-10
        Convergence tolerance for Lanczos method
    reorth : bool, optional, default is True
        Whether to reorthogonalize

    Returns
    -------
    V : numpy array
        Matrix whose columns are A-orthonormal basis of K(A,v0)
    """
    N = len(v0)

    if max_it is None:
        max_it = N

    V = np.zeros((N, max_it + 1))

    Av = A(v0)
    beta = np.sqrt(np.inner(v0, Av))
    V[:, 1] = v0 / beta  # This is index 1 intentionally, 0 not returned
    Av = Av / beta

    i = 1

    while i < max_it and beta > tol:
        # Lanczos Recursions
        w = Av - beta * V[:, i - 1]
        alpha = np.inner(w, Av)
        w = w - alpha * V[:, i]

        # Reorthogonalization
        if reorth:
            w = w - V[:, 1 : i + 1] @ (V[:, 1 : i + 1].T @ (A(w)))
            w = w - V[:, 1 : i + 1] @ (V[:, 1 : i + 1].T @ (A(w)))

        # Norm of New Vector
        Av = A(w)
        beta = np.sqrt(np.inner(w, Av))

        # Store New Vector
        if beta > tol:
            i = i + 1
            Av = Av / beta
            V[:, i] = w / beta

    return V[:, 1 : i + 1]
