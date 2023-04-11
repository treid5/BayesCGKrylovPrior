"""
These functions convert Python floats or 2D lists into LaTeX
"""

import numpy as np


def float2sci(x, figs=2, dollars=True):
    """Converts floating point numbers to scientific notation LaTeX strings

    Parameters
    ----------
    x : float or None
        Number you want converted
    figs : int, optional, default is 2
        number of significant digits to present in number
    dollars : bool, optional, default is True
        Whether to return string with $ for LaTeX

    Returns
    -------
    x : str
        Input x expressed as scientific notation LaTeX
        If x is None, a string of spaces is returned
        You should then run print(x) so that the result is easy to copy/paste
    """

    if x == 0:
        if dollars:
            return r"$ 0 $"
        else:
            return r" 0 "

    if x is None:
        if dollars:
            return r"$  $"
        else:
            return r"   "

    power = int(np.floor(np.log10(np.abs(x))))

    if not -1 <= power <= 2:
        x = np.round(x / (10**power), figs)
        x = str(x) + r" \times 10^{" + str(power) + "}"
    else:
        figs = max(figs - power, 0)
        x = str(np.round(x, figs))

    x = r"$ " + x + r" $"

    if not dollars:
        x = x.replace(r"$", r" ")

    return x


def matrix2tabular(C, figs=2, dollars=True):
    """
    Converts a 2D list or numpy array into scientific notation LaTeX tabular

    Parameters
    ----------
    C : 2 dimensional list or numpy array
        Matrix of data
    figs : int, optional, default is 2
        number of significant digits to present in number
    dollars : bool, optional, default is True
        Whether to return string with $ for LaTeX
        True is useful for tables
        False is useful for matrices

    Returns
    -------
    COut : str
        Input COut expressed as scientific notation LaTeX tabular
        You should run print(COut) so that the result is easy to copy/paste
    """
    M = len(C)
    N = len(C[0])

    COut = ""

    for m in range(M):
        for n in range(N):
            COut = COut + float2sci(C[m][n], figs, dollars)
            if n == N - 1:
                COut = COut + r" \\" + "\n"
            else:
                COut = COut + " & "
    return COut
