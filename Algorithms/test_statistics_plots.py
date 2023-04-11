from .bayescg import *
from .bayescg_k import *
from .utilities import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def convergence_plots(
    AVec, InvAVec, xTrue, x0, it, reorth, NormA, tol=1e-15, legend=True, prior=0
):
    b = AVec(xTrue)
    xTrueNorm = np.inner((xTrue - x0).conj(), AVec(xTrue - x0))

    if prior in [0, 1]:
        _, _, infoInv = bayescg(
            AVec, b, x0, InvAVec, it, tol, None, reorth, NormA, xTrue
        )
    if prior in [0, 2]:
        _, _, _, infoK = bayescg_k(
            AVec, b, x0, 1, it, tol, 1e-128, False, reorth, NormA, xTrue
        )
    if prior in [0, 3]:
        _, _, infoRand = bayescg_random_search(
            AVec, b, x0, InvAVec, it, tol, reorth, NormA, xTrue
        )

    plt.figure(1)
    if prior in [0, 1]:
        plt.semilogy((infoInv["err"] / xTrueNorm) ** (1 / 2), "r-", label="Inverse")
    if prior in [0, 2]:
        plt.semilogy((infoK["err"] / xTrueNorm) ** (1 / 2), "k--", label="Krylov")
    if prior in [0, 3]:
        plt.semilogy((infoRand["err"] / xTrueNorm) ** (1 / 2), "b:", label="Random")
    plt.ylabel("Relative A-norm Error")
    plt.xlabel("Iteration")
    plt.tight_layout()
    if legend:
        plt.legend()

    plt.figure(2)
    if prior in [0, 1]:
        plt.plot((infoInv["err"] / xTrueNorm) ** (1 / 2), "r-", label="Inverse")
    if prior in [0, 2]:
        plt.plot((infoK["err"] / xTrueNorm) ** (1 / 2), "k--", label="Krylov")
    if prior in [0, 3]:
        plt.plot((infoRand["err"] / xTrueNorm) ** (1 / 2), "b:", label="Random")
    plt.ylabel("Relative A-norm Error")
    plt.xlabel("Iteration")
    plt.tight_layout()
    if legend:
        plt.legend()


def test_statistics_plots(
    AVec,
    x0,
    sqrt_prior,
    it,
    samples,
    reorth,
    post_rank=None,
    random_search=False,
    hist_z=False,
    tol=1e-15,
    legend=True,
    advanced_mean=False,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    if post_rank is None:
        krylov = False
    else:
        krylov = True

    xTrue = mv_normal(x0, sqrt_prior)
    b = AVec(xTrue)
    N = len(b)

    SqrtSigVec = lambda w: sqrt_prior @ w
    SqrtSigVecTrans = lambda w: sqrt_prior.conj().T @ w

    if type(it) is list:
        N_tests = len(it)
    else:
        N_tests = 1
        it = [it]

    fig_num = 1

    for i in it:
        z_results_temp, s_results_temp, fig_num = zs_statistic(
            AVec,
            x0,
            i,
            samples,
            sqrt_prior,
            3,
            tol,
            None,
            reorth,
            random_search,
            post_rank,
            advanced_mean,
            hist_z,
            None,
            legend,
            fig_num,
        )

        s_results_temp = np.array(s_results_temp).reshape(1, -1)
        plt.figure(fig_num - 1)
        plt.title("Iteration m = " + str(i))

        z_results_temp = np.array(z_results_temp).reshape(1, -1)
        plt.figure(fig_num - 2)
        plt.title("Iteration m = " + str(i))

        if i == it[0]:
            s_results = s_results_temp
            z_results = z_results_temp
        else:
            s_results = np.concatenate((s_results, s_results_temp), 0)
            z_results = np.concatenate((z_results, z_results_temp), 0)

    it = np.array(it).reshape(-1, 1)

    s_results = np.concatenate((it, s_results), axis=1)
    z_results = np.concatenate((it, z_results), axis=1)

    return s_results, z_results


def zs_statistic(
    AVec,
    x0,
    it,
    samples,
    sqrt_prior,
    zs_choice=3,
    tol=1e-6,
    scale=None,
    reorth=True,
    random_search=False,
    post_rank=None,
    advanced_mean=False,
    histogram=False,
    seed=None,
    legend=True,
    fig_num=0,
):
    """
    zs_statistic

    This function implements the z_statistic and s_statistic

    Inputs
    ------
    AVec -- Function that computes matrix A times a vector
    x0 -- Vector that serves as the mean of the prior distribution
    it -- Number of iterations to run BayesCG
    samples -- Number of Z or S statistic samples to generate
    sqrt_prior -- Factorization of covariance for distribution that solutions
        are sampled from. Usually the square root of the prior covariance
    zs_choice -- Integer to choose which test statistic to run
        1 =  Z statistic, 2 = S statistic, 3 = Both (Default is 3)
    tol -- BayesCG convergence tolerance (Default is 1e-6)
    scale -- Choice whether to scale BayesCG posterior (Default is False)
    reorth -- Choice whether to reorthgonalize BayesCG (Default is True)
    random_search -- Choice whether to run Calibrated BayesCG (Default is False)
    post_rank -- None or Integer. If not None, BayesCG under the Krylov prior
        is run. Integer specifies maximum Krylov posterior rank
        (Default is None)
    histogram -- Choice whether to plot test statistics as histogram. If false
        then kernel density estimate is plotted. (Default is False)
    seed -- None or Integer. If not None, this specifies the random seed.
        (Default is None)
    legend -- Choice to make plots with legend. (Default is True)
    fig_num -- First integer that figures are numbered with. Default is 0,
        you probably do not need or want to change this input.
    """

    if seed is not None:
        np.random.seed(seed)

    N = len(x0)

    if post_rank is None:
        krylov = False
    else:
        krylov = True
        if post_rank == N and reorth:
            full_rank = True
        else:
            full_rank = False
            post_rank = max(min(post_rank, N - it), 1)

    if not krylov:
        prior_norm = linalg.norm(sqrt_prior @ sqrt_prior.conj().T)

        def SqrtSigVec(w):
            return sqrt_prior @ w

        def SqrtSigVecT(w):
            return sqrt_prior.conj().T @ w

    if zs_choice == 1 or zs_choice == 3:
        Z_bool = True
    if zs_choice == 2 or zs_choice == 3:
        S_bool = True

    Z = np.zeros(samples, dtype=np.cdouble)
    Error = np.zeros(samples, dtype=np.cdouble)
    Trace = np.zeros(samples, dtype=np.cdouble)
    rank = np.zeros(samples, int)

    mach_eps = np.finfo(float).eps

    for i in range(samples):
        xTrue = mv_normal(x0, sqrt_prior)
        b = AVec(xTrue)

        if krylov:
            if full_rank:
                r = b - AVec(x0)
                V = a_lanczos(AVec, r, None, 1e-12, True)
                Phi = (V.conj().T @ r) ** 2
                prior_norm = linalg.norm((V * Phi) @ V.conj().T, 2)

                xm = x0 + V[:, :it] @ (V[:, :it].conj().T @ r)
                V = V[:, it:]
                Phi = Phi[it:]
            else:
                xm, V, Phi, _ = bayescg_k(
                    AVec,
                    b,
                    x0,
                    post_rank=post_rank,
                    max_it=it,
                    tol=tol,
                    l_tol=1e-24,
                    advanced_mean=advanced_mean,
                    reorth=reorth,
                )

            sqrt_post = V * (Phi ** (1 / 2))
            post = sqrt_post @ sqrt_post.conj().T
            if full_rank:
                post_norm = linalg.norm(post, 2)

        else:
            if random_search:
                xm, SASm, info = bayescg_random_search(
                    AVec, b, x0, SqrtSigVec, it, tol, True, None, None, SqrtSigVecT
                )
            else:
                xm, SASm, info = bayescg(
                    AVec,
                    b,
                    x0,
                    SqrtSigVec,
                    it,
                    tol,
                    scale,
                    reorth,
                    None,
                    None,
                    SqrtSigVecT,
                )

            sqrt_post = bayescg_post_cov(SqrtSigVec, SASm, SqrtSig=True)

            if (not random_search) and (scale is not None):
                sqrt_post = np.sqrt(1 / info["scale"]) * sqrt_post

            post_norm = linalg.norm(sqrt_post @ sqrt_post.conj().T)

        x_diff = xTrue - xm

        if Z_bool:
            if (krylov and full_rank) or not krylov:
                cond = N * mach_eps
                cond = cond * prior_norm / post_norm
            else:
                cond = None
            try:
                x_diff_weighted = linalg.lstsq(sqrt_post, (xTrue - xm), cond)
            except:
                sqrt_post_inv = linalg.pinv(sqrt_post, rtol=cond, return_rank=True)
                x_diff_weighted = (
                    sqrt_post_inv[0] @ (xTrue - xm),
                    None,
                    sqrt_post_inv[1],
                )
            rank[i] = x_diff_weighted[2]
            x_diff_weighted = x_diff_weighted[0]
            Z[i] = np.inner(x_diff_weighted.conj(), x_diff_weighted)

        if S_bool:
            if not krylov:
                post = sqrt_post @ sqrt_post.conj().T
            Error[i] = np.inner(x_diff.conj(), AVec(x_diff))
            Trace[i] = np.trace(AVec(post))

    if Z_bool:
        ZImaginary = np.imag(Z)
        Z = np.real(Z)
        RelZImag = np.max(np.abs(ZImaginary)) / np.max(np.abs(Z))
        # print('Relative max imaginary part of Z stat',RelZImag)

        ZMed = np.median(Z)
        rank = np.median(rank)

        xPlotMax = max(int(2 * ZMed), int(2 * (rank)))
        xPlot = np.linspace(0, xPlotMax, 1000)

        plt.figure(fig_num)
        fig_num += 1
        if histogram:
            zBins = np.linspace(0, int(2 * ZMed), 100)
            hist_data, _, _ = plt.hist(Z, bins=zBins, density=True, color="b")

        plt.plot(
            xPlot, stats.chi2.pdf(xPlot, rank), "r-", label="$\chi^2$ Distribution"
        )

        if histogram:
            plt.ylim((0, 1.5 * max(hist_data)))

        else:
            kde = stats.gaussian_kde(Z)
            plt.plot(xPlot, kde(xPlot), "b--", label="Z Statistic Distribution")
            if rank > 1:
                plt.ylim((0, 1.5 * max(stats.chi2.pdf(xPlot, rank))))

        plt.ylabel("Probability")
        plt.xlabel("Z Statistic Value")
        plt.title("Z Statistic Distribution")
        plt.tight_layout()
        if legend:
            plt.legend()

        chi2_cdf = lambda x: stats.chi2.cdf(x, rank)
        k_s_test = max(
            max(np.abs(ecdf(Z, Z) - chi2_cdf(Z))),
            max(np.abs(ecdf_shift(Z, Z) - chi2_cdf(Z))),
        )

        Z_table = [ZMed, rank, k_s_test]
    else:
        Z_table = None

    if S_bool:
        ErrImaginary = np.imag(Error)
        TrImaginary = np.imag(Trace)
        Error = np.real(Error)
        Trace = np.real(Trace)
        RelErrImag = np.max(np.abs(ErrImaginary)) / np.max(np.abs(Error))
        RelTrImag = np.max(np.abs(TrImaginary)) / np.max(np.abs(Trace))
        # print('Relative max imaginary part of S stat',np.max(np.abs(RelErrImag)))
        # print('Relative max imaginary part of Trace',np.max(np.abs(RelTrImag)))

        ErrMed = np.mean(Error)
        TrMed = np.mean(Trace)
        ErrSD = np.std(Error)
        TrSD = np.std(Trace)

        xPlotMax = max(1.2 * (1.96 * ErrSD + ErrMed), 1.2 * (1.96 * TrSD + TrMed))
        xPlot = np.linspace(0, xPlotMax, int(max(1000, 5 * xPlotMax)))

        Err_kde = stats.gaussian_kde(Error)
        Tr_kde = stats.gaussian_kde(Trace)

        plt.figure(fig_num)
        fig_num += 1
        if TrMed / 10000 > TrSD:
            plt.axvline(x=TrMed, color="r", label="Trace Distribution")
        else:
            plt.plot(xPlot, Tr_kde(xPlot), "r", label="Trace Distribution")
        plt.plot(xPlot, Err_kde(xPlot), "b--", label="Error Distribution")
        if not krylov and rank > 1:
            plt.ylim((0, 1.5 * max(stats.chi2.pdf(xPlot, N - it))))
        plt.xlabel("S Statistic Value")
        plt.ylabel("Probability")
        plt.title("S Statistic Distribution")
        plt.tight_layout()
        if legend:
            plt.legend()

        S_table = [ErrMed, TrMed, TrSD]
    else:
        S_table = None

    return Z_table, S_table, fig_num
