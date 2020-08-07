''' This module contains the program that runs the numerical experiments.'''

from bayescg_k import bayescg_k
from cgqs import cgqs
import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt

def bayescg_exp_plots(AVec, b, x0, NormA, xTrue, GR, it, it_z=None, L=5,\
                      samples=10, pct=95, MA=20, reorth=False,
                      cgq=True, plt_title=True):
    """ Computes plots of numerical experiments of BayesCG under Krylov prior

    This program must be used in a Jupyter notebook

    Parameters
    ----------
    A : function
        Function that computes matvec of matrix A
    b : numpy array
        Vector b
    x0 : numpy array
        Initial guess for x
    NormA : float or None
        2-norm of A
        If supplied, residual is ||r||/(||A|| ||x_m||)
        If not, residual is ||r||/||b||
    xTrue : numpy array,
        True solution of linear system
    GR : float or None
        Lower bound for smallest eigenvalue of A
        If mu is supplied the Gauss-Radau error bound [1] is computed
        If mu is None, the Gauss-Radau approximation [2] is computed
    it : int
        Number of iterations to perform
    it_z : int or None, optional, default is None
        Number of iterations to zoom in on
    L : int, optional, default is 5
        Delay for computing error estimates
    samples : int or None, optional, default is 10
        Number of S-statistic samples to compute each iteration
    pct : float, 0<pct<100, optional, default is 95
        Percent credible interval to plot
    MA : int, optional, default is 20
        Number of iteration moving average to plot
    reorth : bool, optional, default is False
        Whether to reorthogonalize
    cgq : bool, optional, default is True
        Whether to run CGQ algorithm
    plt_title : bool, optional, default is True
        Whether to have the plot titles display

    Returns
    -------
    Plots with the following figure numbers
    1 -- Emperical samples from S statistic (Only if samples is not None)
    2 -- Emperical S statistic credible interval (If samples is not None)
    3 -- S statistic credible interval and Gauss-Radau estimate
    4 -- Relative accuracy of error estimates
    5 -- Relative accuracy of error estimates, MA iteration moving average
    6 -- BayesCG vs CG error (Only if Samples is not None)
    7 -- BayesCG vs CG residual (Only if Samples is not None)
    8 -- First it_z iterations of Figure 1
    9 -- First it_z iterations of Figure 2
    10 -- First it_z iterations of Figure 3
    11 -- First it_z iterations of Figure 4
    12 -- First it_z iterations of Figure 5
    """

    pct_mult = erfinv(pct/100)*np.sqrt(2)

    if (not cgq) or (samples is not None):
        x,_,_,info = bayescg_k(AVec,b,x0,L,it,None,1e-128,samples,reorth,\
                               NormA,xTrue)

    if cgq:
        _,info2 = cgqs(AVec, b, x0, GR, None, it, L, reorth, NormA, xTrue)
        Emu = info2['GaussRadau']
        SExp = info2['sExp']
        SVar = info2['sSD']
        err2 = info2['err']
        res2 = info2['res']
        comp_color = 'b'
    else:
        Emu = None
        SExp = info['sExp']
        SVar = info['sSD']
        err2 = info['err']
        res2 = info['res']
        comp_color = 'r'

    
    if samples is not None:

        scatter_color = 'r'

        S = info['samples']
        err = info['err']
        res = info['res']

        SSD = np.std(S,0,ddof=1)
        SAvg = np.mean(S,0)
        SMax = SAvg+pct_mult*SSD


        plt.figure(1)
        plt.semilogy(S[0], '.', markersize=1, label='S Stat Sample',
                     color=scatter_color)
        for i in range(1,S.shape[0]):
            plt.semilogy(S[i], '.', markersize=1, color=scatter_color)
    
        plt.semilogy(err,'k',label = 'Error')
        plt.legend()
        Axes = plt.axis()
        plt.xlabel('Iteration')
        plt.ylabel('Squared A-norm Error')
        if plt_title:
            plt.title('BayesCG S Statistic Samples')
        plt.tight_layout()

        plt.figure(2)
        plt.fill_between(range(len(err2)),SAvg,SMax,color=scatter_color,\
                         alpha=.2,label='S Stat '+str(pct)+'% Cred Int')
        plt.semilogy(err,'k', label = 'Error')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Squared A-norm Error')
        if plt_title:
            plt.title('BayesCG S Statistic Mean and Max')
        plt.tight_layout()
        
    plt.figure(3)
    plt.fill_between(range(len(err2)),SExp,SExp+pct_mult*SVar,\
                     color = 'b',alpha = .2,label = 'Exp '+str(pct)\
                     +'% Cred Int')
    if Emu is not None:
        plt.semilogy(Emu,':k',label = 'Gauss-Radau')
    plt.semilogy(err2,'k',label = 'Error')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Squared A-norm Error')
    if plt_title:
        plt.title('CG Error Estimate')
    plt.tight_layout()

    if samples is not None:
        AxesCG = plt.axis()
        Axes = (min(Axes[0],AxesCG[0]),
                max(Axes[1],AxesCG[1]),
                min(Axes[2],AxesCG[2]),
                max(Axes[3],AxesCG[3]))
        plt.figure(1)
        plt.axis(Axes)
        plt.figure(2)
        plt.axis(Axes)
        plt.figure(3)
        plt.axis(Axes)

    plt.figure(4)
    plt.semilogy(np.abs(err2-SExp)/np.min([err2,SExp],0),\
                 '-b',label = 'S Stat Mean')
    plt.semilogy(np.abs(err2-(SExp+pct_mult*SVar))/\
                 np.min([err2,SExp+pct_mult*SVar],0),'--r',\
                 label = 'S('+str(pct)+')')
    if Emu is not None:
        plt.semilogy(np.abs(err2-Emu)/np.min([err2,Emu],0),\
                     ':k',label = 'Gauss-Radau')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Relative Accuracy')
    if plt_title:
        plt.title('Relative Distance Between CG Error and Estimates')
    plt.tight_layout()

    MA1 = np.zeros(len(err2)-MA)
    MA2 = np.copy(MA1)
    MA3 = np.copy(MA1)

    for i in range(len(err2)-MA):
        MA1[i] = np.mean(np.abs(err2[i:i+MA]\
                                      -(SExp[i:i+MA]+pct_mult\
                                        *SVar[i:i+MA]))\
                            /np.min([err2[i:i+MA],SExp[i:i+MA]+pct_mult\
                                        *SVar[i:i+MA]],0))
        MA2[i] = np.mean(np.abs(err2[i:i+MA]\
                                      -SExp[i:i+MA])\
                            /np.min([err2[i:i+MA],SExp[i:i+MA]],0))
        if Emu is not None:
            MA3[i] = np.mean(np.abs(err2[i:i+MA]\
                                    -Emu[i:i+MA])\
                             /np.min([err2[i:i+MA],Emu[i:i+MA]],0))

    
    plt.figure(5)
    plt.semilogy(MA2,'-b',label = 'S Stat Mean')
    plt.semilogy(MA1,'--r',label = 'S('+str(pct)+')')
    plt.semilogy(MA3,':k',label = 'Gauss-Radau')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Relative Accuracy')
    if plt_title:
        plt.title('Distance Between Error and Estimates, '\
                  +str(MA)+' Iteration Avg')
    plt.tight_layout()

    if samples is not None:
        plt.figure(6)
        plt.semilogy(err2, '-', label='CG Error', color=comp_color)
        plt.semilogy(err, '--', label='BayesCG Error', color=scatter_color)
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Squared A-norm Error')
        if plt_title:
            plt.title('CG and BayesCG Error')
        plt.tight_layout()

        plt.figure(7)
        plt.semilogy(res2, '-', label='CG Residual', color=comp_color)
        plt.semilogy(res, '--', label='BayesCG Residual', color=scatter_color)
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Relative Residual')
        if plt_title:
            plt.title('CG and BayesCG Residual')
        plt.tight_layout()

    if it_z is not None:

        if samples is not None:
            plt.figure(8)
            plt.semilogy(S[0,:it_z],'.r',markersize = 1,\
                         label = 'S Stat Sample')
            for i in range(1,S.shape[0]):
                plt.semilogy(S[i,:it_z],'.r',markersize = 1)
    
            plt.semilogy(err[:it_z],'k',label = 'Error')
            plt.legend()
            Axes2 = plt.axis()
            plt.xlabel('Iteration')
            plt.ylabel('Squared A-norm Error')
            if plt_title:
                plt.title('BayesCG S Statistic Samples')
            plt.tight_layout()

            plt.figure(9)
            plt.fill_between(range(len(err2[:it_z])),SAvg[:it_z],\
                             SMax[:it_z],color = 'r',alpha = .2,\
                             label = 'S Stat '+str(pct)+'% Cred Int')
            plt.semilogy(err[:it_z],'k', label = 'Error')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('Squared A-norm Error')
            if plt_title:
                plt.title('BayesCG S Statistic Mean and Max')
            plt.tight_layout()

        plt.figure(10)
        plt.fill_between(range(len(err2[:it_z])),SExp[:it_z],\
                         SExp[:it_z]+pct_mult*SVar[:it_z],color = 'b',\
                         alpha = .2,label = 'Exp '+str(pct)+'% Cred Int')
        if Emu is not None:
            plt.semilogy(Emu[:it_z],':k',label = 'Gauss-Radau')
        plt.semilogy(err2[:it_z],'k',label = 'Error')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Squared A-norm Error')
        if plt_title:
            plt.title('CG Error Estimate')
        plt.tight_layout()

        if samples is not None:
            AxesCG2 = plt.axis()
            Axes2 = (min(Axes2[0],AxesCG2[0]),
                     max(Axes2[1],AxesCG2[1]),
                     min(Axes2[2],AxesCG2[2]),
                     max(Axes2[3],AxesCG2[3]))
            plt.figure(8)
            plt.axis(Axes2)
            plt.figure(9)
            plt.axis(Axes2)
            plt.figure(10)
            plt.axis(Axes2)

        plt.figure(11)
        plt.semilogy(np.abs(err2[:it_z]-SExp[:it_z])\
                     /np.min([err2[:it_z],SExp[:it_z]],0)\
                     ,'-b',label = 'S Stat Mean')
        plt.semilogy(np.abs(err2[:it_z]-(SExp[:it_z]+pct_mult\
                                            *SVar[:it_z]))\
                     /np.min([err2[:it_z],SExp[:it_z]+pct_mult\
                                            *SVar[:it_z]],0)\
                     ,'--r',label = 'S('+str(pct)+')')
        if Emu is not None:
            plt.semilogy(np.abs(err2[:it_z]-Emu[:it_z])\
                         /np.min([err2[:it_z],Emu[:it_z]],0)\
                         ,':k',label = 'Gauss-Radau')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Relative Accuracy')
        if plt_title:
            plt.title('Relative Distance Between CG Error and Estimates')
        plt.tight_layout()

        plt.figure(12)
        plt.semilogy(MA2[:it_z],'-b',label = 'S Stat Mean')
        plt.semilogy(MA1[:it_z],'--r',label = 'S('+str(pct)+')')
        plt.semilogy(MA3[:it_z],':k',label = 'Gauss-Radau')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Relative Accuracy')
        if plt_title:
            plt.title('Distance Between Error and Estimates, '\
                      +str(MA)+' Iteration Avg')
        plt.tight_layout()

