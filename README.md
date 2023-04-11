# BayesCG under the Krylov prior

Numerical experiments for the papers *BayesCG As An Uncertainty Aware Version of CG* and *Statistical Properties of the Probabilistic Numeric Linear Solver BayesCG* by Tim W. Reid, Ilse C. F. Ipsen, Jon Cockayne, and Chris J. Oates.

Both papers share implementations of some algorithms, but each paper has their own Jupyter notebook containing the Python commands that run the numerical experiments and generate the plots.

## *BayesCG As An Uncertainty Aware Version of CG*

### ArXiv link: [https://arxiv.org/abs/2008.03225](https://arxiv.org/abs/2008.03225)

### Jupyter notebook: [BayesCG-as-Uncertainty-Aware-CG.ipynb](BayesCG-as-Uncertainty-Aware-CG.ipynb)

## *Statistical Properties of the Probabilistic Numeric Linear Solver BayesCG*

### ArXiv link: [https://arxiv.org/abs/2208.03885](https://arxiv.org/abs/2208.03885)

### Jupyter notebook: [Statistical-Properties-BayesCG.ipynb](Statistical-Properties-BayesCG.ipynb)

## List of files:

### Algorithms:
- a_lanczos.py
  - Implementation of a modified Lanczos method
- bayescg.py
  - Implementation of the Bayesian Conjugate Gradient Method
- bayescg_k.py
  - Implementation of BayesCG under the Krylov Prior
- cgqs.py
  - Implementation of CG with Gauss-Radau and S Statistic error estimates
- kn_plots.py
  - Creates plots that are in the paper *BayesCG As An Uncertainty Aware Version of CG*
- matrix2tabular.py
  - Converts numpy arrays into LaTeX tabular format
- test_statistics_plots.py
  - Implements the Z and S test statistics and creates the plots that are in the paper *Statistical Properties of the Probabilistic Numeric Linear Solver BayesCG* 
- utilities.py
  - Generates random matrices and samples multivariate Gaussian distributions

### Matrices
- bcsstk14.mtx
  - Sparse matrix used in the paper *Statistical Properties of the Probabilistic Numeric Linear Solver BayesCG*
  - Originally from the [Matrix Market](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc2/bcsstk14.html)
- bcsstk18.mtx
  - Sparse matrix used in the paper *BayesCG As An Uncertainty Aware Version of CG*
  - Originally from the [Matrix Market](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc2/bcsstk18.html)
- bcsstk18_ichol.mtx
  - Incomplete Cholesky factor of bcsstk18.mtx
- bcsstk18_prec.mtx
  - bcsstk18.mtx preconditioned with the incomplete Cholesky factor
