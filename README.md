## Index of Supplementary Materials

### Title of paper: *BayesCG As An Uncertainty Aware Version of CG*

### Authors: *Tim W. Reid, Ilse C. F. Ipsen, Jon Cockayne, and Chris J. Oates*

### arXiv Link: [https://arxiv.org/abs/2008.03225](https://arxiv.org/abs/2008.03225)

#### BayesCGExperiments
- **Type:** Directory of Python codes
- **Contents:** Python codes, Jupyter notebooks, and matrices used to generate
  plots in the paper and supplementary document. The Jupyter notebooks (ipynb files) document how the plots were created. The specific files included are:
    - Paper-Experiments.ipynb
        - Documents how the plots in the paper were created
    - Supplemental-Experiments.ipynb
        - Documents how the plots in the supplementary document were created
    - plots.py
        - Creates plots that are in the paper
    - utilities.py
        - Generates random matrices and samples multivariate Gaussian distributions
    - bayescg.py
        - Implementation of the Bayesian Conjugate Gradient Method
    - bayescg_k.py
        - Implementation of BayesCG under the Krylov Prior
    - cgqs.py
        - Implementation of CG with Gauss-Radau and S Statistic error estimates
    - bcsstk18.mtx
        - Sparse matrix we perform numerical experiments with
        - Originally from the [Matrix Market](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc2/bcsstk18.html)
    - bcsstk18_ichol.mtx
        - Incomplete Cholesky factor of bcsstk18.mtx
    - bcsstk18_prec.mtx
        - bcsstk18.mtx preconditioned with the incomplete Cholesky factor
