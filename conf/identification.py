from conf.conf import noise_level

# Regularization parameters
lambda_lasso = 4e-10
lambda_eiv = 5e-7
lambdaprime = 200

# Tolerance for stopping the iterative algorithm
abs_tol = 1e-13
rel_tol = 1e-16

# What should be regularized with the Bayesian prior
use_tls_diag = False
contrast_each_row = True
regularize_diag = False
