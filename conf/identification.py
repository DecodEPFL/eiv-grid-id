from conf.conf import noise_level

# Regularization parameters
#4e-10 for CIGRE, 1e-9 for ieee33
lambda_lasso = 1e-9
#1e8 for laplacian weighted CIGRE (or 4e5 otherwise), 2e-6 for unweighted CIGRE or ieee33, 2e-7 for unweighted ieee123
lambda_eiv = 0*8e-4#2e-6
lambdaprime = 200 #200 for ieee123, 20 for others

# Tolerance for stopping the iterative algorithm
abs_tol = 1e-16
rel_tol = 1e-13

# What should be regularized with the Bayesian prior
use_tls_diag = False
contrast_each_row = True
regularize_diag = False
