import numpy as np

# Surface parameters
TORUS_PARAMS = {
    'n_theta': 50,  # Number of points in major circle direction
    'n_phi': 20,    # Number of points in minor circle direction
    'R': 1.0,       # Major radius
    'r': 0.6        # Minor radius
}

# Problem parameters
PROBLEM_PARAMS = {
    'epsilon': 0.1,           # Interface width parameter
    'lambda_penalty': 100.0,  # Penalty weight for constant functions (increased from 1.0)
    'max_iter': 1000,        # Maximum number of iterations for LBFGS
    'tol': 1e-6,             # Convergence tolerance
    'm': 10                  # Number of corrections to store in LBFGS (increased from 5)
}

# Optimization parameters
OPTIMIZATION_PARAMS = {
    'n_partitions': 3,       # Number of partitions
    'starget': None         # Target standard deviation (if None, computed from area)
} 