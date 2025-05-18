import numpy as np

# Surface parameters
TORUS_PARAMS = {
    'n_theta': 20,  # Number of points in major circle direction
    'n_phi': 10,    # Number of points in minor circle direction
    'R': 1.0,       # Major radius
    'r': 0.6        # Minor radius
}

# Problem parameters
PROBLEM_PARAMS = {
    'lambda_penalty': 0.01,  # Penalty weight for constant functions
    'max_iter': 15000,        # Maximum number of iterations for LBFGS
    'tol': 1e-6,             # Convergence tolerance
    'm': 10                  # Number of corrections to store in LBFGS
}

# Optimization parameters
OPTIMIZATION_PARAMS = {
    'n_partitions': 3,       # Number of partitions
    'starget': None         # Target standard deviation (if None, computed from area)
}

class Config:
    """Configuration parameters for the optimization."""
    
    def __init__(self):
        # Number of partitions
        self.n_partitions = 3
        
        # Epsilon will be computed automatically based on mesh size
        # in the SLSQPOptimizerAnalytic class
        
        # Penalty weight for constant functions
        self.lambda_penalty = 0.01
        
        # Optimization parameters
        self.max_iter = 15000
        self.tol = 1e-6
        
        # Line search parameters
        self.c = 0.5  # Armijo condition parameter
        self.rho = 0.5  # Step size reduction factor
        
        # L-BFGS parameters
        self.m = 10  # Number of corrections to store
        
        # Target standard deviation
        self.starget = None 