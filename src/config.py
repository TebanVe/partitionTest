import numpy as np

class Config:
    """Configuration parameters for the optimization."""
    def __init__(self, params=None):
        # Surface parameters
        self.n_theta = 10  # Number of points in major circle direction
        self.n_phi = 5    # Number of points in minor circle direction
        self.R = 1.0       # Major radius
        self.r = 0.6       # Minor radius
        # Optimization parameters
        self.n_partitions = 3
        self.lambda_penalty = 0.01
        self.max_iter = 15000
        self.tol = 1e-6
        self.starget = None
        self.c = 0.5  # Armijo condition parameter
        self.rho = 0.5  # Step size reduction factor
        self.m = 10  # Number of corrections to store (for LBFGS)
        # Override with params if provided
        if params:
            for k, v in params.items():
                setattr(self, k, v) 