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
            print("\nOverriding default parameters with:")
            for k, v in params.items():
                if hasattr(self, k):
                    old_value = getattr(self, k)
                    setattr(self, k, v)
                    print(f"  {k}: {old_value} -> {v}")
                else:
                    print(f"  Warning: Unknown parameter '{k}' with value {v}") 
            print("\n")