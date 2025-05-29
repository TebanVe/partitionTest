import numpy as np

class Config:
    """Configuration parameters for the optimization."""
    def __init__(self, params=None):
        # Surface parameters
        self.n_theta = 12  # Number of points in major circle direction
        self.n_phi = 8    # Number of points in minor circle direction
        self.R = 1.0       # Major radius
        self.r = 0.3       # Minor radius
        # Optimization parameters
        self.n_partitions = 3
        self.lambda_penalty = 1.0
        self.max_iter = 1000
        self.tol = 1e-6
        self.starget = 1.0
        self.c = 0.5  # Armijo condition parameter
        self.rho = 0.5  # Step size reduction factor
        self.m = 10  # Number of corrections to store (for LBFGS)
        self.seed = 42  # Default seed for random initialization
        # Refinement parameters
        self.refinement_levels = 1  # Number of mesh refinement levels (1 means no refinement)
        self.n_theta_increment = 2  # Number of n_theta to add per refinement
        self.n_phi_increment = 1    # Number of n_phi to add per refinement
        self.use_analytic = True  # Whether to use analytic gradients
        
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