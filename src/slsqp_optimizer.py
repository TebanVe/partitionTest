import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.optimize import minimize
from .projection_iterative import orthogonal_projection_iterative

class SLSQPOptimizer:
    def __init__(self, 
                 K: np.ndarray,  # Stiffness matrix
                 M: np.ndarray,  # Mass matrix
                 v: np.ndarray,  # Mass matrix column sums (v = 1ᵀM)
                 n_partitions: int,
                 epsilon: float,
                 lambda_penalty: float = 1.0,
                 starget: Optional[float] = None):
        """
        Initialize the SLSQP optimizer for manifold partition optimization.
        
        Args:
            K: Stiffness matrix for gradient term
            M: Mass matrix for area term
            v: Vector of mass matrix column sums (v = 1ᵀM)
            n_partitions: Number of partitions
            epsilon: Interface width parameter
            lambda_penalty: Initial penalty weight for constant functions
            starget: Target standard deviation (if None, computed from area)
        """
        self.K = K
        self.M = M
        self.v = v
        self.n_partitions = n_partitions
        self.epsilon = epsilon
        
        # Compute characteristic scales
        self.total_area = np.sum(v)
        self.avg_edge_length = np.sqrt(self.total_area / len(v))
        
        # More aggressive scaling factors
        self.grad_scale = 1.0  # Base scale
        self.interface_scale = 0.01  # Significantly reduce interface term
        self.penalty_scale = 0.1  # Reduce penalty term influence
        
        # Scale lambda_penalty down
        self.lambda_penalty = lambda_penalty * 0.1
        
        if starget is None:
            normalized_area = 1.0/n_partitions
            self.starget = np.sqrt(normalized_area * (1 - normalized_area))
        else:
            self.starget = starget
            
        # Increase cache tolerance
        self.cache_tolerance = 1e-10
        
        # Cache for function evaluations
        self.cache = {}
        
        # Initialize logging
        self.log = {
            'iterations': [],
            'energies': [],
            'gradient_norms': [],
            'constraint_violations': [],
            'warnings': [],
            'step_sizes': [],
            'optimization_energy_changes': [],
            'term_magnitudes': [],
            'x_history': []  # Add this to track the optimization path
        }
        
    def _get_cache_key(self, x: np.ndarray) -> tuple:
        """Generate a cache key for an input array."""
        return tuple(np.round(x / self.cache_tolerance) * self.cache_tolerance)
    
    def compute_energy(self, x: np.ndarray) -> float:
        """Compute the total energy with caching and improved scaling."""
        # Check cache
        cache_key = self._get_cache_key(x)
        if cache_key in self.cache:
            return self.cache[cache_key]['energy']
        
        N = len(self.v)
        phi = x.reshape(N, self.n_partitions)
        
        # Compute gradient term (scaled by mesh size)
        grad_term = 0
        for i in range(self.n_partitions):
            phi_i = phi[:, i]
            term = self.grad_scale * self.epsilon * float(phi_i.T @ (self.K @ phi_i))
            grad_term += term
            
        # Compute interface term (scaled down)
        interface_term = 0
        for i in range(self.n_partitions):
            phi_i = phi[:, i]
            interface_vec = phi_i**2 * (1 - phi_i)**2
            term = self.interface_scale * (1/self.epsilon) * float(interface_vec.T @ (self.M @ interface_vec))
            interface_term += term
            
        # Compute penalty term (with improved stability)
        penalty_term = 0
        for i in range(self.n_partitions):
            # Use weighted statistics for better numerical stability
            weights = self.v / self.total_area
            mean_i = np.sum(weights * phi[:, i])
            var_i = np.sum(weights * (phi[:, i] - mean_i)**2)
            std_i = np.sqrt(var_i + 1e-10)
            term = self.penalty_scale * self.lambda_penalty * (std_i - self.starget)**2
            penalty_term += term
            
        total_energy = grad_term + interface_term + penalty_term
        
        # Cache result
        self.cache[cache_key] = {
            'energy': total_energy,
            'grad_term': grad_term,
            'interface_term': interface_term,
            'penalty_term': penalty_term
        }
        
        return total_energy
    
    def compute_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the energy."""
        print("\nComputing gradient...")
        N = len(self.v)
        phi = x.reshape(N, self.n_partitions)
        grad = np.zeros_like(x)
        
        for i in range(self.n_partitions):
            # Gradient term
            grad_grad = 2 * self.epsilon * (self.K @ phi[:, i])
            grad_grad_norm = np.linalg.norm(grad_grad)
            print(f"  Gradient term {i} norm: {grad_grad_norm:.6f}")
            
            # Interface term
            interface_vec = phi[:, i]**2 * (1 - phi[:, i])**2
            grad_interface = (2/self.epsilon) * (self.M @ (interface_vec * (1 - 2*phi[:, i])))
            grad_interface_norm = np.linalg.norm(grad_interface)
            print(f"  Interface term {i} norm: {grad_interface_norm:.6f}")
            
            # Penalty term
            mean_i = np.mean(phi[:, i])
            var_i = np.mean((phi[:, i] - mean_i)**2)
            std_i = np.sqrt(var_i + 1e-10)
            grad_penalty = 2 * self.lambda_penalty * (std_i - self.starget) * \
                        (phi[:, i] - mean_i) / (N * std_i)
            grad_penalty_norm = np.linalg.norm(grad_penalty)
            print(f"  Penalty term {i} norm: {grad_penalty_norm:.6f}")
            
            grad[i*N:(i+1)*N] = grad_grad + grad_interface + grad_penalty
        
        grad_norm = np.linalg.norm(grad)
        print(f"Total gradient norm: {grad_norm:.6f}")
        return grad
    
    def check_constraints(self, x: np.ndarray) -> dict:
        """Check constraint violations for a given point."""
        N = len(self.v)
        phi = x.reshape(N, self.n_partitions)
        
        # Check partition constraint
        row_sums = np.sum(phi, axis=1)
        row_sum_violation = np.max(np.abs(row_sums - 1.0))
        
        # Check area constraint
        area_sums = self.v @ phi
        target_area = np.sum(self.v)/self.n_partitions
        area_violation = np.max(np.abs(area_sums - target_area))
        
        # Check non-negativity
        nonneg_violation = -np.min(phi) if np.min(phi) < 0 else 0
        
        return {
            'row_sum': row_sum_violation,
            'area': area_violation,
            'nonneg': nonneg_violation
        }
    
    def constraint_fun(self, x: np.ndarray) -> np.ndarray:
        """Compute constraints with relaxed tolerances."""
        N = len(self.v)
        phi = x.reshape(N, self.n_partitions)
        
        # Compute row sum constraints (reduced number)
        row_sums = np.sum(phi, axis=1) - 1.0
        # Only use a subset of row constraints
        n_row_constraints = min(N-1, 100)  # Limit number of row constraints
        row_indices = np.arange(0, N-1, N//n_row_constraints)
        row_violations = row_sums[row_indices]
        
        # Compute area constraints
        area_sums = self.v @ phi
        target_area = self.total_area / self.n_partitions
        area_constraints = (area_sums - target_area) / target_area  # Scale by target area
        
        # Return all constraints except the last area constraint
        return np.concatenate([row_violations, area_constraints[:-1]])
    
    def constraint_jac(self, x: np.ndarray) -> np.ndarray:
        """Compute constraint Jacobian with reduced constraints."""
        N = len(self.v)
        n = self.n_partitions
        
        # Reduced number of row constraints
        n_row_constraints = min(N-1, 100)
        row_indices = np.arange(0, N-1, N//n_row_constraints)
        
        # Row sum Jacobian (reduced)
        row_sum_jac = np.zeros((len(row_indices), N * n))
        for idx, i in enumerate(row_indices):
            for j in range(n):
                row_sum_jac[idx, i + j*N] = 1.0
        
        # Area Jacobian (excluding last constraint)
        area_jac = np.zeros((n-1, N * n))
        target_area = self.total_area / n
        for i in range(n-1):
            area_jac[i, i*N:(i+1)*N] = self.v / target_area
        
        return np.vstack([row_sum_jac, area_jac])
    
    def optimize(self, x0: np.ndarray, maxiter: int = 100, ftol: float = 1e-8) -> tuple:
        """Optimize with more aggressive parameters."""
        print("\nStarting SLSQP optimization...")
        
        # Initialize logging
        self.log = {
            'iterations': [],
            'energies': [],
            'constraint_violations': [],
            'warnings': [],
            'step_sizes': [],
            'optimization_energy_changes': [],
            'x_history': []
        }
        
        # Clear cache
        self.cache.clear() 
        
        # Project initial point more aggressively
        N = len(self.v)
        x0_reshaped = x0.reshape(N, self.n_partitions)
        x0_reshaped = np.maximum(x0_reshaped, 0)
        
        # More aggressive row normalization
        row_sums = np.sum(x0_reshaped, axis=1, keepdims=True)
        x0_reshaped = x0_reshaped / np.maximum(row_sums, 1e-6)
        
        # More aggressive area scaling
        target_area = self.total_area / self.n_partitions
        current_areas = self.v @ x0_reshaped
        area_scales = target_area / np.maximum(current_areas, target_area/5)
        for i in range(self.n_partitions):
            x0_reshaped[:, i] *= area_scales[i]
        
        x0 = x0_reshaped.flatten()
        
        # More aggressive SLSQP options
        options = {
            'maxiter': maxiter,
            'ftol': 1e-6,  # Looser tolerance
            'disp': True,
            'eps': 1e-3,  # Larger steps
            'finite_diff_rel_step': 1e-4  # Larger finite difference step
        }
        
        # Define constraints with reduced row constraints
        constraints = [
            {'type': 'eq', 'fun': self.constraint_fun, 'jac': self.constraint_jac}
        ]
        
        # Looser bounds
        bounds = [(0, 2.0) for _ in range(len(x0))]
        
        # Run optimization
        result = minimize(
            self.compute_energy,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
            callback=self.callback
        )
        
        print("\nOptimization completed:")
        print(f"Success: {result.success}")
        print(f"Status: {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Final energy: {result.fun:.6e}")
        print(f"Final constraint violation: {np.max(np.abs(self.constraint_fun(result.x))):.6e}")
        
        return result.x, result.success
    
    def print_optimization_log(self):
        """Print detailed optimization log."""
        if not self.log['energies']:
            print("No optimization steps were logged!")
            return
            
        print("\nDetailed Optimization Log:")
        print("=" * 120)
        print(f"{'Iter':>5} {'Energy':>12} {'Grad Norm':>12} {'Row Sum':>12} {'Area':>12} {'Nonneg':>12}")
        print("-" * 120)
        
        for i in range(len(self.log['iterations'])):
            iter_idx = self.log['iterations'][i]
            energy = self.log['energies'][i]
            grad_norm = self.log.get('gradient_norms', [0.0] * len(self.log['iterations']))[i]
            constraints = self.log['constraint_violations'][i]
            
            print(f"{iter_idx:5d} {energy:12.6f} {grad_norm:12.6f} "
                  f"{constraints['row_sum']:12.6f} {constraints['area']:12.6f} {constraints['nonneg']:12.6f}")
        
        print("\nWarnings and Notable Events:")
        print("=" * 100)
        for warning in self.log['warnings']:
            print(warning)
            
    def callback(self, xk):
        """Callback function to track optimization progress."""
        iter_num = len(self.log['iterations'])
        
        # Store iteration number
        self.log['iterations'].append(iter_num)
        
        # Store current point and compute step size
        self.log['x_history'].append(xk.copy())
        if iter_num > 0:
            step_size = np.linalg.norm(xk - self.log['x_history'][-2])
        else:
            step_size = 0.0
        self.log['step_sizes'].append(step_size)
        
        # Compute and log energy
        energy = self.compute_energy(xk)
        self.log['energies'].append(energy)
        
        # Compute and log constraint violations
        violations = self.check_constraints(xk)  # This returns a dictionary
        self.log['constraint_violations'].append(violations)
        
        # Log energy changes
        if iter_num > 0:
            energy_change = self.log['energies'][-1] - self.log['energies'][-2]
            self.log['optimization_energy_changes'].append(energy_change)
        else:
            self.log['optimization_energy_changes'].append(0.0)
        
        # Print progress every iteration
        print(f"\nIteration {iter_num}:")
        print(f"  Energy: {energy:.6e}")
        print(f"  Constraint violations:")
        print(f"    Row sum: {violations['row_sum']:.6e}")
        print(f"    Area: {violations['area']:.6e}")
        print(f"    Non-negativity: {violations['nonneg']:.6e}")
        print(f"  Step size: {step_size:.6e}")
        if iter_num > 0:
            print(f"  Energy change: {self.log['optimization_energy_changes'][-1]:.6e}")
        
        # Print energy breakdown from cache
        cache_key = self._get_cache_key(xk)
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            print(f"  Energy breakdown:")
            print(f"    Gradient term: {cache_entry['grad_term']:.6e}")
            print(f"    Interface term: {cache_entry['interface_term']:.6e}")
            print(f"    Penalty term: {cache_entry['penalty_term']:.6e}") 