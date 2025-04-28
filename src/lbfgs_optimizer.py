import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from typing import Tuple, List, Optional
from .projection_iterative import orthogonal_projection_iterative

class LBFGSOptimizer:
    def __init__(self, 
                 K: np.ndarray,  # Stiffness matrix
                 M: np.ndarray,  # Mass matrix
                 v: np.ndarray,  # Mass matrix column sums (v = 1ᵀM)
                 n_partitions: int,
                 epsilon: float,
                 lambda_penalty: float = 1.0,
                 starget: Optional[float] = None,
                 enable_lambda_tuning: bool = False):
        """
        Initialize the LBFGS optimizer for manifold partition optimization.
        
        Args:
            K: Stiffness matrix for gradient term
            M: Mass matrix for area term
            v: Vector of mass matrix column sums (v = 1ᵀM)
            n_partitions: Number of partitions
            epsilon: Interface width parameter
            lambda_penalty: Initial penalty weight for constant functions
            starget: Target standard deviation (if None, computed from area)
            enable_lambda_tuning: Whether to enable automatic λ-tuning
        """
        self.K = K
        self.M = M
        self.v = v
        self.n_partitions = n_partitions
        self.epsilon = epsilon
        self.lambda_penalty = lambda_penalty
        self.enable_lambda_tuning = enable_lambda_tuning
        
        # Compute target standard deviation if not provided
        if starget is None:
            # Normalize area to [0,1] range
            total_area = np.sum(v)
            normalized_area = 1.0/n_partitions  # Each partition should have 1/n of total area
            self.starget = np.sqrt(normalized_area * (1 - normalized_area))
        else:
            self.starget = starget
            
        # Initialize storage for LBFGS
        self.m = 5  # Number of corrections to store
        
        # Initialize logging
        self.log = {
            'iterations': [],
            'energies': [],
            'gradient_norms': [],
            'line_search_steps': [],
            'projection_violations': [],
            'line_search_details': [],  # Detailed line search info
            'lambda_values': [],  # Track lambda values during auto-tuning
            'std_values': [],  # Track standard deviations during auto-tuning
            'pre_projection_energies': [],
            'post_projection_energies': [],
            'energy_changes': [],
            'gradient_norms_pre_projection': [],
            'gradient_norms_post_projection': [],
            'projection_distances': [],
            'warnings': []
        }
        
        # Auto-tuning parameters (only used if enable_lambda_tuning is True)
        self.lambda_min = 0.1/epsilon
        self.lambda_max = 10.0/epsilon
        self.lambda_increase_factor = 2.0
        self.lambda_decrease_factor = 0.5
        self.std_threshold_low = 0.1 * self.starget
        self.std_threshold_high = 0.9 * self.starget
        self.energy_decrease_threshold = 1e-4
        
    def optimize(self, 
                x0: np.ndarray,
                max_iter: int = 1000,
                tol: float = 1e-6,
                logger: Optional[object] = None) -> Tuple[np.ndarray, float, dict]:
        """
        Perform LBFGS optimization.
        
        Args:
            x0: Initial guess
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            logger: Optional logger object to track optimization progress
            
        Returns:
            Tuple of (optimal solution, final energy value, optimization info)
        """
        # Reset logging
        self.log = {
            'iterations': [],
            'energies': [],
            'gradient_norms': [],
            'line_search_steps': [],
            'projection_violations': [],
            'line_search_details': [],
            'lambda_values': [],
            'std_values': [],
            'pre_projection_energies': [],
            'post_projection_energies': [],
            'energy_changes': [],
            'gradient_norms_pre_projection': [],
            'gradient_norms_post_projection': [],
            'projection_distances': [],
            'warnings': []
        }
        
        # Log initial state before any projection
        initial_energy = self.compute_energy(x0)
        initial_grad = self.compute_gradient(x0)
        initial_grad_norm = np.linalg.norm(initial_grad)
        self.log['warnings'].append(f"Initial state - Energy: {initial_energy:.6f}, Gradient norm: {initial_grad_norm:.6f}")
        
        # Project initial point
        x0_projected = self.project(x0)
        initial_energy_after_projection = self.compute_energy(x0_projected)
        initial_grad_after_projection = self.compute_gradient(x0_projected)
        initial_grad_norm_after_projection = np.linalg.norm(initial_grad_after_projection)
        
        # Log projection effects
        projection_distance = np.linalg.norm(x0_projected - x0)
        energy_change = initial_energy_after_projection - initial_energy
        self.log['warnings'].append(f"After initial projection - Energy: {initial_energy_after_projection:.6f}, " +
                                  f"Gradient norm: {initial_grad_norm_after_projection:.6f}, " +
                                  f"Projection distance: {projection_distance:.6f}, " +
                                  f"Energy change: {energy_change:.6f}")
        
        if abs(energy_change) > 1.0:
            self.log['warnings'].append(f"WARNING: Large energy change during initial projection: {energy_change:.6f}")
        
        # Store initial state in logs
        self.log['iterations'].append(0)
        self.log['energies'].append(initial_energy_after_projection)
        self.log['gradient_norms'].append(initial_grad_norm_after_projection)
        self.log['pre_projection_energies'].append(initial_energy)
        self.log['post_projection_energies'].append(initial_energy_after_projection)
        self.log['gradient_norms_pre_projection'].append(initial_grad_norm)
        self.log['gradient_norms_post_projection'].append(initial_grad_norm_after_projection)
        self.log['projection_distances'].append(projection_distance)
        
        # Define projected objective and gradient
        def objective(x):
            # Project the input point
            x_proj = self.project(x)
            # Compute energy at the projected point
            energy = self.compute_energy(x_proj)
            return energy

        def gradient(x):
            # Project the input point
            x_proj = self.project(x)
            # Compute gradient at the projected point
            grad = self.compute_gradient(x_proj)
            return grad

        # Define callback for logging at each iteration
        def callback(xk):
            current_iter = len(self.log['iterations'])
            # Always project xk for logging
            xk_proj = self.project(xk)
            current_energy = self.compute_energy(xk_proj)
            current_grad = self.compute_gradient(xk_proj)
            current_grad_norm = np.linalg.norm(current_grad)
            projection_distance = np.linalg.norm(xk_proj - xk)
            pre_proj_energy = self.compute_energy(xk)
            energy_change = current_energy - pre_proj_energy

            if abs(energy_change) > 1.0:
                self.log['warnings'].append(f"WARNING: Large energy change at iteration {current_iter}: {energy_change:.6f}")

            if current_iter > 1:
                prev_energy = self.log['energies'][-1]
                if abs(current_energy - prev_energy) < 1e-10:
                    self.log['warnings'].append(f"WARNING: Energy plateau detected at iteration {current_iter}")

            self.log['iterations'].append(current_iter)
            self.log['energies'].append(current_energy)
            self.log['gradient_norms'].append(current_grad_norm)
            self.log['pre_projection_energies'].append(pre_proj_energy)
            self.log['post_projection_energies'].append(current_energy)
            self.log['gradient_norms_pre_projection'].append(np.linalg.norm(self.compute_gradient(xk)))
            self.log['gradient_norms_post_projection'].append(current_grad_norm)
            self.log['projection_distances'].append(projection_distance)
            self.log['energy_changes'].append(energy_change)

        # Run optimization
        result = fmin_l_bfgs_b(
            func=objective,
            x0=x0,
            fprime=gradient,
            maxiter=max_iter,
            callback=callback,
            factr=1e7,
            pgtol=1e-5,
            maxls=50,
            maxfun=15000
        )

        # Log final state
        final_x = result[0]
        final_x_projected = self.project(final_x)
        final_energy = self.compute_energy(final_x_projected)
        final_grad = self.compute_gradient(final_x_projected)
        final_grad_norm = np.linalg.norm(final_grad)
        final_projection_distance = np.linalg.norm(final_x_projected - final_x)
        final_energy_change = final_energy - self.compute_energy(final_x)

        self.log['warnings'].append(f"Final state - Energy: {final_energy:.6f}, Gradient norm: {final_grad_norm:.6f}")
        self.log['warnings'].append(f"After final projection - Energy: {final_energy:.6f}, " +
                                  f"Gradient norm: {final_grad_norm:.6f}, " +
                                  f"Projection distance: {final_projection_distance:.6f}, " +
                                  f"Energy change: {final_energy_change:.6f}")

        return final_x_projected, final_energy, result[2]
        
    def _update_lambda(self, x: np.ndarray) -> None:
        """Update lambda_penalty based on current state."""
        # Reshape x into matrix form
        N = len(self.v)
        phi = x.reshape(N, self.n_partitions)
        
        # Compute standard deviations for each phase
        stds = np.array([np.std(phi[:, i]) for i in range(self.n_partitions)])
        
        # Log current values
        self.log['lambda_values'].append(self.lambda_penalty)
        self.log['std_values'].append(stds)
        
        # Check if any phase is too flat
        if np.any(stds < self.std_threshold_low):
            self.lambda_penalty = min(self.lambda_penalty * self.lambda_increase_factor, self.lambda_max)
            
        # Check if all phases are well-separated
        elif np.all(stds > self.std_threshold_high):
            self.lambda_penalty = max(self.lambda_penalty * self.lambda_decrease_factor, self.lambda_min)
            
        # Check energy decrease rate if we have enough history
        elif len(self.log['energies']) >= 2:
            energy_decrease = self.log['energies'][-2] - self.log['energies'][-1]
            if energy_decrease < self.energy_decrease_threshold:
                self.lambda_penalty = max(self.lambda_penalty * self.lambda_decrease_factor, self.lambda_min)

    def compute_energy(self, x: np.ndarray) -> float:
        """
        Compute the total energy of the system.
        
        Args:
            x: Flattened array of partition functions
            
        Returns:
            Total energy value
        """
        # Reshape x into matrix form
        N = len(self.v)
        phi = x.reshape(N, self.n_partitions)
        
        # Compute gradient term
        grad_term = 0
        for i in range(self.n_partitions):
            grad_term += self.epsilon * phi[:, i].T @ self.K @ phi[:, i]
            
        # Compute interface term
        interface_term = 0
        for i in range(self.n_partitions):
            interface_vec = phi[:, i]**2 * (1 - phi[:, i])**2
            interface_term += (1/self.epsilon) * interface_vec.T @ self.M @ interface_vec
            
        # Compute penalty term for constant functions
        penalty_term = 0
        for i in range(self.n_partitions):
            std = np.std(phi[:, i])
            if std > 0:  # Avoid division by zero
                penalty_term += self.lambda_penalty * (std - self.starget)**2
            else:
                penalty_term += self.lambda_penalty * self.starget**2  # Maximum penalty for constant functions
            
        return grad_term + interface_term + penalty_term
    
    def compute_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the energy.
        
        Args:
            x: Flattened array of partition functions
            
        Returns:
            Flattened gradient array
        """
        N = len(self.v)
        phi = x.reshape(N, self.n_partitions)
        grad = np.zeros_like(x)
        
        # Compute gradient for each partition
        for i in range(self.n_partitions):
            # Gradient term
            grad_grad = 2 * self.epsilon * self.K @ phi[:, i]
            
            # Interface term
            interface_vec = phi[:, i]**2 * (1 - phi[:, i])**2
            grad_interface = (2/self.epsilon) * self.M @ (interface_vec * (1 - 2*phi[:, i]))
            
            # Penalty term
            std = np.std(phi[:, i])
            if std > 0:  # Avoid division by zero
                grad_penalty = 2 * self.lambda_penalty * (std - self.starget) * \
                            (phi[:, i] - np.mean(phi[:, i])) / (N * std)
            else:
                # For constant functions, push towards non-constant
                grad_penalty = 2 * self.lambda_penalty * self.starget * \
                            (phi[:, i] - np.mean(phi[:, i])) / N
            
            # Combine gradients
            grad[i*N:(i+1)*N] = grad_grad + grad_interface + grad_penalty
            
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
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project a point onto the constraint set."""
        N = len(self.v)
        phi = x.reshape(N, self.n_partitions)
        
        # Create target area constraints
        target_area = np.sum(self.v)/self.n_partitions
        d = np.ones(self.n_partitions) * target_area
        
        # Project onto constraints using orthogonal projection
        phi_proj = orthogonal_projection_iterative(
            phi, 
            np.ones(self.n_partitions),
            d,
            self.v,
            max_iter=1000,
            tol=1e-10
        )
        return phi_proj.flatten()

    def print_optimization_log(self):
        """Print detailed optimization log."""
        if not self.log['energies']:
            print("No optimization steps were logged!")
            return
            
        print("\nDetailed Optimization Log:")
        print("=" * 100)
        print(f"{'Iter':>5} {'Energy':>12} {'Grad Norm':>12} {'Pre-Proj Energy':>15} {'Post-Proj Energy':>15} {'Energy Change':>15} {'Proj Distance':>15}")
        print("-" * 100)
        
        for i in range(len(self.log['iterations'])):
            iter_idx = self.log['iterations'][i]
            if iter_idx >= len(self.log['energies']):
                print(f"Warning: Iteration {i} index {iter_idx} out of range")
                continue
                
            energy = self.log['energies'][i]
            grad_norm = self.log['gradient_norms'][i]
            pre_proj_energy = self.log['pre_projection_energies'][i]
            post_proj_energy = self.log['post_projection_energies'][i]
            energy_change = self.log['energy_changes'][i] if i < len(self.log['energy_changes']) else 0.0
            proj_distance = self.log['projection_distances'][i]
            
            print(f"{iter_idx:5d} {energy:12.6f} {grad_norm:12.6f} {pre_proj_energy:15.6f} {post_proj_energy:15.6f} {energy_change:15.6f} {proj_distance:15.6f}")
        
        print("\nWarnings and Notable Events:")
        print("=" * 100)
        for warning in self.log['warnings']:
            print(warning) 