import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.optimize import minimize
from .projection_iterative import orthogonal_projection_iterative
from .mesh import TorusMesh  # Add import for type hints
import matplotlib.pyplot as plt

class SLSQPOptimizerAnalytic:
    def __init__(self, 
                 K: np.ndarray,  # Stiffness matrix
                 M: np.ndarray,  # Mass matrix
                 v: np.ndarray,  # Mass matrix column sums (v = 1ᵀM)
                 n_partitions: int,
                 epsilon: float,  # Interface width parameter
                 lambda_penalty: float = 1.0,
                 starget: Optional[float] = None):
        """
        Initialize the SLSQP optimizer with analytic gradients for manifold partition optimization.
        
        Args:
            K: Stiffness matrix for gradient term
            M: Mass matrix for area term
            v: Vector of mass matrix column sums (v = 1ᵀM)
            n_partitions: Number of partitions
            mesh: TorusMesh object containing mesh statistics
            lambda_penalty: Initial penalty weight for constant functions
            starget: Target standard deviation (if None, computed from area)
        """
        self.K = K
        self.M = M
        self.v = v
        self.n_partitions = n_partitions
        self.lambda_penalty = lambda_penalty
        
        # Get mesh statistics
        self.total_area = np.sum(v)

        # Set epsilon from mesh statistics
        self.epsilon = epsilon
        
        # Compute target standard deviation if not provided
        if starget is None:
            normalized_area = 1.0/n_partitions
            self.starget = np.sqrt(normalized_area * (1 - normalized_area))
        else:
            self.starget = starget
            
        # Initialize logging
        self.log = {
            'iterations': [],
            'energies': [],
            'gradient_norms': [],
            'constraint_violations': [],
            'warnings': [],
            'step_sizes': [],
            'optimization_energy_changes': [],
            'x_history': [],
                    }
        
    def compute_energy(self, x: np.ndarray) -> float:
        """Compute the total energy of the system."""
        N = len(self.v)
        phi = x.reshape(N, self.n_partitions)
        
        # Compute gradient term
        grad_term = 0
        for i in range(self.n_partitions):
            phi_i = phi[:, i]
            term = self.epsilon * float(phi_i.T @ (self.K @ phi_i))
            grad_term += term
            
        # Compute interface term
        interface_term = 0
        for i in range(self.n_partitions):
            interface_vec = phi[:, i]**2 * (1 - phi[:, i])**2
            term = (1/self.epsilon) * float(interface_vec.T @ (self.M @ interface_vec))
            interface_term += term
            
        # Compute penalty term
        penalty_term = 0
        for i in range(self.n_partitions):
            mean_i = np.sum(self.v * phi[:, i]) / self.total_area
            var_i = np.sum(self.v * (phi[:, i] - mean_i)**2) / self.total_area
            std_i = np.sqrt(var_i + 1e-10)
            term = self.lambda_penalty * (std_i - self.starget)**2
            penalty_term += term
            
        return grad_term + interface_term + penalty_term
    
    def compute_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the analytic gradient of the energy."""
        N = len(self.v)
        phi = x.reshape(N, self.n_partitions)
        grad = np.zeros_like(x)
        
        for i in range(self.n_partitions):
            # Gradient term
            grad_grad = 2 * self.epsilon * (self.K @ phi[:, i])
            
            # Interface term
            phi_i = phi[:, i]
            interface_vec = phi_i**2 * (1 - phi_i)**2
            grad_interface = (2/self.epsilon) * (self.M @ (interface_vec * (1 - 2*phi_i)))
            
            # Penalty term
            mean_i = np.sum(self.v * phi_i) / self.total_area
            var_i = np.sum(self.v * (phi_i - mean_i)**2) / self.total_area
            std_i = np.sqrt(var_i + 1e-10)
            grad_penalty = 2 * self.lambda_penalty * (std_i - self.starget) * \
                         (self.v * (phi_i - mean_i)) / (self.total_area * std_i)
            
            grad[i*N:(i+1)*N] = grad_grad + grad_interface + grad_penalty
        
        return grad
    
    def constraint_fun(self, x: np.ndarray) -> np.ndarray:
        """Compute constraint functions for SLSQP."""
        N = len(self.v)
        n = self.n_partitions
        phi = x.reshape(N, n)
        
        # Row sum constraints (all but last row)
        row_sums = np.sum(phi, axis=1)[:-1] - 1.0
        
        # Area constraints (all but last partition, absolute, not normalized)
        area_sums = self.v @ phi
        target_area = self.total_area / n
        area_constraints = area_sums[:-1] - target_area
        
        return np.concatenate([row_sums, area_constraints])
    
    def constraint_jac(self, x: np.ndarray) -> np.ndarray:
        """Compute analytic Jacobian of constraint functions (vectorized)."""
        N = len(self.v)
        n = self.n_partitions
        
        # Row sum Jacobian: (N-1) x (N*n)
        # For each row i, the Jacobian is 1 for all entries in that row
        row_sum_jac = np.zeros((N-1, N * n))
        for i in range(N-1):
            row_sum_jac[i, i::N] = 1.0
        
        # Area Jacobian: (n-1) x (N*n)
        # For each partition i, the Jacobian is v for that partition's block
        area_jac = np.zeros((n-1, N * n))
        for i in range(n-1):
            area_jac[i, i*N:(i+1)*N] = self.v
        
        return np.vstack([row_sum_jac, area_jac])
    
    def optimize(self, x0: np.ndarray, maxiter: int = 100, ftol: float = 1e-8) -> tuple:
        """
        Optimize using SLSQP with analytic gradients.
        
        Args:
            x0: Initial point
            maxiter: Maximum number of iterations
            ftol: Function tolerance
            
        Returns:
            Tuple of (optimized point, success flag)
        """
        print("\nStarting SLSQP optimization with analytic gradients...")
        
        # Initialize logging
        self.log = {
            'iterations': [],
            'energies': [],
            'gradient_norms': [],
            'constraint_violations': [],
            'warnings': [],
            'step_sizes': [],
            'optimization_energy_changes': [],
            'x_history': []
        }
        
        # Store initial point
        self.log['x_history'].append(x0.copy())
        
        # Project initial point
        N = len(self.v)
        n = self.n_partitions
        x0_reshaped = x0.reshape(N, n)
        x0_reshaped = np.clip(x0_reshaped, 0, 1)  # Enforce bounds [0,1]
        
        # Normalize rows
        row_sums = np.sum(x0_reshaped, axis=1, keepdims=True)
        x0_reshaped = x0_reshaped / np.maximum(row_sums, 1e-10)
        
        # Scale to satisfy area constraints approximately
        target_area = self.total_area / n
        current_areas = self.v @ x0_reshaped
        area_scales = target_area / np.maximum(current_areas, target_area/10)
        for i in range(n):
            x0_reshaped[:, i] *= area_scales[i]
        
        x0 = x0_reshaped.flatten()
        
        # Set up SLSQP options with tight tolerances
        options = {
            'maxiter': maxiter,
            'ftol': 1e-8,
            'eps': 1e-8,  # Match finite-difference step size to gradient accuracy
            'disp': True
        }
        
        # Define constraints with analytic Jacobian
        constraints = [
            {'type': 'eq', 'fun': self.constraint_fun, 'jac': self.constraint_jac}
        ]
        
        # Add bounds [0,1] for all variables
        bounds = [(0.0, 1.0) for _ in range(N * n)]
        
        # Run optimization with analytic gradients
        result = minimize(
            self.compute_energy,
            x0,
            method='SLSQP',
            jac=self.compute_gradient,  # Use analytic gradient
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
    
    def callback(self, xk):
        """Callback function to track optimization progress with detailed diagnostics."""
        iter_num = len(self.log['iterations'])
        N = len(self.v)
        n = self.n_partitions
        phi = xk.reshape(N, n)
        
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
        
        # Compute and log gradient norm
        grad_norm = np.linalg.norm(self.compute_gradient(xk))
        self.log['gradient_norms'].append(grad_norm)
        
        # Detailed constraint analysis
        row_sums = np.sum(phi, axis=1)
        area_sums = self.v @ phi
        target_area = self.total_area / n
        
        # Row sum constraint details
        row_sum_violations = np.abs(row_sums - 1.0)
        max_row_violation = np.max(row_sum_violations)
        avg_row_violation = np.mean(row_sum_violations)
        worst_row_idx = np.argmax(row_sum_violations)
        
        # Area constraint details
        area_violations = np.abs(area_sums - target_area)
        max_area_violation = np.max(area_violations)
        avg_area_violation = np.mean(area_violations)
        worst_partition_idx = np.argmax(area_violations)
        
        # Variable bounds check
        min_val = np.min(phi)
        max_val = np.max(phi)
        n_below_0 = np.sum(phi < 0)
        n_above_1 = np.sum(phi > 1)
        
        # Compute and log overall constraint violations
        violations = np.max(np.abs(self.constraint_fun(xk)))
        self.log['constraint_violations'].append(violations)
        
        # Log energy changes
        if iter_num > 0:
            energy_change = self.log['energies'][-1] - self.log['energies'][-2]
            self.log['optimization_energy_changes'].append(energy_change)
        else:
            self.log['optimization_energy_changes'].append(0.0)
        
        # Print detailed progress every 100 iterations
        if iter_num % 100 == 0:
            print(f"\nIteration {iter_num}:")
            print(f"  Energy: {energy:.6e}")
            print(f"  Gradient norm: {grad_norm:.6e}")
            print(f"  Overall constraint violation: {violations:.6e}")
            print(f"  Step size: {step_size:.6e}")
            if iter_num > 0:
                print(f"  Energy change: {self.log['optimization_energy_changes'][-1]:.6e}")
            
            # Print detailed constraint information
            print("\n  Constraint Details:")
            print(f"    Row Sum Constraints:")
            print(f"      Max violation: {max_row_violation:.6e} (at row {worst_row_idx})")
            print(f"      Avg violation: {avg_row_violation:.6e}")
            print(f"      Worst row sum: {row_sums[worst_row_idx]:.6f}")
            
            print(f"    Area Constraints:")
            print(f"      Max violation: {max_area_violation:.6e} (at partition {worst_partition_idx})")
            print(f"      Avg violation: {avg_area_violation:.6e}")
            print(f"      Target area: {target_area:.6e}")
            print(f"      Worst partition area: {area_sums[worst_partition_idx]:.6e}")
            
            print(f"    Variable Bounds:")
            print(f"      Min value: {min_val:.6f}")
            print(f"      Max value: {max_val:.6f}")
            print(f"      Number < 0: {n_below_0}")
            print(f"      Number > 1: {n_above_1}")
            
            # Print warning if constraints are getting worse
            if iter_num > 0:
                prev_violation = self.log['constraint_violations'][-2]
                if violations > prev_violation * 1.5:  # 50% increase
                    print("\n  WARNING: Significant increase in constraint violation!")
                    print(f"    Previous: {prev_violation:.6e}")
                    print(f"    Current:  {violations:.6e}")
                    self.log['warnings'].append(f"Constraint violation increased at iteration {iter_num}")
            
            # Every 100 iterations, print a more detailed summary
            if iter_num > 0:
                print("\n  === Optimization Progress Summary ===")
                print(f"    Initial energy: {self.log['energies'][0]:.6e}")
                print(f"    Current energy: {energy:.6e}")
                print(f"    Energy reduction: {self.log['energies'][0] - energy:.6e}")
                print(f"    Initial constraint violation: {self.log['constraint_violations'][0]:.6e}")
                print(f"    Current constraint violation: {violations:.6e}")
                print(f"    Constraint violation reduction: {self.log['constraint_violations'][0] - violations:.6e}")
                print("  =================================")

    def plot_optimization_metrics(self, save_path='optimization_metrics.png'):
        """Plot optimization metrics including energy, gradient norm, constraint violations, and step size."""
        if not self.log['energies']:
            print("No optimization steps were logged!")
            return
            
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Energy vs Iterations
        axs[0, 0].plot(self.log['iterations'], self.log['energies'], 'b-', label='Energy')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('Energy')
        axs[0, 0].set_title('Energy Convergence')
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        
        # Plot 2: Gradient Norm vs Iterations
        axs[0, 1].plot(self.log['iterations'], self.log['gradient_norms'], 'r-', label='Gradient Norm')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel('Gradient Norm')
        axs[0, 1].set_title('Gradient Norm Convergence')
        axs[0, 1].grid(True)
        axs[0, 1].legend()
        
        # Plot 3: Constraint Violations vs Iterations
        axs[1, 0].plot(self.log['iterations'], self.log['constraint_violations'], 'g-', label='Overall Constraint Violation')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Constraint Violation')
        axs[1, 0].set_title('Constraint Violation Convergence')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # Plot 4: Step Size vs Iterations
        axs[1, 1].plot(self.log['iterations'], self.log['step_sizes'], 'm-', label='Step Size')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Step Size')
        axs[1, 1].set_title('Step Size Evolution')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def print_optimization_log(self):
        """Print a summary of the optimization log."""
        print("\nOptimization Log Summary:")
        print("=" * 80)
        print(f"{'Iteration':>10} {'Energy':>12} {'Grad Norm':>12} {'Constraint':>12} {'Step Size':>12}")
        print("-" * 80)
        
        for i in range(len(self.log['iterations'])):
            print(f"{self.log['iterations'][i]:10d} "
                  f"{self.log['energies'][i]:12.6e} "
                  f"{self.log['gradient_norms'][i]:12.6e} "
                  f"{self.log['constraint_violations'][i]:12.6e} "
                  f"{self.log['step_sizes'][i]:12.6e}") 