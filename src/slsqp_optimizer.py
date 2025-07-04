import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.optimize import minimize
from .projection_iterative import orthogonal_projection_iterative
from .mesh import TorusMesh  # Add import for type hints
import matplotlib.pyplot as plt
import logging

class RefinementTriggered(Exception):
    pass

class SLSQPOptimizer:
    def __init__(self, 
                 K: np.ndarray,  # Stiffness matrix
                 M: np.ndarray,  # Mass matrix
                 v: np.ndarray,  # Mass matrix column sums (v = 1ᵀM)
                 n_partitions: int,
                 epsilon: float,  # Interface width parameter
                 lambda_penalty: float = 1.0,
                 starget: Optional[float] = None,
                 refine_patience: int = 30,
                 refine_delta_energy: float = 1e-4,
                 refine_grad_tol: float = 1e-2,
                 refine_constraint_tol: float = 1e-2):
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
            refine_patience: Patience for hybrid refinement trigger
            refine_delta_energy: Energy threshold for refinement
            refine_grad_tol: Gradient tolerance for refinement
            refine_constraint_tol: Constraint tolerance for refinement
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

        self.refine_patience = refine_patience
        self.refine_delta_energy = refine_delta_energy
        self.refine_grad_tol = refine_grad_tol
        self.refine_constraint_tol = refine_constraint_tol
            
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
            'area_evolution': []  # Initialize area_evolution list
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
            grad_interface = (2/self.epsilon) * (self.M @ interface_vec) * (1 - 2*phi_i)
            
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
    
    def compute_area_evolution(self, x):
        """
        Compute the area of each partition at the current point.
        
        Args:
            x: Current point in optimization
            
        Returns:
            Array of areas for each partition
        """
        N = len(self.v)
        n_partitions = self.n_partitions
        x_reshaped = x.reshape(N, n_partitions)
        return self.v @ x_reshaped

    def optimize(self, x0: np.ndarray, maxiter: int = 100, ftol: float = 1e-8, eps: float = 1e-8, 
                disp: bool = False, use_analytic=True, logger=None, log_frequency: int = 50,
                use_last_valid_iterate: bool = True, is_mesh_refinement: bool = False) -> tuple:
        """
        Optimize using SLSQP with optional analytic gradients.
        
        Args:
            x0: Initial point
            maxiter: Maximum number of iterations
            ftol: Function tolerance
            use_analytic: Whether to use analytic gradients
            logger: Logger for optimization progress
            log_frequency: How often to log optimization progress
            use_last_valid_iterate: Whether to use last valid iterate on unsuccessful termination
            is_mesh_refinement: True if this is a mesh refinement step (new mesh resolution)
            
        Returns:
            Tuple of (optimized point, success flag)
        """
        # Set up logger
        if logger is None:
            logger = logging.getLogger('partition_optimization')
            if not logger.hasHandlers():
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                logger.addHandler(ch)
            logger.setLevel(logging.INFO)
        self.logger = logger
        self.logger.info("Starting SLSQP optimization with " + ("analytic" if use_analytic else "finite-difference") + " gradients...")
        
        # Store parameters for callback
        self.log_frequency = log_frequency
        self.use_last_valid_iterate = use_last_valid_iterate
        
        if not is_mesh_refinement:
            # Initialize logging with initial point
            self.log = {
                'iterations': [0],  # Start with iteration 0
                'energies': [self.compute_energy(x0)],  # Initial energy
                'gradient_norms': [np.linalg.norm(self.compute_gradient(x0))],  # Initial gradient
                'constraint_violations': [np.max(np.abs(self.constraint_fun(x0)))],  # Initial violations
                'warnings': [],
                'step_sizes': [0.0],  # Initial step size is 0
                'optimization_energy_changes': [0.0],  # No change at start
                'x_history': [x0.copy()],  # Initial point
                'area_evolution': [self.compute_area_evolution(x0)]  # Initial areas
            }
            
            # Log initial state
            self.logger.info(f"Initial state before optimization:")
            self.logger.info(f"  Energy: {self.log['energies'][0]:.6e}")
            self.logger.info(f"  Gradient norm: {self.log['gradient_norms'][0]:.6e}")
            self.logger.info(f"  Constraint violation: {self.log['constraint_violations'][0]:.6e}")
        else:
            # For continuation, just initialize empty logs
            self.log = {
                'iterations': [],
                'energies': [],
                'gradient_norms': [],
                'constraint_violations': [],
                'warnings': [],
                'step_sizes': [],
                'optimization_energy_changes': [],
                'x_history': [],
                'area_evolution': []
            }
        
        # Set up SLSQP options with tight tolerances
        options = {
            'maxiter': maxiter,
            'ftol': ftol,
            'eps': eps,  # Match finite-difference step size to gradient accuracy
            'disp': disp
        }
        
        # Define constraints with analytic Jacobian
        constraints = [
            {'type': 'eq', 'fun': self.constraint_fun, 'jac': self.constraint_jac}
        ]
        
        # Add bounds [0,1] for all variables
        bounds = [(0.0, 1.0) for _ in range(len(x0))]
        
        # Initialize last valid iterate tracking
        self.prev_x = None
        self.curr_x = None
        
        # Run optimization with or without analytic gradients
        result = minimize(
            self.compute_energy,
            x0,
            method='SLSQP',
            jac=self.compute_gradient if use_analytic else None,
            bounds=bounds,
            constraints=constraints,
            options=options,
            callback=self.callback
        )
        
        self.logger.info("Optimization completed:")
        self.logger.info(f"Success: {result.success}")
        self.logger.info(f"Status: {result.message}")
        self.logger.info(f"Iterations: {result.nit}")
        self.logger.info(f"Final energy: {result.fun:.6e}")
        self.logger.info(f"Final constraint violation: {np.max(np.abs(self.constraint_fun(result.x))):.6e}")
        
        # If not successful (status != 0), use last valid iterate and trim logs
        if hasattr(result, 'status') and result.status != 0 and self.prev_x is not None and self.use_last_valid_iterate:
            self.logger.warning("Returning last valid iterate before unsuccessful termination.\n")
            # Remove last entry from logs (corresponding to problematic final step)
            for key in ['iterations', 'energies', 'gradient_norms', 'constraint_violations', 'step_sizes', 'optimization_energy_changes', 'x_history']:
                if self.log[key]:
                    self.log[key].pop()
            # Compute and append metrics for self.prev_x
            energy = self.compute_energy(self.prev_x)
            grad_norm = np.linalg.norm(self.compute_gradient(self.prev_x))
            violations = np.max(np.abs(self.constraint_fun(self.prev_x)))
            if self.log['x_history']:
                step_size = np.linalg.norm(self.prev_x - self.log['x_history'][-1])
            else:
                step_size = 0.0 # Check this!!!
            self.log['energies'].append(energy)
            self.log['gradient_norms'].append(grad_norm)
            self.log['constraint_violations'].append(violations)
            self.log['step_sizes'].append(step_size)
            self.log['x_history'].append(self.prev_x.copy())
            self.log['iterations'].append(self.log['iterations'][-1]+1 if self.log['iterations'] else 0)
            return self.prev_x.copy(), result.success
        else:
            return result.x, result.success
    
    def callback(self, xk):
        """Callback function to track optimization progress with detailed diagnostics."""
        self.prev_x = getattr(self, 'curr_x', None)
        iter_num = len(self.log['iterations'])
        N = len(self.v)
        n = self.n_partitions
        phi = xk.reshape(N, n)
        self.curr_x = xk.copy()
        if iter_num > 0:
            step_size = np.linalg.norm(xk - self.log['x_history'][-1])
        else:
            step_size = 0.0
        energy = self.compute_energy(xk)
        grad_norm = np.linalg.norm(self.compute_gradient(xk))
        violations = np.max(np.abs(self.constraint_fun(xk)))

        self.log['iterations'].append(iter_num)
        self.log['x_history'].append(xk.copy())
        self.log['step_sizes'].append(step_size)
        self.log['energies'].append(energy)
        self.log['gradient_norms'].append(grad_norm)

        row_sums = np.sum(phi, axis=1)
        area_sums = self.v @ phi
        target_area = self.total_area / n
        row_sum_violations = np.abs(row_sums - 1.0)
        max_row_violation = np.max(row_sum_violations)
        avg_row_violation = np.mean(row_sum_violations)
        worst_row_idx = np.argmax(row_sum_violations)
        area_violations = np.abs(area_sums - target_area)
        max_area_violation = np.max(area_violations)
        avg_area_violation = np.mean(area_violations)
        worst_partition_idx = np.argmax(area_violations)
        min_val = np.min(phi)
        max_val = np.max(phi)
        n_below_0 = np.sum(phi < 0)
        n_above_1 = np.sum(phi > 1)
        
        self.log['constraint_violations'].append(violations)
        if iter_num > 0:
            energy_change = self.log['energies'][-1] - self.log['energies'][-2]
            self.log['optimization_energy_changes'].append(energy_change)
        else:
            self.log['optimization_energy_changes'].append(0.0)
        # --- Hybrid refinement trigger logic ---
        patience = self.refine_patience
        delta_energy = self.refine_delta_energy
        grad_tol = self.refine_grad_tol
        constraint_tol = self.refine_constraint_tol
        if len(self.log['energies']) >= patience:
            energy_change = abs(self.log['energies'][-1] - self.log['energies'][-patience])
            grad_recent = min(self.log['gradient_norms'][-patience:])
            constraint_recent = min(self.log['constraint_violations'][-patience:])
            if (energy_change < delta_energy and grad_recent < grad_tol and constraint_recent < constraint_tol):
                self.logger.info(f"Refinement triggered at iteration {iter_num} by convergence criteria.")
                raise RefinementTriggered()
        # Detailed progress every log_frequency iterations
        if iter_num % self.log_frequency == 0:
            self.logger.debug(f"  Iteration {iter_num}:")
            self.logger.debug(f"  Energy: {energy:.6e}")
            self.logger.debug(f"  Gradient norm: {grad_norm:.6e}")
            self.logger.debug(f"  Overall constraint violation: {violations:.6e}")
            self.logger.debug(f"  Step size: {step_size:.6e}\n")
            if iter_num > 0:
                self.logger.debug(f"  Energy change: {self.log['optimization_energy_changes'][-1]:.6e}")
            self.logger.debug("  Constraint Details:")
            self.logger.debug(f"    Row Sum Constraints:")
            self.logger.debug(f"      Max violation: {max_row_violation:.6e} (at row {worst_row_idx})")
            self.logger.debug(f"      Avg violation: {avg_row_violation:.6e}")
            self.logger.debug(f"      Worst row sum: {row_sums[worst_row_idx]:.6f}")
            self.logger.debug(f"    Area Constraints:")
            self.logger.debug(f"      Max violation: {max_area_violation:.6e} (at partition {worst_partition_idx})")
            self.logger.debug(f"      Avg violation: {avg_area_violation:.6e}")
            self.logger.debug(f"      Target area: {target_area:.6e}")
            self.logger.debug(f"      Worst partition area: {area_sums[worst_partition_idx]:.6e}")
            self.logger.debug(f"    Variable Bounds:")
            self.logger.debug(f"      Min value: {min_val:.6f}")
            self.logger.debug(f"      Max value: {max_val:.6f}")
            self.logger.debug(f"      Number < 0: {n_below_0}")
            self.logger.debug(f"      Number > 1: {n_above_1}\n")
            if iter_num > 0:
                prev_violation = self.log['constraint_violations'][-2]
                if violations > prev_violation * 1.5:
                    self.logger.warning("  WARNING: Significant increase in constraint violation!")
                    self.logger.warning(f"    Previous: {prev_violation:.6e}")
                    self.logger.warning(f"    Current:  {violations:.6e}\n")
                    self.log['warnings'].append(f"Constraint violation increased at iteration {iter_num}")
            if (iter_num > 0) and (iter_num % 500 == 0):
                self.logger.debug("  === Optimization Progress Summary ===")
                self.logger.debug(f"    Initial energy: {self.log['energies'][0]:.6e}")
                self.logger.debug(f"    Current energy: {energy:.6e}")
                self.logger.debug(f"    Energy reduction: {self.log['energies'][0] - energy:.6e}")
                self.logger.debug(f"    Initial constraint violation: {self.log['constraint_violations'][0]:.6e}")
                self.logger.debug(f"    Current constraint violation: {violations:.6e}")
                self.logger.debug(f"    Constraint violation reduction: {self.log['constraint_violations'][0] - violations:.6e}")
                self.logger.debug("  =================================\n")

        # Store area values for each partition
        self.log['area_evolution'].append(area_sums.copy())

    def print_optimization_log(self):
        """Print a summary of the optimization log."""
        self.logger.info("Optimization Log Summary:")
        self.logger.info("=" * 80)
        self.logger.info(f"{'Iteration':>10} {'Energy':>12} {'Grad Norm':>12} {'Constraint':>12} {'Step Size':>12}")
        self.logger.info("-" * 80)
        for i in range(len(self.log['iterations'])):
            self.logger.info(f"{self.log['iterations'][i]:10d} "
                  f"{self.log['energies'][i]:12.6e} "
                  f"{self.log['gradient_norms'][i]:12.6e} "
                  f"{self.log['constraint_violations'][i]:12.6e} "
                  f"{self.log['step_sizes'][i]:12.6e}")

