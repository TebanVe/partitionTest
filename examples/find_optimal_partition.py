import os
# Control thread usage for numerical libraries to prevent oversubscription
# These environment variables ensure single-threaded operation for:
# - OpenMP: Used by many scientific computing libraries
# - MKL: Intel's Math Kernel Library
# - OpenBLAS: Basic Linear Algebra Subprograms
# - NumExpr: Fast numerical expression evaluator
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import argparse
import h5py
import yaml
import datetime
import getpass
import platform
import socket
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mesh import TorusMesh
from src.slsqp_optimizer import SLSQPOptimizer, RefinementTriggered
from src.config import Config
from src.projection_iterative import orthogonal_projection_iterative

def setup_logging(logfile_path):
    logger = logging.getLogger('partition_optimization')
    logger.setLevel(logging.DEBUG)
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    # File handler for detailed logs
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(logging.DEBUG)
    # Console handler for concise output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def plot_refinement_optimization_metrics(
    energies, grad_norms, constraints, steps, level_boundaries, save_path='refinement_optimization_metrics.png',
    n_partitions=None, n_theta_info=None, n_phi_info=None, lambda_penalty=None, seed=None, use_analytic=None
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    x = range(len(energies))
    axs[0, 0].plot(x, energies, 'b-', label='Energy')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Energy')
    axs[0, 0].set_title('Energy Convergence')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 1].plot(x, grad_norms, 'r-', label='Gradient Norm')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Gradient Norm')
    axs[0, 1].set_title('Gradient Norm Convergence')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[1, 0].plot(x, constraints, 'g-', label='Overall Constraint Violation')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Constraint Violation')
    axs[1, 0].set_title('Constraint Violation Convergence')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 1].plot(x, steps, 'm-', label='Step Size')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Step Size')
    axs[1, 1].set_title('Step Size Evolution')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    for ax in axs.flat:
        for boundary in level_boundaries[:-1]:  # skip last, which is end
            ax.axvline(boundary, color='k', linestyle='--', alpha=0.5)
    # Add a meaningful title at the top
    if n_partitions is not None and n_theta_info is not None and n_phi_info is not None and lambda_penalty is not None and seed is not None:
        analytic_str = f", analytic_gradients={'yes' if use_analytic else 'no'}" if use_analytic is not None else ""
        fig.suptitle(f"Partition Optimization: n_partitions={n_partitions}, n_theta={n_theta_info}, n_phi={n_phi_info}, lambda={lambda_penalty}, seed={seed}{analytic_str}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()

def should_refine(energies, grad_norms, constraints, patience=30, delta_energy=1e-4, grad_tol=1e-2, constraint_tol=1e-2):
    if len(energies) < patience:
        return False
    energy_change = abs(energies[-1] - energies[-patience])
    grad_recent = min(grad_norms[-patience:])
    constraint_recent = min(constraints[-patience:])
    return (energy_change < delta_energy and grad_recent < grad_tol and constraint_recent < constraint_tol)

def plot_area_evolution(area_evolution, level_boundaries, save_path='area_evolution.png',
                       n_partitions=None, n_theta_info=None, n_phi_info=None, lambda_penalty=None, seed=None, use_analytic=None):
    """Plot the evolution of areas for each partition."""
    plt.figure(figsize=(12, 6))
    
    # Convert area_evolution to numpy array for easier handling
    area_evolution = np.array(area_evolution)
    n_partitions = area_evolution.shape[1]
    
    # Plot target area
    target_area = np.mean(area_evolution[0])  # Use first iteration as reference
    plt.axhline(y=target_area, color='k', linestyle='-', label='Target Area')
    
    # Plot each partition's area
    for i in range(n_partitions):
        plt.plot(area_evolution[:, i], linestyle='--', label=f'Partition {i+1}')
    
    # Add vertical lines for refinement boundaries
    for boundary in level_boundaries[:-1]:  # skip last, which is end
        plt.axvline(boundary, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel('Iteration')
    plt.ylabel('Area')
    plt.title('Evolution of Partition Areas')
    plt.grid(True)
    plt.legend()
    
    # Add a meaningful title at the top
    if n_partitions is not None and n_theta_info is not None and n_phi_info is not None and lambda_penalty is not None and seed is not None:
        analytic_str = f", analytic_gradients={'yes' if use_analytic else 'no'}" if use_analytic is not None else ""
        plt.suptitle(f"Area Evolution: n_partitions={n_partitions}, n_theta={n_theta_info}, n_phi={n_phi_info}, lambda={lambda_penalty}, seed={seed}{analytic_str}", fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()

def load_initial_condition(h5_path: str, mesh: TorusMesh, n_partitions: int, logger=None) -> np.ndarray:
    """
    Load and validate initial condition from an HDF5 file.
    
    Args:
        h5_path: Path to the HDF5 file containing the solution
        mesh: Current mesh object
        n_partitions: Number of partitions
        logger: Optional logger for output
        
    Returns:
        Initial condition vector x0
    """
    if logger:
        logger.info(f"Loading initial condition from {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # Load the solution
        x_opt = f['x_opt'][:]
        old_vertices = f['vertices'][:]
        
        # Validate dimensions
        N_old = len(old_vertices)
        N_new = len(mesh.vertices)
        
        if logger:
            logger.info(f"Loaded solution with {N_old} vertices and {n_partitions} partitions")
        
        # If meshes are identical, return solution directly
        if N_old == N_new and np.allclose(old_vertices, mesh.vertices):
            if logger:
                logger.info("Meshes are identical, using solution directly")
            return x_opt
        
        # If dimensions don't match but we have a valid solution, interpolate
        if x_opt.shape[0] == N_old * n_partitions:
            if logger:
                logger.info(f"Interpolating solution from {N_old} to {N_new} vertices")
            # Create a temporary mesh object for the old vertices
            old_mesh = TorusMesh(n_theta=mesh.n_theta, n_phi=mesh.n_phi, R=mesh.R, r=mesh.r)
            old_mesh.vertices = old_vertices
            return interpolate_solution(x_opt, old_mesh, mesh)
        
        # If we get here, the solution is invalid
        raise ValueError(f"Solution in {h5_path} has incompatible dimensions. "
                       f"Expected {N_old * n_partitions} elements, got {x_opt.shape[0]}")

def validate_initial_condition(x0: np.ndarray, v: np.ndarray, n_partitions: int, logger=None) -> bool:
    """
    Validate that an initial condition satisfies the partition and area constraints.
    
    Args:
        x0: Initial condition vector
        v: Vector of mass matrix column sums
        n_partitions: Number of partitions
        logger: Optional logger for output
        
    Returns:
        True if constraints are satisfied, False otherwise
    """
    N = len(v)
    phi = x0.reshape(N, n_partitions)
    
    # Check partition constraints (row sums should be 1)
    row_sums = np.sum(phi, axis=1)
    row_violations = np.abs(row_sums - 1.0)
    max_row_violation = np.max(row_violations)
    avg_row_violation = np.mean(row_violations)
    
    # Check area constraints (equal areas)
    area_sums = v @ phi
    target_area = np.sum(v) / n_partitions
    area_violations = np.abs(area_sums - target_area)
    max_area_violation = np.max(area_violations)
    avg_area_violation = np.mean(area_violations)
    
    # Check bounds (should be in [0, 1])
    min_val = np.min(phi)
    max_val = np.max(phi)
    bounds_violation = max(0, -min_val, max_val - 1)
    
    if logger:
        logger.info(f"Initial condition validation:")
        logger.info(f"  Partition constraints: max violation = {max_row_violation:.2e}, avg violation = {avg_row_violation:.2e}")
        logger.info(f"  Area constraints: max violation = {max_area_violation:.2e}, avg violation = {avg_area_violation:.2e}")
        logger.info(f"  Bounds: min = {min_val:.6f}, max = {max_val:.6f}")
    
    # Consider it valid if violations are small
    tol = 1e-6
    is_valid = (max_row_violation < tol and max_area_violation < tol and bounds_violation < tol)
    
    if logger:
        if is_valid:
            logger.info(f"  ✓ Initial condition is feasible")
        else:
            logger.warning(f"  ✗ Initial condition has constraint violations")
    
    return is_valid

def check_analytic_vs_fd_gradient(optimizer, x0, logger=None, eps=1e-6, n_check=10):
    """
    Numerically check analytic gradients against finite-difference gradients at a feasible point.
    Args:
        optimizer: Optimizer object with compute_energy and compute_gradient methods
        x0: Feasible point (1D array)
        logger: Optional logger for output
        eps: Finite-difference step size
        n_check: Number of entries to print for comparison
    """
    def finite_difference_gradient(f, x, eps=1e-6):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += eps
            x2[i] -= eps
            grad[i] = (f(x1) - f(x2)) / (2 * eps)
        return grad

    analytic_grad = optimizer.compute_gradient(x0)
    fd_grad = finite_difference_gradient(optimizer.compute_energy, x0, eps=eps)
    diff = np.linalg.norm(analytic_grad - fd_grad)
    max_abs_diff = np.max(np.abs(analytic_grad - fd_grad))
    if logger:
        logger.info(f"Norm of difference between analytic and finite-difference gradient: {diff:.2e}")
        logger.info(f"Max abs diff: {max_abs_diff:.2e}")
        logger.info(f"Analytic grad (first {n_check}): {analytic_grad[:n_check]}")
        logger.info(f"FD grad (first {n_check}): {fd_grad[:n_check]}")
    else:
        print(f"Norm of difference between analytic and finite-difference gradient: {diff:.2e}")
        print(f"Max abs diff: {max_abs_diff:.2e}")
        print(f"Analytic grad (first {n_check}): {analytic_grad[:n_check]}")
        print(f"FD grad (first {n_check}): {fd_grad[:n_check]}")
    return diff, max_abs_diff

def initialize_random_solution_with_projection(N: int, n_partitions: int, v: np.ndarray, seed: int, projection_max_iter: int = 100) -> np.ndarray:
    """
    Initialize a random solution using the paper's orthogonal projection algorithm.
    
    Args:
        N: Number of vertices in the mesh
        n_partitions: Number of partitions
        v: Vector of mass matrix column sums
        seed: Random seed for reproducibility
        projection_max_iter: Maximum iterations for orthogonal projection
        
    Returns:
        Initialized solution vector x0
    """
    np.random.seed(seed)
    x0 = np.random.rand(N * n_partitions)
    
    # Reshape to matrix form
    A = x0.reshape(N, n_partitions)
    
    # Define constraints
    c = np.ones(n_partitions)  # Row sums should be 1
    d = np.sum(v) / n_partitions * np.ones(n_partitions)  # Equal areas
    
    # Apply orthogonal projection
    A_projected = orthogonal_projection_iterative(
        A, c, d, v, 
        max_iter=projection_max_iter,
        tol=1e-8
    )
    
    return A_projected.flatten()

def initialize_random_solution(N: int, n_partitions: int, v: np.ndarray, seed: int, projection_max_iter: int = 100) -> np.ndarray:
    """
    Initialize a random solution with proper normalization and area constraints.
    Now uses the iterative orthogonal projection algorithm from the paper.
    
    Args:
        N: Number of vertices in the mesh
        n_partitions: Number of partitions
        v: Vector of mass matrix column sums
        seed: Random seed for reproducibility
        projection_max_iter: Maximum iterations for orthogonal projection
        
    Returns:
        Initialized solution vector x0
    """
    return initialize_random_solution_with_projection(N, n_partitions, v, seed, projection_max_iter)

def optimize_partition(config, solution_dir=None):
    """
    Optimize partition on a torus mesh using SLSQP.
    
    Args:
        config: Configuration object containing mesh and optimization parameters
        solution_dir: Optional directory to save solution files (for cluster execution)
    """
    initial_n_theta = config.n_theta
    initial_n_phi = config.n_phi
    initial_n_partitions = config.n_partitions
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    use_analytic = config.use_analytic
    refinement_levels = config.refinement_levels
    # Build mesh info strings for naming
    if refinement_levels > 1:
        final_n_theta = initial_n_theta + (refinement_levels - 1) * config.n_theta_increment
        final_n_phi = initial_n_phi + (refinement_levels - 1) * config.n_phi_increment
        n_theta_info = f"{initial_n_theta}-{final_n_theta}_inct{config.n_theta_increment}"
        n_phi_info = f"{initial_n_phi}-{final_n_phi}_incp{config.n_phi_increment}"
    else:
        n_theta_info = f"{initial_n_theta}"
        n_phi_info = f"{initial_n_phi}"
    outdir = f'results/run_{timestamp}_npart{initial_n_partitions}_nt{n_theta_info}_np{n_phi_info}_lam{config.lambda_penalty}_seed{config.seed}'
    os.makedirs(outdir, exist_ok=True)
    logfile_path = os.path.join(outdir, 'run.log')
    logger = setup_logging(logfile_path)
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)
        logger.info(f"Solutions will be stored in: {solution_dir}")
    results = []
    all_energies, all_grad_norms, all_constraints, all_steps = [], [], [], []
    level_boundaries = []
    total_iters = 0
    logger.info(f"Starting partition optimization with {refinement_levels} refinement levels, n_theta_increment={config.n_theta_increment}, n_phi_increment={config.n_phi_increment}, analytic={use_analytic}, seed={config.seed}")
    logger.info(f"Results will be saved in: {outdir}")
    for level in range(refinement_levels):
        logger.info(f"{'='*80}")
        logger.info(f"Refinement Level {level + 1}/{refinement_levels}")
        logger.info(f"{'='*80}")
        logger.info(f"Creating mesh with parameters: n_theta={config.n_theta}, n_phi={config.n_phi}, R={config.R}, r={config.r}")
        mesh = TorusMesh(
            n_theta=config.n_theta,
            n_phi=config.n_phi,
            R=config.R,
            r=config.r
        )
        logger.info(f"Mesh created with size: {mesh.n_theta}x{mesh.n_phi}")
        M, K = mesh.compute_matrices_stable_fem()
        v = np.sum(M.toarray(), axis=0)
        mesh_stats = mesh.mesh_statistics
        epsilon = mesh_stats['avg_triangle_side']
        logger.info(f"Setting epsilon to average triangle side length: {epsilon:.6e}")
        optimizer = SLSQPOptimizer(
            K=K,
            M=M,
            v=v,
            n_partitions=config.n_partitions,
            epsilon=epsilon,
            lambda_penalty=config.lambda_penalty,
            starget=config.starget,
            refine_patience=int(getattr(config, 'refine_patience', 30)),
            refine_delta_energy=float(getattr(config, 'refine_delta_energy', 1e-4)),
            refine_grad_tol=float(getattr(config, 'refine_grad_tol', 1e-2)),
            refine_constraint_tol=float(getattr(config, 'refine_constraint_tol', 1e-2))
        )
        
        N = len(v)
        if level == 0:
            if config.use_custom_initial_condition and config.initial_condition_path:
                try:
                    x0 = load_initial_condition(config.initial_condition_path, mesh, config.n_partitions, logger)
                    logger.info("Successfully loaded and interpolated initial condition")
                except Exception as e:
                    logger.error(f"Failed to load/interpolate initial condition: {e}")
                    if getattr(config, 'allow_random_fallback', True):
                        logger.info("Falling back to random initialization")
                        logger.info(f"Creating random initial condition using orthogonal projection (max_iter={config.projection_max_iter})")
                        x0 = initialize_random_solution(N, config.n_partitions, v, config.seed, config.projection_max_iter)
                        validate_initial_condition(x0, v, config.n_partitions, logger)
                    else:
                        raise RuntimeError("Failed to load initial condition and random fallback is disabled")
            else:
                # Original random initialization code
                logger.info(f"Creating random initial condition using orthogonal projection (max_iter={config.projection_max_iter})")
                x0 = initialize_random_solution(N, config.n_partitions, v, config.seed, config.projection_max_iter)
                validate_initial_condition(x0, v, config.n_partitions, logger)
        else:
            if config.n_theta_increment == 0 and config.n_phi_increment == 0:
                # Mesh is unchanged, just copy the solution
                x0 = results[-1]['x_opt'].copy()
            else:
                # Mesh changed, interpolate
                logger.info("Interpolating solution from previous mesh to new mesh")
                x0 = interpolate_solution(results[-1]['x_opt'], results[-1]['mesh'], mesh)
                # Project to feasible region after mesh refinement
                logger.info(f"Projecting interpolated solution to feasible region (max_iter={config.projection_max_iter})")
                A = x0.reshape(N, config.n_partitions)
                c = np.ones(config.n_partitions)
                d = np.sum(v) / config.n_partitions * np.ones(config.n_partitions)
                A_projected = orthogonal_projection_iterative(A, c, d, v, max_iter=config.projection_max_iter, tol=1e-8)
                x0 = A_projected.flatten()
                validate_initial_condition(x0, v, config.n_partitions, logger)
        # Gradient check (only if using analytic gradients)
        if use_analytic:
            logger.info("Checking analytic vs finite-difference gradients at projected feasible point...")
            check_analytic_vs_fd_gradient(optimizer, x0, logger=logger, eps=1e-6, n_check=10)
        # Determine if this is a mesh refinement step
        is_mesh_refinement = (
            #(level == 0 and not config.use_custom_initial_condition) or  # Only log initial state for random initialization at level 0
            (level > 0 and (config.n_theta_increment > 0 or config.n_phi_increment > 0))  # Log initial state for mesh resolution changes
        )
        
        # Add debug logging
        logger.debug(f"Debug is_mesh_refinement calculation:")
        logger.debug(f"  level = {level}")
        logger.debug(f"  use_custom_initial_condition = {config.use_custom_initial_condition}")
        logger.debug(f"  n_theta_increment = {config.n_theta_increment}")
        logger.debug(f"  n_phi_increment = {config.n_phi_increment}")
        logger.debug(f"  is_mesh_refinement = {is_mesh_refinement}")
        
        start_time = time.time()
        try:
            x_opt, success = optimizer.optimize(x0, 
                                                maxiter=config.max_iter,
                                                ftol=config.tol,
                                                eps=config.slsqp_eps,
                                                disp=config.slsqp_disp,
                                                use_analytic=use_analytic, 
                                                logger=logger,
                                                log_frequency=config.log_frequency,
                                                use_last_valid_iterate=config.use_last_valid_iterate,
                                                is_mesh_refinement=is_mesh_refinement)
        except RefinementTriggered:
            logger.info(f"Refinement triggered early at level {level+1} by convergence criteria.")
            x_opt = optimizer.log['x_history'][-1]
            success = False
        opt_time = time.time() - start_time
        results.append({
            'level': level,
            'mesh_params': {'n_theta': config.n_theta, 'n_phi': config.n_phi, 'R': config.R, 'r': config.r},
            'mesh_stats': mesh_stats,
            'epsilon': epsilon,
            'x_opt': x_opt,
            'energy': optimizer.compute_energy(x_opt),
            'iterations': optimizer.log['iterations'][-1],
            'time': opt_time,
            'success': success,
            'mesh': mesh,
            'optimizer': optimizer
        })
        all_energies.extend(optimizer.log['energies'])
        all_grad_norms.extend(optimizer.log['gradient_norms'])
        all_constraints.extend(optimizer.log['constraint_violations'])
        all_steps.extend(optimizer.log['step_sizes'])
        total_iters += len(optimizer.log['energies'])
        level_boundaries.append(total_iters)
        logger.info(f"Results for level {level + 1}:")
        logger.info(f"  Energy: {results[-1]['energy']:.6e}")
        logger.info(f"  Time: {opt_time:.2f}s")
        logger.info(f"  Success: {success}\n")
        # Always refine mesh for next level if refinement_levels > 1
        if level < refinement_levels - 1:
            config.n_theta += config.n_theta_increment
            config.n_phi += config.n_phi_increment
    logger.info("Refinement Summary:")
    logger.info("=" * 80)
    logger.info(f"{'Level':>6} {'Mesh Size':>12} {'Energy':>12} {'Time (s)':>10}")
    logger.info("-" * 80)
    for r in results:
        mesh_size = f"{r['mesh_params']['n_theta']}x{r['mesh_params']['n_phi']}"
        logger.info(f"{r['level']+1:6d} {mesh_size:>12} {r['energy']:12.6e} {r['time']:10.2f}")
    final_result = results[-1]
    x_opt = final_result['x_opt']
    mesh = final_result['mesh']
    if solution_dir:
        solution_filename = f"part{config.n_partitions}_nt{n_theta_info}_np{n_phi_info}_lam{config.lambda_penalty}_seed{config.seed}_{timestamp}.h5"
    else:
        solution_filename = f"part{config.n_partitions}_nt{n_theta_info}_np{n_phi_info}_lam{config.lambda_penalty}_seed{config.seed}_{timestamp}.h5"
    solution_path = os.path.join(solution_dir if solution_dir else outdir, solution_filename)
    with h5py.File(solution_path, 'w') as f:
        f.create_dataset('x_opt', data=x_opt)
        f.create_dataset('vertices', data=mesh.vertices)
        f.create_dataset('faces', data=mesh.faces, dtype='i4')  # or 'i8' for 64-bit integers
    meta = {
        'input_parameters': {
            'TORUS_PARAMS': {
                'n_theta': config.n_theta,
                'n_phi': config.n_phi,
                'R': config.R,
                'r': config.r
            },
            'refinement_levels': refinement_levels,
            'n_theta_increment': config.n_theta_increment,
            'n_phi_increment': config.n_phi_increment,
            'use_analytic': use_analytic,
            'seed': config.seed,
            'lambda_penalty': config.lambda_penalty,
            'use_custom_initial_condition': config.use_custom_initial_condition,
            'initial_condition_path': config.initial_condition_path
        },
        'final_mesh_stats': final_result['mesh_stats'],
        'final_epsilon': float(final_result['epsilon']),
        'final_energy': float(final_result['energy']),
        'final_iterations': int(final_result['iterations']),
        'run_time_seconds': float(final_result['time']),
        'success': bool(final_result['success']),
        'datetime': timestamp,
        'user': getpass.getuser(),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'solution_path': solution_path
    }
    with open(os.path.join(outdir, 'metadata.yaml'), 'w') as f:
        yaml.dump(meta, f)
    plot_path = os.path.join(outdir, 'refinement_optimization_metrics.png')
    plot_refinement_optimization_metrics(
        all_energies, all_grad_norms, all_constraints, all_steps, level_boundaries,
        save_path=plot_path,
        n_partitions=initial_n_partitions, n_theta_info=n_theta_info, n_phi_info=n_phi_info, lambda_penalty=config.lambda_penalty, seed=config.seed, use_analytic=config.use_analytic
    )
    logger.info(f"Saved optimization metrics plot to {plot_path}")
    
    # Add new area evolution plot
    area_evolution = []
    for result in results:
        area_evolution.extend(result['optimizer'].log['area_evolution'])
    
    area_plot_path = os.path.join(outdir, 'area_evolution.png')
    plot_area_evolution(
        area_evolution, level_boundaries,
        save_path=area_plot_path,
        n_partitions=initial_n_partitions, n_theta_info=n_theta_info, n_phi_info=n_phi_info, lambda_penalty=config.lambda_penalty, seed=config.seed, use_analytic=use_analytic
    )
    logger.info(f"Saved area evolution plot to {area_plot_path}")
    
    print(f"Partition optimization complete. See {logfile_path} for detailed logs.\n")
    return results

def interpolate_solution(old_x, old_mesh, new_mesh):
    """
    Interpolate solution from old mesh to new mesh.
    Args:
        old_x: Solution vector on old mesh
        old_mesh: Old mesh object
        new_mesh: New mesh object
    Returns:
        Interpolated solution vector on new mesh
    """
    N_old = old_mesh.vertices.shape[0]
    N_new = new_mesh.vertices.shape[0]
    n_partitions = old_x.shape[0] // N_old
    old_phi = old_x.reshape(N_old, n_partitions)
    new_phi = np.zeros((N_new, n_partitions))
    for i in range(N_new):
        new_point = new_mesh.vertices[i]
        distances = np.linalg.norm(old_mesh.vertices - new_point, axis=1)
        closest_idx = np.argmin(distances)
        new_phi[i] = old_phi[closest_idx]
    return new_phi.flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SLSQP optimizer for manifold partition optimization')
    parser.add_argument('--input', type=str, required=True, help='Path to input YAML file with parameters')
    parser.add_argument('--solution-dir', type=str, help='Directory for storing solution files (if not provided, uses local results directory)')
    parser.add_argument('--initial-condition', type=str, help='Path to .h5 file containing initial condition')
    args = parser.parse_args()

    # Load parameters from YAML
    print(f"\nLoading parameters from {args.input}")
    with open(args.input, 'r') as f:
        params = yaml.safe_load(f)
    
    # Override initial condition path if provided via command line
    if args.initial_condition:
        params['use_custom_initial_condition'] = True
        params['initial_condition_path'] = args.initial_condition
    
    # Create config and check for overridden parameters
    config = Config(params)
    overridden = {k: v for k, v in params.items() if hasattr(config, k) and getattr(config, k) != v}
    if overridden:
        print("\nParameters overridden:")
        for k, v in overridden.items():
            print(f"  {k}: {v}")

    # Run the test
    optimize_partition(
        config=config,
        solution_dir=args.solution_dir
    ) 