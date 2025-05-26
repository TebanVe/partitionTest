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
from src.slsqp_optimizer import SLSQPOptimizer
from src.config import Config

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
    n_partitions=None, n_vertices=None, lambda_penalty=None
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
    if n_partitions is not None and n_vertices is not None and lambda_penalty is not None:
        fig.suptitle(f"Partition Optimization: n_partitions={n_partitions}, n_vertices={n_vertices}, lambda={lambda_penalty}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()

def optimize_partition(config, use_analytic=False, refinement_levels=1, vertices_increment=1000, solution_dir=None):
    """
    Optimize partition on a torus mesh using SLSQP.
    
    Args:
        config: Configuration object containing mesh and optimization parameters
        use_analytic: Whether to use analytic gradients
        refinement_levels: Number of mesh refinement levels (1 means no refinement)
        vertices_increment: Number of vertices to add at each refinement
        solution_dir: Optional directory to save solution files (for cluster execution)
    """
    # Store initial values for reference
    initial_n_theta = config.n_theta
    initial_n_phi = config.n_phi
    initial_aspect_ratio = initial_n_theta / initial_n_phi
    
    # Setup logging and output directory first
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    initial_n_partitions = config.n_partitions
    initial_n_vertices = initial_n_theta * initial_n_phi
    outdir = f'results/run_{timestamp}_npart{initial_n_partitions}_nvert{initial_n_vertices}'
    os.makedirs(outdir, exist_ok=True)
    logfile_path = os.path.join(outdir, 'run.log')
    logger = setup_logging(logfile_path)
    
    # If solution_dir is provided, ensure it exists
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)
        logger.info(f"Solutions will be stored in: {solution_dir}")
    
    results = []
    
    # For unified plotting
    all_energies, all_grad_norms, all_constraints, all_steps = [], [], [], []
    level_boundaries = []
    total_iters = 0
    
    logger.info(f"Starting partition optimization with {refinement_levels} refinement levels, {vertices_increment} vertices increment, analytic={use_analytic}")
    logger.info(f"Results will be saved in: {outdir}")
    
    for level in range(refinement_levels):
        logger.info(f"{'='*80}")
        logger.info(f"Refinement Level {level + 1}/{refinement_levels}")
        logger.info(f"{'='*80}")
        
        # Create mesh with current parameters
        logger.info(f"Creating mesh with parameters: n_theta={config.n_theta}, n_phi={config.n_phi}, R={config.R}, r={config.r}")
        mesh = TorusMesh(
            n_theta=config.n_theta,
            n_phi=config.n_phi,
            R=config.R,
            r=config.r
        )
        logger.info(f"Mesh created with size: {mesh.n_theta}x{mesh.n_phi}")
        logger.info(f"Current aspect ratio: {config.n_theta/config.n_phi:.3f}")
        
        # Compute matrices and statistics
        M, K = mesh.compute_matrices()
        v = np.sum(M.toarray(), axis=0)
        mesh_stats = mesh.mesh_statistics
        
        # Calculate epsilon based on mesh size
        epsilon = mesh_stats['avg_triangle_side']
        logger.info(f"Setting epsilon to average triangle side length: {epsilon:.6e}")
        
        # Create optimizer
        optimizer = SLSQPOptimizer(
            K=K,
            M=M,
            v=v,
            n_partitions=config.n_partitions,
            epsilon=epsilon,
            lambda_penalty=config.lambda_penalty,
            starget=config.starget
        )
        logger.info(f"Using SLSQP with {'analytic' if use_analytic else 'finite-difference'} gradients")
        
        # Set refinement level attribute for logging
        optimizer.refinement_level = level
        
        # Generate initial guess
        N = len(v)
        if level == 0:
            # First level: random initial guess
            np.random.seed(42)  # Fixed seed for reproducibility
            x0 = np.random.rand(N * config.n_partitions)
        else:
            # Interpolate solution from previous level
            x0 = interpolate_solution(results[-1]['x_opt'], results[-1]['mesh'], mesh)
        
        # Run optimization
        start_time = time.time()
        x_opt, success = optimizer.optimize(x0, maxiter=config.max_iter, ftol=config.tol, use_analytic=use_analytic, logger=logger)
        opt_time = time.time() - start_time
        
        # Store results
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
            'mesh': mesh  # Store mesh object for interpolation
        })
        
        # Aggregate logs for unified plot
        all_energies.extend(optimizer.log['energies'])
        all_grad_norms.extend(optimizer.log['gradient_norms'])
        all_constraints.extend(optimizer.log['constraint_violations'])
        all_steps.extend(optimizer.log['step_sizes'])
        total_iters += len(optimizer.log['energies'])
        level_boundaries.append(total_iters)
        
        # Print results for this level
        logger.info(f"Results for level {level + 1}:")
        logger.info(f"  Energy: {results[-1]['energy']:.6e}")
        logger.info(f"  Time: {opt_time:.2f}s")
        logger.info(f"  Success: {success}\n")
        
        # Update mesh size for next refinement (add ~vertices_increment vertices)
        if level < refinement_levels - 1:  # Don't update on last iteration
            current_vertices = config.n_theta * config.n_phi
            target_vertices = current_vertices + vertices_increment
            
            # Calculate new dimensions while maintaining aspect ratio
            new_n_phi = int(np.sqrt(target_vertices / initial_aspect_ratio))
            new_n_theta = int(initial_aspect_ratio * new_n_phi)
            
            logger.info(f"Refining mesh:")
            logger.info(f"  Current: {config.n_theta}x{config.n_phi} ({current_vertices} vertices)")
            logger.info(f"  Target: {new_n_theta}x{new_n_phi} ({target_vertices} vertices)")
            
            config.n_theta = new_n_theta
            config.n_phi = new_n_phi
    
    # Print summary of all levels
    logger.info("Refinement Summary:")
    logger.info("=" * 80)
    logger.info(f"{'Level':>6} {'Mesh Size':>12} {'Energy':>12} {'Time (s)':>10}")
    logger.info("-" * 80)
    
    for r in results:
        mesh_size = f"{r['mesh_params']['n_theta']}x{r['mesh_params']['n_phi']}"
        logger.info(f"{r['level']+1:6d} {mesh_size:>12} {r['energy']:12.6e} {r['time']:10.2f}")
    
    # Save results
    final_result = results[-1]
    x_opt = final_result['x_opt']
    mesh = final_result['mesh']
    
    # Determine solution file path with descriptive naming for cluster runs
    if solution_dir:
        # For cluster runs, use descriptive filename with parameters and timestamp
        solution_filename = f"part{config.n_partitions}_vert{config.n_theta * config.n_phi}_{timestamp}.h5"
    else:
        # For local runs, keep the current behavior with simple filename
        solution_filename = 'solution.h5'
    
    solution_path = os.path.join(solution_dir if solution_dir else outdir, solution_filename)
    
    # Save solution and mesh as HDF5
    with h5py.File(solution_path, 'w') as f:
        f.create_dataset('x_opt', data=x_opt)
        f.create_dataset('vertices', data=mesh.vertices)
        if hasattr(mesh, 'faces'):
            f.create_dataset('faces', data=mesh.faces)
    
    # Save metadata as YAML (always in local results directory)
    meta = {
        'input_parameters': {
            'TORUS_PARAMS': {
                'n_theta': config.n_theta,
                'n_phi': config.n_phi,
                'R': config.R,
                'r': config.r
            },
            'refinement_levels': refinement_levels,
            'vertices_increment': vertices_increment,
            'use_analytic': use_analytic,
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
        'solution_path': solution_path  # Add solution path to metadata
    }
    with open(os.path.join(outdir, 'metadata.yaml'), 'w') as f:
        yaml.dump(meta, f)
    
    logger.info(f"Saved solution to {solution_path}")
    logger.info(f"Saved metadata to {os.path.join(outdir, 'metadata.yaml')}")
    
    # Save the plot (always in local results directory)
    plot_path = os.path.join(outdir, 'refinement_optimization_metrics.png')
    plot_refinement_optimization_metrics(
        all_energies, all_grad_norms, all_constraints, all_steps, level_boundaries,
        save_path=plot_path,
        n_partitions=initial_n_partitions, n_vertices=initial_n_vertices, lambda_penalty=config.lambda_penalty
    )
    logger.info(f"Saved plot to {plot_path}")
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
    parser.add_argument('--analytic', action='store_true', help='Use analytic gradients instead of finite differences')
    parser.add_argument('--refinement-levels', type=int, default=1, help='Number of mesh refinement levels (1 means no refinement)')
    parser.add_argument('--vertices-increment', type=int, default=1000, help='Number of vertices to add per refinement')
    parser.add_argument('--input', type=str, help='Path to input YAML file with parameters')
    parser.add_argument('--solution-dir', type=str, help='Directory for storing solution files (if not provided, uses local results directory)')
    args = parser.parse_args()

    # Load parameters from YAML if provided
    if args.input:
        print(f"\nLoading parameters from {args.input}")
        with open(args.input, 'r') as f:
            params = yaml.safe_load(f)
        
        # Create config and check for overridden parameters
        config = Config(params)
        overridden = {k: v for k, v in params.items() if hasattr(config, k) and getattr(config, k) != v}
        if overridden:
            print("\nParameters overridden:")
            for k, v in overridden.items():
                print(f"  {k}: {v}")
    else:
        config = Config()

    # Run the test
    optimize_partition(
        config=config,
        use_analytic=args.analytic,
        refinement_levels=args.refinement_levels,
        vertices_increment=args.vertices_increment,
        solution_dir=args.solution_dir
    ) 