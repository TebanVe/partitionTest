import os
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
from src.slsqp_optimizer_analytic import SLSQPOptimizerAnalytic
from src.config import Config, TORUS_PARAMS

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

def test_slsqp(use_analytic=False):
    """Test SLSQP optimizer for manifold partition optimization."""
    # Create torus mesh using parameters from config
    mesh = TorusMesh(
        n_theta=TORUS_PARAMS['n_theta'],
        n_phi=TORUS_PARAMS['n_phi'],
        R=TORUS_PARAMS['R'],
        r=TORUS_PARAMS['r']
    )
    
    # Compute mass and stiffness matrices
    M, K = mesh.compute_matrices()
    v = np.sum(M.toarray(), axis=0)  # Mass matrix column sums
    
    # Get mesh statistics
    mesh_stats = mesh.mesh_statistics
    #print("\nMesh Statistics:")
    #for key, value in mesh_stats.items():
    #    print(f"{key}: {value:.6e}")
    
    # Calculate epsilon based on mesh size
    epsilon = mesh_stats['avg_triangle_side']
    print(f"\nSetting epsilon to average triangle side length: {epsilon:.6e}")
    
    # Create optimizer with config values
    config = Config()
    if use_analytic:
        optimizer = SLSQPOptimizerAnalytic(
            K=K,
            M=M,
            v=v,
            n_partitions=config.n_partitions,
            epsilon=epsilon,  # Pass epsilon directly
            lambda_penalty=config.lambda_penalty,
        )
        print("\nUsing SLSQP with analytic gradients")
    else:
        optimizer = SLSQPOptimizer(
            K=K,
            M=M,
            v=v,
            n_partitions=config.n_partitions,
            epsilon=epsilon,
            lambda_penalty=config.lambda_penalty
        )
        print("\nUsing SLSQP with finite differences")
    
    # Test with multiple random seeds
    seeds = [42, 123, 456, 789, 101]
    results = []
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Test with seed {seed}")
        print(f"{'='*80}")
        
        # Set random seed
        np.random.seed(seed)
        
        # Generate random initial guess
        N = len(v)
        x0 = np.random.rand(N * config.n_partitions)
        
        # Run optimization
        start_time = time.time()
        x_opt, success = optimizer.optimize(x0, maxiter=config.max_iter, ftol=config.tol, level=0)
        opt_time = time.time() - start_time
        optimizer.print_optimization_log()
        
        # Store results
        results.append({
            'seed': seed,
            'energy': optimizer.compute_energy(x_opt),
            'iterations': optimizer.log['iterations'][-1],
            'time': opt_time,
            'success': success,
            'mesh_stats': mesh_stats  # Store mesh statistics with results
        })
    
    # Print results
    print("\nResults:")
    print("=" * 80)
    print(f"{'Seed':>5} {'Energy':>12} {'Iterations':>10} {'Time (s)':>10} {'Success':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['seed']:5d} "
              f"{r['energy']:12.6f} "
              f"{r['iterations']:10d} "
              f"{r['time']:10.2f} "
              f"{str(r['success']):>10}")
    
    # Plot energy convergence for the best seed
    best_seed = min(results, key=lambda x: x['energy'])['seed']
    np.random.seed(best_seed)
    x0 = np.random.rand(N * config.n_partitions)
    
    # Run optimization again for plotting
    x_opt, _ = optimizer.optimize(x0, maxiter=config.max_iter, ftol=config.tol)
    
    # Plot all optimization metrics
    optimizer.plot_optimization_metrics(save_path='slsqp_optimization_metrics.png')

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

def test_slsqp_with_refinement(use_analytic=False, refinement_levels=4, vertices_increment=1000):
    """
    Test SLSQP optimizer with mesh refinement for manifold partition optimization.
    Args:
        use_analytic: Whether to use analytic gradients
        refinement_levels: Number of mesh refinement levels
        vertices_increment: Number of vertices to add at each refinement
    """
    # Initial mesh parameters
    initial_params = TORUS_PARAMS.copy()
    results = []
    # For unified plotting
    all_energies, all_grad_norms, all_constraints, all_steps = [], [], [], []
    level_boundaries = []
    total_iters = 0
    # Start with initial mesh size
    current_n_theta = initial_params['n_theta']
    current_n_phi = initial_params['n_phi']
    aspect_ratio = current_n_theta / current_n_phi
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config = Config()  # Ensure config is available here
    initial_n_partitions = config.n_partitions
    initial_n_vertices = current_n_theta * current_n_phi
    outdir = f'results/run_{timestamp}_npart{initial_n_partitions}_nvert{initial_n_vertices}'
    os.makedirs(outdir, exist_ok=True)
    logfile_path = os.path.join(outdir, 'run.log')
    logger = setup_logging(logfile_path)
    logger.info(f"Starting partition optimization with {refinement_levels} refinement levels, {vertices_increment} vertices increment, analytic={use_analytic}")
    logger.info(f"Results will be saved in: {outdir}")
    
    for level in range(refinement_levels):
        logger.info(f"{'='*80}")
        logger.info(f"Refinement Level {level + 1}/{refinement_levels}")
        logger.info(f"{'='*80}")
        
        # Create mesh with current parameters
        mesh = TorusMesh(
            n_theta=current_n_theta,
            n_phi=current_n_phi,
            R=initial_params['R'],
            r=initial_params['r']
        )
        
        # Compute matrices and statistics
        M, K = mesh.compute_matrices()
        v = np.sum(M.toarray(), axis=0)
        mesh_stats = mesh.mesh_statistics
        
        # Calculate epsilon based on mesh size
        epsilon = mesh_stats['avg_triangle_side']
        print(f"\nSetting epsilon to average triangle side length: {epsilon:.6e}")
        
        # Create optimizer
        config = Config()
        if use_analytic:
            optimizer = SLSQPOptimizerAnalytic(
                K=K,
                M=M,
                v=v,
                n_partitions=config.n_partitions,
                epsilon=epsilon,
                lambda_penalty=config.lambda_penalty
            )
            print("\nUsing SLSQP with analytic gradients")
        else:
            optimizer = SLSQPOptimizer(
                K=K,
                M=M,
                v=v,
                n_partitions=config.n_partitions,
                epsilon=epsilon,
                lambda_penalty=config.lambda_penalty
            )
            print("\nUsing SLSQP with finite differences")
        
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
        x_opt, success = optimizer.optimize(x0, maxiter=config.max_iter, ftol=config.tol, logger=logger)
        opt_time = time.time() - start_time
        
        # Store results
        results.append({
            'level': level,
            'mesh_params': {'n_theta': current_n_theta, 'n_phi': current_n_phi, 'R': initial_params['R'], 'r': initial_params['r']},
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
        
        # Debug print
        logger.debug(f"Level {level+1}: {len(optimizer.log['energies'])} energies, {len(all_energies)} total energies")
        
        # Print results for this level
        logger.info(f"\nResults for level {level + 1}:")
        logger.info(f"Mesh size: {current_n_theta}x{current_n_phi}")
        logger.info(f"Epsilon: {epsilon:.6e}")
        logger.info(f"Energy: {results[-1]['energy']:.6e}")
        logger.info(f"Time: {opt_time:.2f}s")
        logger.info(f"Success: {success}")
        
        # Update mesh size for next refinement (add ~vertices_increment vertices)
        current_vertices = current_n_theta * current_n_phi
        target_vertices = current_vertices + vertices_increment
        new_n_phi = int(np.sqrt(target_vertices / aspect_ratio))
        new_n_theta = int(aspect_ratio * new_n_phi)
        current_n_theta, current_n_phi = new_n_theta, new_n_phi
    
    # Print summary of all levels
    logger.info("\nRefinement Summary:")
    logger.info("=" * 80)
    logger.info(f"{'Level':>6} {'Mesh Size':>12} {'Epsilon':>12} {'Energy':>12} {'Time (s)':>10}")
    logger.info("-" * 80)
    
    for r in results:
        mesh_size = f"{r['mesh_params']['n_theta']}x{r['mesh_params']['n_phi']}"
        logger.info(f"{r['level']+1:6d} {mesh_size:>12} {r['epsilon']:12.6e} "
              f"{r['energy']:12.6e} {r['time']:10.2f}")
    
    # After the refinement loop, save results
    # Save final solution and mesh as HDF5
    final_result = results[-1]
    x_opt = final_result['x_opt']
    mesh = final_result['mesh']
    with h5py.File(os.path.join(outdir, 'solution.h5'), 'w') as f:
        f.create_dataset('x_opt', data=x_opt)
        f.create_dataset('vertices', data=mesh.vertices)
        if hasattr(mesh, 'faces'):
            f.create_dataset('faces', data=mesh.faces)
    # Save input parameters and metadata as YAML
    meta = {
        'input_parameters': {
            'TORUS_PARAMS': dict(TORUS_PARAMS),
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
    }
    with open(os.path.join(outdir, 'metadata.yaml'), 'w') as f:
        yaml.dump(meta, f)
    logger.info(f"Saved solution to {os.path.join(outdir, 'solution.h5')}")
    logger.info(f"Saved metadata to {os.path.join(outdir, 'metadata.yaml')}")
    # Save the plot as well
    plot_refinement_optimization_metrics(
        all_energies, all_grad_norms, all_constraints, all_steps, level_boundaries,
        save_path=os.path.join(outdir, 'refinement_optimization_metrics.png'),
        n_partitions=initial_n_partitions, N_vertices=initial_n_vertices, lambda_penalty=config.lambda_penalty
    )
    logger.info(f"Saved plot to {os.path.join(outdir, 'refinement_optimization_metrics.png')}")
    print(f"Partition optimization complete. See {logfile_path} for detailed logs.")
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
    parser.add_argument('--refine', action='store_true', help='Use mesh refinement')
    parser.add_argument('--refinement-levels', type=int, default=4, help='Number of mesh refinement levels')
    parser.add_argument('--vertices-increment', type=int, default=1000, help='Number of vertices to add per refinement')
    args = parser.parse_args()
    
    if args.refine:
        test_slsqp_with_refinement(use_analytic=args.analytic, refinement_levels=args.refinement_levels, vertices_increment=args.vertices_increment)
    else:
        test_slsqp(use_analytic=args.analytic) 