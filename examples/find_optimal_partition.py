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

def optimize_partition(config, solution_dir=None):
    """
    Optimize partition on a torus mesh using SLSQP.
    
    Args:
        config: Configuration object containing mesh and optimization parameters
        use_analytic: Whether to use analytic gradients
        refinement_levels: Number of mesh refinement levels (1 means no refinement)
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
        M, K = mesh.compute_matrices()
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
        optimizer.refinement_level = level
        N = len(v)
        if level == 0:
            np.random.seed(config.seed)
            x0 = np.random.rand(N * config.n_partitions)
            # Project/normalize x0 here
            x0_reshaped = x0.reshape(N, config.n_partitions)
            x0_reshaped = np.clip(x0_reshaped, 0, 1)
            row_sums = np.sum(x0_reshaped, axis=1, keepdims=True)
            x0_reshaped = x0_reshaped / np.maximum(row_sums, 1e-10)
            target_area = np.sum(v) / config.n_partitions
            current_areas = v @ x0_reshaped
            area_scales = target_area / np.maximum(current_areas, target_area/10)
            for i in range(config.n_partitions):
                x0_reshaped[:, i] *= area_scales[i]
            x0 = x0_reshaped.flatten()
        else:
            if config.n_theta_increment == 0 and config.n_phi_increment == 0:
                # Mesh is unchanged, just copy the solution
                x0 = results[-1]['x_opt'].copy()
            else:
                # Mesh changed, interpolate
                x0 = interpolate_solution(results[-1]['x_opt'], results[-1]['mesh'], mesh)
            
        start_time = time.time()
        try:
            x_opt, success = optimizer.optimize(x0, 
                                                maxiter=config.max_iter,
                                                ftol=config.tol,
                                                eps=config.slsqp_eps,
                                                disp=config.slsqp_disp,
                                                use_analytic=use_analytic, 
                                                logger=logger)
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
            'lambda_penalty': config.lambda_penalty
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
    args = parser.parse_args()

    # Load parameters from YAML
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

    # Run the test
    optimize_partition(
        config=config,
        solution_dir=args.solution_dir
    ) 