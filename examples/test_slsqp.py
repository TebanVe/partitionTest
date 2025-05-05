import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mesh import TorusMesh
from src.slsqp_optimizer import SLSQPOptimizer
from src.slsqp_optimizer_analytic import SLSQPOptimizerAnalytic
from src.config import Config, TORUS_PARAMS

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
        x_opt, success = optimizer.optimize(x0, maxiter=config.max_iter, ftol=config.tol)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SLSQP optimizer for manifold partition optimization')
    parser.add_argument('--analytic', action='store_true', help='Use analytic gradients instead of finite differences')
    args = parser.parse_args()
    
    test_slsqp(use_analytic=args.analytic) 