import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lbfgs_optimizer import LBFGSOptimizer
from src.mesh import TorusMesh
from src.config import Config

config = Config()

def test_torus_optimization_multiple():
    """Test LBFGS optimizer on torus mesh with multiple random starts."""
    # Create torus mesh
    mesh = TorusMesh(
        n_theta=config.n_theta,
        n_phi=config.n_phi,
        R=config.R,
        r=config.r
    )
    mesh.compute_matrices()
    
    # Get mass and stiffness matrices
    M = mesh.mass_matrix
    K = mesh.stiffness_matrix
    v = np.array(M.sum(axis=0)).flatten()  # Mass matrix column sums
    
    # Initialize optimizer with config parameters
    optimizer = LBFGSOptimizer(
        K=K,
        M=M,
        v=v,
        n_partitions=config.n_partitions,
        epsilon=None,  # Set appropriately if needed
        lambda_penalty=config.lambda_penalty,  # Use config lambda value
        enable_lambda_tuning=False  # Disable automatic tuning
    )
    
    # Run multiple tests with different random seeds
    seeds = [42, 123, 456, 789, 101]
    results = []
    
    for seed in seeds:
        print("\n" + "="*80)
        print(f"Test with seed {seed}")
        print("="*80)
        
        # Set random seed
        np.random.seed(seed)
        
        # Generate random initial guess
        N = len(v)
        x0 = np.random.rand(N * config.n_partitions)
        
        # Run optimization
        x_opt, energy, info = optimizer.optimize(
            x0=x0,
            max_iter=config.max_iter,
            tol=config.tol
        )
        
        # Compute energy before final projection
        energy_before_projection = optimizer.compute_energy(x_opt)
        
        # Store results
        results.append({
            'seed': seed,
            'initial_energy': optimizer.log['energies'][0],
            'final_energy_before_projection': energy_before_projection,
            'final_energy_after_projection': energy,
            'iterations': len(optimizer.log['iterations']),
            'function_calls': len(optimizer.log['energies']),
            'last_iteration_energy': optimizer.log['energies'][-1] if optimizer.log['energies'] else None
        })
        
        # Print optimization log
        optimizer.print_optimization_log()
        
        # Print detailed energy information
        print("\nDetailed Energy Information:")
        print(f"Initial energy (after first projection): {results[-1]['initial_energy']:.6f}")
        print(f"Last iteration energy: {results[-1]['last_iteration_energy']:.6f}")
        print(f"Final energy before projection: {results[-1]['final_energy_before_projection']:.6f}")
        print(f"Final energy after projection: {results[-1]['final_energy_after_projection']:.6f}")
        print(f"Energy decrease during optimization: {results[-1]['initial_energy'] - results[-1]['last_iteration_energy']:.6f}")
        print(f"Energy decrease due to final projection: {results[-1]['final_energy_before_projection'] - results[-1]['final_energy_after_projection']:.6f}")
        print(f"Total energy decrease: {results[-1]['initial_energy'] - results[-1]['final_energy_after_projection']:.6f}")
    
    # Print comparison of results
    print("\nComparison of multiple starts:")
    print("Seed\tInitial\tLast Iter\tBefore Proj\tAfter Proj\tTotal Dec\tIter\tFunc Calls")
    print("-" * 120)
    
    for r in results:
        print(f"{r['seed']}\t{r['initial_energy']:.6f}\t{r['last_iteration_energy']:.6f}\t"
              f"{r['final_energy_before_projection']:.6f}\t{r['final_energy_after_projection']:.6f}\t"
              f"{r['initial_energy'] - r['final_energy_after_projection']:.6f}\t"
              f"{r['iterations']}\t{r['function_calls']}")
    
    # Find best result based on final energy after projection
    best_result = min(results, key=lambda x: x['final_energy_after_projection'])
    print(f"\nBest result from seed {best_result['seed']}")
    print(f"Initial energy: {best_result['initial_energy']:.6f}")
    print(f"Last iteration energy: {best_result['last_iteration_energy']:.6f}")
    print(f"Final energy before projection: {best_result['final_energy_before_projection']:.6f}")
    print(f"Final energy after projection: {best_result['final_energy_after_projection']:.6f}")
    print(f"Energy decrease during optimization: {best_result['initial_energy'] - best_result['last_iteration_energy']:.6f}")
    print(f"Energy decrease due to final projection: {best_result['final_energy_before_projection'] - best_result['final_energy_after_projection']:.6f}")
    print(f"Total energy decrease: {best_result['initial_energy'] - best_result['final_energy_after_projection']:.6f}")

if __name__ == "__main__":
    test_torus_optimization_multiple() 