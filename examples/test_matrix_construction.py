import os
import sys
import time
import argparse
import yaml
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mesh import TorusMesh
from src.config import Config

def setup_logging(logfile_path):
    """Set up logging configuration."""
    logger = logging.getLogger('matrix_test')
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

def analyze_matrix_properties(M, K, method_name, logger):
    """
    Analyze properties of mass and stiffness matrices.
    
    Parameters:
    -----------
    M : scipy.sparse.csr_matrix
        Mass matrix
    K : scipy.sparse.csr_matrix
        Stiffness matrix
    method_name : str
        Name of the method being analyzed
    logger : logging.Logger
        Logger for output
    """
    logger.info(f"{'='*20} {method_name} Analysis {'='*20}")
    
    # Convert to dense for analysis
    M_array = M.toarray()
    K_array = K.toarray()
    
    # Basic properties
    logger.info(f"Mass matrix shape: {M.shape}")
    logger.info(f"Mass matrix non-zero elements: {M.nnz}")
    logger.info(f"Stiffness matrix non-zero elements: {K.nnz}")
    
    # Symmetry check
    M_symmetric = np.allclose(M_array, M_array.T)
    K_symmetric = np.allclose(K_array, K_array.T)
    logger.info(f"Mass matrix is symmetric: {M_symmetric}")
    logger.info(f"Stiffness matrix is symmetric: {K_symmetric}")
    
    # Numerical issues check
    logger.info(f"Mass matrix contains NaN: {np.isnan(M_array).any()}")
    logger.info(f"Mass matrix contains Inf: {np.isinf(M_array).any()}")
    logger.info(f"Stiffness matrix contains NaN: {np.isnan(K_array).any()}")
    logger.info(f"Stiffness matrix contains Inf: {np.isinf(K_array).any()}")
    
    # Condition numbers
    M_cond = np.linalg.cond(M_array)
    K_cond = np.linalg.cond(K_array)
    logger.info(f"Mass matrix condition number: {M_cond:.2e}")
    logger.info(f"Stiffness matrix condition number: {K_cond:.2e}")
    
    # Eigenvalue analysis
    M_eigenvals = np.linalg.eigvals(M_array)
    K_eigenvals = np.linalg.eigvals(K_array)
    
    logger.info(f"Mass matrix - Min eigenvalue: {np.min(M_eigenvals):.2e}")
    logger.info(f"Mass matrix - Max eigenvalue: {np.max(M_eigenvals):.2e}")
    logger.info(f"Mass matrix - Negative eigenvalues: {np.sum(M_eigenvals < 0)}")
    logger.info(f"Mass matrix - Zero eigenvalues (< 1e-10): {np.sum(np.abs(M_eigenvals) < 1e-10)}")
    
    logger.info(f"Stiffness matrix - Min eigenvalue: {np.min(K_eigenvals):.2e}")
    logger.info(f"Stiffness matrix - Max eigenvalue: {np.max(K_eigenvals):.2e}")
    logger.info(f"Stiffness matrix - Negative eigenvalues: {np.sum(K_eigenvals < 0)}")
    logger.info(f"Stiffness matrix - Zero eigenvalues (< 1e-10): {np.sum(np.abs(K_eigenvals) < 1e-10)}")
    
    return M_eigenvals, K_eigenvals

def plot_eigenvalue_comparison(results, save_path):
    """Plot eigenvalue comparisons between methods."""
    plt.figure(figsize=(15, 10))
    
    # Mass matrix eigenvalues
    plt.subplot(221)
    for method_name, (M_eig, K_eig) in results.items():
        plt.semilogy(np.sort(np.abs(M_eig)), '.', label=method_name, markersize=2)
    plt.title('Mass Matrix Eigenvalue Distribution')
    plt.xlabel('Index')
    plt.ylabel('|Eigenvalue|')
    plt.grid(True)
    plt.legend()
    
    # Stiffness matrix eigenvalues
    plt.subplot(222)
    for method_name, (M_eig, K_eig) in results.items():
        plt.semilogy(np.sort(np.abs(K_eig)), '.', label=method_name, markersize=2)
    plt.title('Stiffness Matrix Eigenvalue Distribution')
    plt.xlabel('Index')
    plt.ylabel('|Eigenvalue|')
    plt.grid(True)
    plt.legend()
    
    # Histogram of mass matrix eigenvalues
    plt.subplot(223)
    for method_name, (M_eig, K_eig) in results.items():
        plt.hist(M_eig, bins=50, alpha=0.3, label=method_name)
    plt.title('Mass Matrix Eigenvalue Histogram')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Count')
    plt.legend()
    
    # Histogram of stiffness matrix eigenvalues
    plt.subplot(224)
    for method_name, (M_eig, K_eig) in results.items():
        plt.hist(K_eig, bins=50, alpha=0.3, label=method_name)
    plt.title('Stiffness Matrix Eigenvalue Histogram')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_matrix_construction(config):
    """
    Test and compare different matrix construction methods.
    
    Parameters:
    -----------
    config : Config
        Configuration object containing test parameters
    """
    # Create output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = f"{config.matrix_test_output_dir}/run_{timestamp}_nt{config.n_theta}_np{config.n_phi}_R{config.R}_r{config.r}"
    os.makedirs(outdir, exist_ok=True)
    
    # Setup logging
    logfile_path = os.path.join(outdir, 'matrix_test.log')
    logger = setup_logging(logfile_path)
    
    logger.info(f"Starting matrix construction test")
    logger.info(f"Results will be saved in: {outdir}")
    logger.info(f"Mesh parameters: n_theta={config.n_theta}, n_phi={config.n_phi}, R={config.R}, r={config.r}")
    
    # Create a torus mesh
    logger.info("Creating torus mesh...")
    mesh = TorusMesh(n_theta=config.n_theta, n_phi=config.n_phi, R=config.R, r=config.r)
    logger.info(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    
    results = {}
    matrix_data = {}
    
    # Test methods based on configuration
    methods_to_test = []
    if config.test_barycentric:
        methods_to_test.append(("Barycentric", mesh.compute_matrices))
    if config.test_stable:
        methods_to_test.append(("Stable", mesh.compute_matrices_stable))
    if config.test_stable_fem:
        methods_to_test.append(("Stable FEM", mesh.compute_matrices_stable_fem))
    
    logger.info(f"Testing {len(methods_to_test)} matrix construction methods")
    
    # Test each method
    for method_name, method_func in methods_to_test:
        logger.info(f"\nTesting {method_name} method...")
        try:
            start_time = time.time()
            M, K = method_func()
            compute_time = time.time() - start_time
            
            logger.info(f"{method_name} method completed in {compute_time:.3f} seconds")
            
            # Analyze properties
            M_eig, K_eig = analyze_matrix_properties(M, K, method_name, logger)
            results[method_name] = (M_eig, K_eig)
            matrix_data[method_name] = (M, K)
            
        except Exception as e:
            logger.error(f"Error testing {method_name} method: {e}")
            continue
    
    # Compare matrices if we have at least 2 methods
    if len(matrix_data) >= 2:
        logger.info(f"\n{'='*20} Matrix Comparison {'='*20}")
        methods = list(matrix_data.keys())
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                M1, K1 = matrix_data[method1]
                M2, K2 = matrix_data[method2]
                
                M_diff = np.max(np.abs(M1.toarray() - M2.toarray()))
                K_diff = np.max(np.abs(K1.toarray() - K2.toarray()))
                
                logger.info(f"Maximum difference in mass matrices ({method1} vs {method2}): {M_diff:.2e}")
                logger.info(f"Maximum difference in stiffness matrices ({method1} vs {method2}): {K_diff:.2e}")
    
    # Plot eigenvalue comparisons
    if len(results) > 1:
        plot_path = os.path.join(outdir, 'eigenvalue_comparison.png')
        plot_eigenvalue_comparison(results, plot_path)
        logger.info(f"Eigenvalue comparison plot saved to: {plot_path}")
    
    # Save results summary
    summary_path = os.path.join(outdir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Matrix Construction Test Summary\n")
        f.write(f"================================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Mesh parameters: n_theta={config.n_theta}, n_phi={config.n_phi}, R={config.R}, r={config.r}\n\n")
        
        for method_name, (M_eig, K_eig) in results.items():
            f.write(f"{method_name} Method:\n")
            f.write(f"  Mass matrix - Min eigenvalue: {np.min(M_eig):.2e}\n")
            f.write(f"  Mass matrix - Max eigenvalue: {np.max(M_eig):.2e}\n")
            f.write(f"  Stiffness matrix - Min eigenvalue: {np.min(K_eig):.2e}\n")
            f.write(f"  Stiffness matrix - Max eigenvalue: {np.max(K_eig):.2e}\n\n")
    
    logger.info(f"Test completed. Results saved in: {outdir}")
    return results, outdir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test and compare different matrix construction methods')
    parser.add_argument('--input', type=str, help='Path to input YAML file with parameters')
    args = parser.parse_args()

    # Load parameters
    if args.input:
        print(f"Loading parameters from {args.input}")
        with open(args.input, 'r') as f:
            params = yaml.safe_load(f)
        config = Config(params)
    else:
        print("Using default parameters")
        config = Config()
    
    # Run the test
    results, outdir = test_matrix_construction(config)
    print(f"\nMatrix construction test completed. Results saved in: {outdir}") 