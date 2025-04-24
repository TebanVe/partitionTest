import numpy as np
import matplotlib.pyplot as plt
from src.mesh import TorusMesh

def analyze_matrix_properties(M, K, method_name):
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
    """
    print(f"\n{'-'*20} {method_name} Analysis {'-'*20}")
    
    # Convert to dense for analysis
    M_array = M.toarray()
    K_array = K.toarray()
    
    # Basic properties
    print("\nMatrix Properties:")
    print(f"Mass matrix shape: {M.shape}")
    print(f"Mass matrix non-zero elements: {M.nnz}")
    print(f"Stiffness matrix non-zero elements: {K.nnz}")
    
    # Symmetry check
    M_symmetric = np.allclose(M_array, M_array.T)
    K_symmetric = np.allclose(K_array, K_array.T)
    print(f"\nSymmetry Check:")
    print(f"Mass matrix is symmetric: {M_symmetric}")
    print(f"Stiffness matrix is symmetric: {K_symmetric}")
    
    # Numerical issues check
    print("\nNumerical Issues Check:")
    print(f"Mass matrix contains NaN: {np.isnan(M_array).any()}")
    print(f"Mass matrix contains Inf: {np.isinf(M_array).any()}")
    print(f"Stiffness matrix contains NaN: {np.isnan(K_array).any()}")
    print(f"Stiffness matrix contains Inf: {np.isinf(K_array).any()}")
    
    # Condition numbers
    M_cond = np.linalg.cond(M_array)
    K_cond = np.linalg.cond(K_array)
    print(f"\nCondition Numbers:")
    print(f"Mass matrix: {M_cond:.2e}")
    print(f"Stiffness matrix: {K_cond:.2e}")
    
    # Eigenvalue analysis
    M_eigenvals = np.linalg.eigvals(M_array)
    K_eigenvals = np.linalg.eigvals(K_array)
    
    print("\nMass Matrix Eigenvalue Analysis:")
    print(f"Minimum eigenvalue: {np.min(M_eigenvals):.2e}")
    print(f"Maximum eigenvalue: {np.max(M_eigenvals):.2e}")
    print(f"Number of negative eigenvalues: {np.sum(M_eigenvals < 0)}")
    print(f"Number of zero eigenvalues (< 1e-10): {np.sum(np.abs(M_eigenvals) < 1e-10)}")
    
    print("\nStiffness Matrix Eigenvalue Analysis:")
    print(f"Minimum eigenvalue: {np.min(K_eigenvals):.2e}")
    print(f"Maximum eigenvalue: {np.max(K_eigenvals):.2e}")
    print(f"Number of negative eigenvalues: {np.sum(K_eigenvals < 0)}")
    print(f"Number of zero eigenvalues (< 1e-10): {np.sum(np.abs(K_eigenvals) < 1e-10)}")
    
    return M_eigenvals, K_eigenvals

def plot_eigenvalue_comparison(M_eig_bary, K_eig_bary, M_eig_mani, K_eig_mani, M_eig_stable, K_eig_stable, M_eig_stable_fem, K_eig_stable_fem):
    """Plot eigenvalue comparisons between methods."""
    plt.figure(figsize=(15, 10))
    
    # Mass matrix eigenvalues
    plt.subplot(221)
    plt.semilogy(np.sort(np.abs(M_eig_bary)), 'b.', label='Barycentric', markersize=2)
    plt.semilogy(np.sort(np.abs(M_eig_mani)), 'r.', label='Manifold', markersize=2)
    plt.semilogy(np.sort(np.abs(M_eig_stable)), 'g.', label='Stable', markersize=2)
    plt.semilogy(np.sort(np.abs(M_eig_stable_fem)), 'y.', label='Stable FEM', markersize=2)
    plt.title('Mass Matrix Eigenvalue Distribution')
    plt.xlabel('Index')
    plt.ylabel('|Eigenvalue|')
    plt.grid(True)
    plt.legend()
    
    # Stiffness matrix eigenvalues
    plt.subplot(222)
    plt.semilogy(np.sort(np.abs(K_eig_bary)), 'b.', label='Barycentric', markersize=2)
    plt.semilogy(np.sort(np.abs(K_eig_mani)), 'r.', label='Manifold', markersize=2)
    plt.semilogy(np.sort(np.abs(K_eig_stable)), 'g.', label='Stable', markersize=2)
    plt.semilogy(np.sort(np.abs(K_eig_stable_fem)), 'y.', label='Stable FEM', markersize=2)
    plt.title('Stiffness Matrix Eigenvalue Distribution')
    plt.xlabel('Index')
    plt.ylabel('|Eigenvalue|')
    plt.grid(True)
    plt.legend()
    
    # Histogram of mass matrix eigenvalues
    plt.subplot(223)
    plt.hist(M_eig_bary, bins=50, alpha=0.3, label='Barycentric')
    plt.hist(M_eig_mani, bins=50, alpha=0.3, label='Manifold')
    plt.hist(M_eig_stable, bins=50, alpha=0.3, label='Stable')
    plt.hist(M_eig_stable_fem, bins=50, alpha=0.3, label='Stable FEM')
    plt.title('Mass Matrix Eigenvalue Histogram')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Count')
    plt.legend()
    
    # Histogram of stiffness matrix eigenvalues
    plt.subplot(224)
    plt.hist(K_eig_bary, bins=50, alpha=0.3, label='Barycentric')
    plt.hist(K_eig_mani, bins=50, alpha=0.3, label='Manifold')
    plt.hist(K_eig_stable, bins=50, alpha=0.3, label='Stable')
    plt.hist(K_eig_stable_fem, bins=50, alpha=0.3, label='Stable FEM')
    plt.title('Stiffness Matrix Eigenvalue Histogram')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('eigenvalue_comparison.png')
    plt.close()

def test_matrix_construction():
    """
    Test and compare different matrix construction methods.
    """
    # Create a torus mesh with relatively small number of vertices for testing
    mesh = TorusMesh(n_theta=20, n_phi=15, R=2.0, r=0.5)
    
    # Test barycentric method
    print("\nTesting barycentric method...")
    M_bary, K_bary = mesh.compute_matrices_barycentric()
    M_eig_bary, K_eig_bary = analyze_matrix_properties(M_bary, K_bary, "Barycentric Method")
    
    # Test manifold method
    print("\nTesting manifold method...")
    M_mani, K_mani = mesh.compute_manifold_matrices()
    M_eig_mani, K_eig_mani = analyze_matrix_properties(M_mani, K_mani, "Manifold Method")
    
    # Test stable method
    print("\nTesting stable method...")
    M_stable, K_stable = mesh.compute_matrices_stable()
    M_eig_stable, K_eig_stable = analyze_matrix_properties(M_stable, K_stable, "Stable Method")
    
    # Test stable FEM method
    print("\nTesting stable FEM method...")
    M_stable_fem, K_stable_fem = mesh.compute_matrices_stable_fem()
    M_eig_stable_fem, K_eig_stable_fem = analyze_matrix_properties(M_stable_fem, K_stable_fem, "Stable FEM Method")
    
    # Compare the matrices
    print("\nMatrix Comparison:")
    M_diff_bary_mani = np.max(np.abs(M_bary.toarray() - M_mani.toarray()))
    K_diff_bary_mani = np.max(np.abs(K_bary.toarray() - K_mani.toarray()))
    M_diff_bary_stable = np.max(np.abs(M_bary.toarray() - M_stable.toarray()))
    K_diff_bary_stable = np.max(np.abs(K_bary.toarray() - K_stable.toarray()))
    M_diff_bary_stable_fem = np.max(np.abs(M_bary.toarray() - M_stable_fem.toarray()))
    K_diff_bary_stable_fem = np.max(np.abs(K_bary.toarray() - K_stable_fem.toarray()))
    
    print(f"Maximum difference in mass matrices (Barycentric vs Manifold): {M_diff_bary_mani:.2e}")
    print(f"Maximum difference in stiffness matrices (Barycentric vs Manifold): {K_diff_bary_mani:.2e}")
    print(f"Maximum difference in mass matrices (Barycentric vs Stable): {M_diff_bary_stable:.2e}")
    print(f"Maximum difference in stiffness matrices (Barycentric vs Stable): {K_diff_bary_stable:.2e}")
    print(f"Maximum difference in mass matrices (Barycentric vs Stable FEM): {M_diff_bary_stable_fem:.2e}")
    print(f"Maximum difference in stiffness matrices (Barycentric vs Stable FEM): {K_diff_bary_stable_fem:.2e}")
    
    # Plot eigenvalue comparisons
    plot_eigenvalue_comparison(M_eig_bary, K_eig_bary, M_eig_mani, K_eig_mani, 
                             M_eig_stable, K_eig_stable, M_eig_stable_fem, K_eig_stable_fem)
    
    print("\nTest completed. Comparison plots saved as 'eigenvalue_comparison.png'")

if __name__ == "__main__":
    test_matrix_construction() 