import numpy as np
from src.projection_direct import orthogonal_projection_direct
from src.projection_iterative import orthogonal_projection_iterative
from src.mesh import TorusMesh

def test_compare_projections():
    """Compare the direct and iterative projection algorithms"""
    
    # Create a torus mesh
    n_theta, n_phi = 50, 20  # Number of points in major and minor circle directions
    R, r = 1.0, 0.6  # Major and minor radii
    mesh = TorusMesh(n_theta, n_phi, R, r)
    
    # Compute mass matrix
    mass_matrix, _ = mesh.compute_matrices()
    
    # Compute v as sum of columns of mass matrix (as per paper)
    v = np.ones((1, mass_matrix.shape[0])) @ mass_matrix  # This computes 1_{1×N} · M
    v = v.flatten()  # Convert to 1D array
    
    # Compute actual total area from mass matrix
    total_area = np.sum(v)
    
    # Test case 1: Two partitions on torus
    n_partitions = 2
    N = mesh.get_vertex_count()
    
    # Generate initial density functions (random but normalized)
    A1 = np.random.rand(N, n_partitions)
    A1 = A1 / np.sum(A1, axis=1)[:, np.newaxis]
    
    # Target area constraints (equal areas)
    d1 = np.ones(n_partitions) * (total_area / n_partitions)
    
    # Run both versions
    result1_direct = orthogonal_projection_direct(A1, np.ones(n_partitions), d1, v)
    result1_iterative = orthogonal_projection_iterative(A1, np.ones(n_partitions), d1, v)
    
    print("\nTest Case 1 Results (2 Partitions on Torus):")
    print("Number of vertices:", N)
    print("Number of partitions:", n_partitions)
    print("Total mesh area:", total_area)
    print("Theoretical torus area:", 4 * np.pi**2 * R * r)
    
    print("\nDirect Version:")
    print("Area constraints (actual):", np.sum(v[:, np.newaxis] * result1_direct, axis=0))
    print("Area constraints (target):", d1)
    print("Row sums (min, max):", np.min(np.sum(result1_direct, axis=1)), np.max(np.sum(result1_direct, axis=1)))
    
    print("\nIterative Version:")
    print("Area constraints (actual):", np.sum(v[:, np.newaxis] * result1_iterative, axis=0))
    print("Area constraints (target):", d1)
    print("Row sums (min, max):", np.min(np.sum(result1_iterative, axis=1)), np.max(np.sum(result1_iterative, axis=1)))
    
    # Test case 2: Three partitions on torus
    n_partitions = 3
    A2 = np.random.rand(N, n_partitions)
    A2 = A2 / np.sum(A2, axis=1)[:, np.newaxis]
    d2 = np.ones(n_partitions) * (total_area / n_partitions)
    
    # Run both versions
    result2_direct = orthogonal_projection_direct(A2, np.ones(n_partitions), d2, v)
    result2_iterative = orthogonal_projection_iterative(A2, np.ones(n_partitions), d2, v)
    
    print("\nTest Case 2 Results (3 Partitions on Torus):")
    print("Number of vertices:", N)
    print("Number of partitions:", n_partitions)
    print("Total mesh area:", total_area)
    print("Theoretical torus area:", 4 * np.pi**2 * R * r)
    
    print("\nDirect Version:")
    print("Area constraints (actual):", np.sum(v[:, np.newaxis] * result2_direct, axis=0))
    print("Area constraints (target):", d2)
    print("Row sums (min, max):", np.min(np.sum(result2_direct, axis=1)), np.max(np.sum(result2_direct, axis=1)))
    
    print("\nIterative Version:")
    print("Area constraints (actual):", np.sum(v[:, np.newaxis] * result2_iterative, axis=0))
    print("Area constraints (target):", d2)
    print("Row sums (min, max):", np.min(np.sum(result2_iterative, axis=1)), np.max(np.sum(result2_iterative, axis=1)))
    
    # Test case 3: Four partitions with unequal areas
    n_partitions = 4
    A3 = np.random.rand(N, n_partitions)
    A3 = A3 / np.sum(A3, axis=1)[:, np.newaxis]
    
    # Define unequal area constraints (e.g., 0.4, 0.3, 0.2, 0.1 of total area)
    d3 = total_area * np.array([0.4, 0.3, 0.2, 0.1])
    
    # Run both versions
    result3_direct = orthogonal_projection_direct(A3, np.ones(n_partitions), d3, v)
    result3_iterative = orthogonal_projection_iterative(A3, np.ones(n_partitions), d3, v)
    
    print("\nTest Case 3 Results (4 Partitions with Unequal Areas):")
    print("Number of vertices:", N)
    print("Number of partitions:", n_partitions)
    print("Total mesh area:", total_area)
    print("Theoretical torus area:", 4 * np.pi**2 * R * r)
    
    print("\nDirect Version:")
    print("Area constraints (actual):", np.sum(v[:, np.newaxis] * result3_direct, axis=0))
    print("Area constraints (target):", d3)
    print("Row sums (min, max):", np.min(np.sum(result3_direct, axis=1)), np.max(np.sum(result3_direct, axis=1)))
    
    print("\nIterative Version:")
    print("Area constraints (actual):", np.sum(v[:, np.newaxis] * result3_iterative, axis=0))
    print("Area constraints (target):", d3)
    print("Row sums (min, max):", np.min(np.sum(result3_iterative, axis=1)), np.max(np.sum(result3_iterative, axis=1)))

if __name__ == "__main__":
    test_compare_projections() 