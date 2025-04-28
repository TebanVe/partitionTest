import numpy as np
from typing import Tuple

def orthogonal_projection_direct(A: np.ndarray, c: np.ndarray, d: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Implements the orthogonal projection algorithm for partition and area constraints
    as described in the paper "Partitions of Minimal Length on Manifolds".
    This is the direct, non-iterative version.
    
    Args:
        A: Matrix of size N x n containing the density functions
        c: Vector of size n containing the target column sums (usually ones)
        d: Vector of size n containing the target area constraints
        v: Vector of size N containing the sum of mass matrix columns (v = 1ᵀM)
        
    Returns:
        The orthogonally projected matrix A that satisfies the constraints
    """
    N, n = A.shape
    A = A.copy()  # Make a copy to avoid modifying the input
    
    # Step 1: Calculate line sum error (N x 1 column vector)
    e = np.sum(A, axis=1) - np.ones(N)  # Each row should sum to 1
    
    # Step 2: Calculate column scalar product error (n x 1 column vector)
    #f = np.sum(v[:, np.newaxis] * A, axis=0) - d
    f = v @ A - d
    
    # Step 3: Define matrix C of size n x n
    v_norm_squared = np.sum(v**2)
    C = np.full((n, n), -v_norm_squared/n)
    np.fill_diagonal(C, v_norm_squared - v_norm_squared/n)
    
    # Step 4: Calculate q vector
    q = f - np.dot(v, e)/n
    
    # Step 5: Solve for lambda
    lambda_vec = np.zeros(n)
    lambda_vec[:-1] = np.linalg.solve(C[:-1, :-1], q[:-1])
    
    # Step 6: Calculate S
    S = np.sum(lambda_vec)
    
    # Step 7: Calculate eta vector
    eta = (e - S * v)/n
    
    # Step 8: Calculate orthogonal correction
    A_orth = np.outer(eta, np.ones(n)) + np.outer(v, lambda_vec)
    
    # Step 9: Apply correction
    A = A - A_orth
    
    # Step 10: Ensure non-negativity
    A = np.maximum(A, 0)
    
    # Step 11: Normalize rows to ensure partition constraint
    row_sums = np.sum(A, axis=1)
    mask = row_sums > 0  # Avoid division by zero
    A[mask] = A[mask] / row_sums[mask, np.newaxis]
    A[~mask] = 1.0/n  # Set uniform distribution for zero rows
    
    return A 