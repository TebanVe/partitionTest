import numpy as np
from scipy.optimize import minimize

class PartitionOptimizer:
    def __init__(self, mesh, n_phases, epsilon, verbose=False):
        """
        Initialize the partition optimizer.
        
        Parameters:
        -----------
        mesh : TorusMesh
            The mesh to partition
        n_phases : int
            Number of partitions to create
        epsilon : float
            Interface width parameter
        verbose : bool, optional
            Whether to print progress information
        """
        self.mesh = mesh
        self.n_phases = n_phases
        self.epsilon = epsilon
        self.verbose = verbose
        self.A = None
        
        # Compute necessary matrices if not already computed
        if self.mesh.mass_matrix is None or self.mesh.stiffness_matrix is None:
            if self.verbose:
                print("Computing necessary matrices...")
            self.mesh.compute_matrices()
        
        # Compute target areas and mass matrix diagonal
        self.d = np.array(self.mesh.mass_matrix.diagonal()).flatten()
        self.total_area = np.sum(self.d)
        self.c = np.ones(self.n_phases) * (self.total_area / self.n_phases)
        
    def orthogonal_projection(self, A):
        """Project onto partition and area constraints."""
        n_vertices, n_phases = A.shape
        
        # Step 1: Compute row sum errors
        e = A - np.tile(1.0/n_phases, (n_vertices, n_phases))
        
        # Step 2: Compute column area errors
        f = np.zeros(n_phases)
        for j in range(n_phases):
            f[j] = np.dot(self.d, A[:, j]) - self.c[j]
        
        # Step 3: Construct constraint matrix C
        v = np.sqrt(self.d)  # From mass matrix
        v_norm_squared = np.sum(v**2)
        
        C = np.zeros((n_phases, n_phases))
        for k in range(n_phases):
            for l in range(n_phases):
                if k == l:
                    C[k, l] = v_norm_squared - v_norm_squared/n_phases
                else:
                    C[k, l] = -v_norm_squared/n_phases
        
        # Step 4: Compute q vector and solve for μ
        q = np.zeros(n_phases)
        for j in range(n_phases):
            q[j] = f[j] - np.dot(v, e[:, j])/n_phases
        
        # Remove last row and column to make C invertible
        C_reduced = C[:-1, :-1]
        q_reduced = q[:-1]
        
        mu_reduced = np.linalg.solve(C_reduced, q_reduced)
        mu = np.zeros(n_phases)
        mu[:-1] = mu_reduced
        
        # Step 5: Apply corrections
        A_orth = A - e + (1/n_phases) * np.outer(v, mu)
        
        return A_orth
    
    def objective(self, a_flat):
        """Compute the objective function for optimization."""
        a = a_flat.reshape(self.mesh.get_vertex_count(), self.n_phases)
        
        # Perimeter term (ε|∇u|²)
        perimeter = 0
        for i in range(self.n_phases):
            perimeter += self.epsilon * (a[:, i].T @ self.mesh.stiffness_matrix @ a[:, i])
        
        # Double-well term ((1/ε)W(u)) where W(u) = u²(1-u)²
        double_well = 0
        for i in range(self.n_phases):
            w = a[:, i]**2 * (1 - a[:, i])**2
            double_well += (1/self.epsilon) * np.dot(self.d, w)
        
        # Add penalty for constant functions
        penalty = 0
        for i in range(self.n_phases):
            std_i = np.sqrt(np.var(a[:, i]))
            target_std = np.sqrt(self.c[i]/self.total_area * (1 - self.c[i]/self.total_area))
            penalty += 5.0 * (std_i - target_std)**2
        
        return perimeter + double_well + penalty
    
    def gradient(self, a_flat):
        """Compute the gradient of the objective function."""
        a = a_flat.reshape(self.mesh.get_vertex_count(), self.n_phases)
        grad = np.zeros_like(a)
        
        # Perimeter gradient
        for i in range(self.n_phases):
            grad[:, i] += 2 * self.epsilon * (self.mesh.stiffness_matrix @ a[:, i])
        
        # Double-well gradient
        for i in range(self.n_phases):
            w_prime = 2*a[:, i] * (1-a[:, i])**2 - 2*a[:, i]**2 * (1-a[:, i])
            grad[:, i] += (1/self.epsilon) * self.d * w_prime
        
        # Penalty gradient
        for i in range(self.n_phases):
            std_i = np.sqrt(np.var(a[:, i]) + 1e-10)  # Avoid division by zero
            target_std = np.sqrt(self.c[i]/self.total_area * (1 - self.c[i]/self.total_area))
            dev = a[:, i] - np.mean(a[:, i])
            grad[:, i] += 10.0 * (std_i - target_std) * dev / (std_i * self.mesh.get_vertex_count())
        
        return grad.flatten()
    
    def optimize(self, max_iter=100):
        """Optimize the partitioning using gradient descent and L-BFGS."""
        if self.mesh.mass_matrix is None or self.mesh.stiffness_matrix is None:
            raise ValueError("Matrices must be computed before optimization")
            
        n_vertices = self.mesh.get_vertex_count()
        
        # Initialize density matrix with random values
        A = np.random.rand(n_vertices, self.n_phases)
        A = self.orthogonal_projection(A)
        
        # Multi-stage optimization with decreasing epsilon
        epsilons = [self.epsilon, self.epsilon/2, self.epsilon/4]
        iterations = [max_iter//3, max_iter//3, max_iter//3]
        
        for eps, iters in zip(epsilons, iterations):
            self.epsilon = eps
            
            for iter in range(iters):
                # L-BFGS optimization step
                result = minimize(self.objective, A.flatten(), method='L-BFGS-B', 
                                jac=self.gradient, bounds=[(0, 1)]*n_vertices*self.n_phases,
                                options={'maxiter': 1})
                
                # Update and project
                A = result.x.reshape(n_vertices, self.n_phases)
                A = self.orthogonal_projection(A)
                
                if self.verbose and iter % 10 == 0:
                    print(f"Epsilon: {self.epsilon:.6f}, Iteration: {iter}, Energy: {result.fun:.4f}")
        
        self.A = A
        return A 