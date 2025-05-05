import numpy as np
import scipy.sparse as sparse

class TorusMesh:
    def __init__(self, n_theta, n_phi, R, r):
        """
        Initialize a torus mesh.
        
        Parameters:
        -----------
        n_theta : int
            Number of points in the major circle direction
        n_phi : int
            Number of points in the minor circle direction
        R : float
            Major radius of the torus
        r : float
            Minor radius of the torus
        """
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.R = R
        self.r = r
        
        # Generate vertices
        self.vertices = self._generate_vertices()
        
        # Generate triangles
        self.triangles = self._generate_triangles()
        
        # Compute triangle statistics
        self._compute_triangle_statistics()
        
        self.mass_matrix = None
        self.stiffness_matrix = None
        
    def _generate_vertices(self):
        """Generate the vertices of the torus mesh."""
        theta = np.linspace(0, 2*np.pi, self.n_theta, endpoint=False)
        phi = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        
        vertices = []
        for t in theta:
            for p in phi:
                x = (self.R + self.r * np.cos(p)) * np.cos(t)
                y = (self.R + self.r * np.cos(p)) * np.sin(t)
                z = self.r * np.sin(p)
                vertices.append([x, y, z])
                
        return np.array(vertices)
    
    def _generate_triangles(self):
        """Generate the triangles of the torus mesh."""
        triangles = []
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                # Get indices of current and next points in both directions
                current = i * self.n_phi + j
                next_theta = ((i + 1) % self.n_theta) * self.n_phi + j
                next_phi = i * self.n_phi + ((j + 1) % self.n_phi)
                next_both = ((i + 1) % self.n_theta) * self.n_phi + ((j + 1) % self.n_phi)
                
                # Add two triangles for each quad
                triangles.append([current, next_theta, next_phi])
                triangles.append([next_theta, next_both, next_phi])
                
        return np.array(triangles)
    
    def get_vertex_neighbors(self, vertex_idx):
        """
        Get the indices of neighboring vertices for a given vertex.
        
        Parameters:
        -----------
        vertex_idx : int
            Index of the vertex
            
        Returns:
        --------
        list
            List of neighboring vertex indices
        """
        i = vertex_idx // self.n_phi
        j = vertex_idx % self.n_phi
        
        neighbors = []
        # Add neighbors in theta direction
        neighbors.append(((i - 1) % self.n_theta) * self.n_phi + j)
        neighbors.append(((i + 1) % self.n_theta) * self.n_phi + j)
        # Add neighbors in phi direction
        neighbors.append(i * self.n_phi + ((j - 1) % self.n_phi))
        neighbors.append(i * self.n_phi + ((j + 1) % self.n_phi))
        
        return neighbors
    
    def compute_matrices(self):
        """Compute mass and stiffness matrices for FEM."""
        if self.vertices is None or self.triangles is None:
            raise ValueError("Mesh must be created before computing matrices")
            
        n_vertices = self.vertices.shape[0]
        M = sparse.lil_matrix((n_vertices, n_vertices))
        K = sparse.lil_matrix((n_vertices, n_vertices))
        
        for t in self.triangles:
            v1, v2, v3 = self.vertices[t]
            
            # Compute area using cross product
            edge1 = v2 - v1
            edge2 = v3 - v1
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            
            # Lumped mass matrix (diagonal approximation)
            M[t[0], t[0]] += area/3
            M[t[1], t[1]] += area/3
            M[t[2], t[2]] += area/3
            
            # Simplified stiffness matrix computation
            for i in range(3):
                for j in range(3):
                    if i == j:
                        K[t[i], t[j]] += area
                    else:
                        K[t[i], t[j]] -= area/2
        
        self.mass_matrix = M.tocsr()
        self.stiffness_matrix = K.tocsr()
        return self.mass_matrix, self.stiffness_matrix
    
    def compute_manifold_matrices(self):
        """
        Compute mass and stiffness matrices for FEM on the torus manifold.
        This implementation properly accounts for the manifold geometry and uses
        consistent mass matrix formulation.
        """
        if self.vertices is None or self.triangles is None:
            raise ValueError("Mesh must be created before computing matrices")
            
        n_vertices = self.vertices.shape[0]
        M = sparse.lil_matrix((n_vertices, n_vertices))
        K = sparse.lil_matrix((n_vertices, n_vertices))
        
        # For each triangle
        for t in self.triangles:
            v1, v2, v3 = self.vertices[t]
            
            # Compute triangle area
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            unit_normal = normal / np.linalg.norm(normal)
            
            # Calculate local basis for tangent space
            # First tangent vector along edge1
            tau1 = edge1 / np.linalg.norm(edge1)
            # Second tangent vector orthogonal to both tau1 and normal
            tau2 = np.cross(unit_normal, tau1)
            tau2 = tau2 / np.linalg.norm(tau2)
            
            # Compute local metric tensor
            g = np.zeros((2, 2))
            g[0,0] = np.dot(tau1, tau1)  # <tau1, tau1>
            g[0,1] = g[1,0] = np.dot(tau1, tau2)  # <tau1, tau2>
            g[1,1] = np.dot(tau2, tau2)  # <tau2, tau2>
            
            # Calculate gradients of barycentric coordinates
            grad_phi = np.zeros((3, 3))
            
            # Compute gradients of barycentric coordinates in the global frame
            # These formulas come from standard FEM theory
            grad_phi[0] = np.cross(v3 - v2, unit_normal) / (2 * area)
            grad_phi[1] = np.cross(v1 - v3, unit_normal) / (2 * area)
            grad_phi[2] = np.cross(v2 - v1, unit_normal) / (2 * area)
            
            # Project gradients to tangent space
            grad_phi_tau = np.zeros((3, 2))
            for i in range(3):
                grad_phi_tau[i, 0] = np.dot(grad_phi[i], tau1)
                grad_phi_tau[i, 1] = np.dot(grad_phi[i], tau2)
            
            # Consistent mass matrix (not lumped)
            # Standard FEM mass matrix for P1 elements
            local_mass = area / 12 * np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
            ])
            
            # Add to global mass matrix
            for i in range(3):
                for j in range(3):
                    M[t[i], t[j]] += local_mass[i, j]
            
            # Stiffness matrix computation using tangential gradients
            for i in range(3):
                for j in range(3):
                    # Compute dot product in tangent space using metric
                    dot_prod = np.dot(grad_phi_tau[i], np.dot(g, grad_phi_tau[j]))
                    K[t[i], t[j]] += area * dot_prod
        
        self.manifold_mass_matrix = M.tocsr()
        self.manifold_stiffness_matrix = K.tocsr()
        return self.manifold_mass_matrix, self.manifold_stiffness_matrix
    
    def compute_matrices_alternative(self):
        """
        Alternative implementation of mass and stiffness matrix computation
        using a more systematic approach with reference elements and local coordinates.
        """
        if self.vertices is None or self.triangles is None:
            raise ValueError("Mesh must be created before computing matrices")
            
        n_vertices = self.vertices.shape[0]
        M = sparse.lil_matrix((n_vertices, n_vertices))
        K = sparse.lil_matrix((n_vertices, n_vertices))
        
        def triangle_area(v1, v2, v3):
            """Calculate the area of a triangle in 3D space."""
            return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1))
        
        def compute_tangential_gradients(p1, p2, p3):
            """Compute the tangential gradients of basis functions on a triangle."""
            # Compute triangle normal
            edge1 = p2 - p1
            edge2 = p3 - p1
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            
            # Compute area
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            
            # Set up local coordinates in the triangle's plane
            basis1 = edge1 / np.linalg.norm(edge1)
            basis2 = np.cross(normal, basis1)
            basis2 = basis2 / np.linalg.norm(basis2)
            
            # Map triangle to reference element
            local_coords = np.zeros((3, 2))
            local_coords[0] = [0, 0]  # First vertex at origin
            local_coords[1] = [np.dot(edge1, basis1), np.dot(edge1, basis2)]
            local_coords[2] = [np.dot(edge2, basis1), np.dot(edge2, basis2)]
            
            # Compute gradients in local coordinates
            A = np.array([
                [1, local_coords[0, 0], local_coords[0, 1]],
                [1, local_coords[1, 0], local_coords[1, 1]],
                [1, local_coords[2, 0], local_coords[2, 1]]
            ])
            
            # Shape function values at vertices
            b1 = np.array([1, 0, 0])
            b2 = np.array([0, 1, 0])
            b3 = np.array([0, 0, 1])
            
            try:
                # Solve for the coefficients of the shape functions
                c1 = np.linalg.solve(A, b1)
                c2 = np.linalg.solve(A, b2)
                c3 = np.linalg.solve(A, b3)
                
                # The gradients in local coordinates
                local_grad1 = np.array([c1[1], c1[2]])
                local_grad2 = np.array([c2[1], c2[2]])
                local_grad3 = np.array([c3[1], c3[2]])
                
                # Convert to global coordinates
                global_grad1 = local_grad1[0] * basis1 + local_grad1[1] * basis2
                global_grad2 = local_grad2[0] * basis1 + local_grad2[1] * basis2
                global_grad3 = local_grad3[0] * basis1 + local_grad3[1] * basis2
                
                # Project onto the tangent plane
                grad1 = global_grad1 - np.dot(global_grad1, normal) * normal
                grad2 = global_grad2 - np.dot(global_grad2, normal) * normal
                grad3 = global_grad3 - np.dot(global_grad3, normal) * normal
                
            except np.linalg.LinAlgError:
                # Fall back to simplified computation for degenerate triangles
                grad1 = np.zeros(3)
                grad2 = np.zeros(3)
                grad3 = np.zeros(3)
            
            return [grad1, grad2, grad3], area
        
        # Main computation loop
        for t in self.triangles:
            v1, v2, v3 = self.vertices[t]
            
            # Compute mass matrix
            area = triangle_area(v1, v2, v3)
            local_mass = (area / 12) * np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
            ])
            
            # Add to global mass matrix
            for i, vi in enumerate(t):
                for j, vj in enumerate(t):
                    M[vi, vj] += local_mass[i, j]
            
            # Compute stiffness matrix
            gradients, area = compute_tangential_gradients(v1, v2, v3)
            
            # Construct local stiffness matrix
            K_local = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    K_local[i, j] = np.dot(gradients[i], gradients[j]) * area
            
            # Add to global stiffness matrix
            for i, vi in enumerate(t):
                for j, vj in enumerate(t):
                    K[vi, vj] += K_local[i, j]
        
        self.alternative_mass_matrix = M.tocsr()
        self.alternative_stiffness_matrix = K.tocsr()
        return self.alternative_mass_matrix, self.alternative_stiffness_matrix
    
    def compute_matrices_barycentric(self):
        """
        Compute mass and stiffness matrices using barycentric coordinates for gradient computation.
        This approach uses the natural barycentric coordinates of the triangle to compute
        the tangential gradients directly.
        """
        if self.vertices is None or self.triangles is None:
            raise ValueError("Mesh must be created before computing matrices")
            
        n_vertices = self.vertices.shape[0]
        M = sparse.lil_matrix((n_vertices, n_vertices))
        K = sparse.lil_matrix((n_vertices, n_vertices))
        
        def compute_tangential_gradients_barycentric(p1, p2, p3):
            """Compute tangential gradients using barycentric coordinates."""
            # Compute triangle normal and area
            edge1 = p2 - p1
            edge2 = p3 - p1
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            normal = normal / np.linalg.norm(normal)
            
            # Compute gradients in barycentric coordinates
            # These are the gradients of the barycentric coordinates (which are the P1 basis functions)
            grad_lambda1 = np.cross(p3 - p2, normal) / (2 * area)
            grad_lambda2 = np.cross(p1 - p3, normal) / (2 * area)
            grad_lambda3 = np.cross(p2 - p1, normal) / (2 * area)
            
            # Project onto tangent plane
            grad_lambda1 = grad_lambda1 - np.dot(grad_lambda1, normal) * normal
            grad_lambda2 = grad_lambda2 - np.dot(grad_lambda2, normal) * normal
            grad_lambda3 = grad_lambda3 - np.dot(grad_lambda3, normal) * normal
            
            return [grad_lambda1, grad_lambda2, grad_lambda3], area
        
        # Main computation loop
        for t in self.triangles:
            v1, v2, v3 = self.vertices[t]
            
            # Compute mass matrix (same as before)
            area = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1))
            local_mass = (area / 12) * np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
            ])
            
            # Add to global mass matrix
            for i, vi in enumerate(t):
                for j, vj in enumerate(t):
                    M[vi, vj] += local_mass[i, j]
            
            # Compute stiffness matrix using barycentric gradients
            gradients, area = compute_tangential_gradients_barycentric(v1, v2, v3)
            
            # Construct local stiffness matrix
            K_local = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    K_local[i, j] = np.dot(gradients[i], gradients[j]) * area
            
            # Add to global stiffness matrix
            for i, vi in enumerate(t):
                for j, vj in enumerate(t):
                    K[vi, vj] += K_local[i, j]
        
        self.barycentric_mass_matrix = M.tocsr()
        self.barycentric_stiffness_matrix = K.tocsr()
        return self.barycentric_mass_matrix, self.barycentric_stiffness_matrix
    
    def compute_matrices_reference(self):
        """
        Compute mass and stiffness matrices using the reference element approach.
        This implementation explicitly uses the standard reference element basis functions
        and transforms their gradients to the physical space.
        """
        if self.vertices is None or self.triangles is None:
            raise ValueError("Mesh must be created before computing matrices")
            
        n_vertices = self.vertices.shape[0]
        M = sparse.lil_matrix((n_vertices, n_vertices))
        K = sparse.lil_matrix((n_vertices, n_vertices))
        
        def compute_tangential_gradients_reference(p1, p2, p3):
            """Compute gradients using reference element transformation."""
            # Reference gradients (constant for P1 elements)
            ref_grad_phi1 = np.array([-1, -1])
            ref_grad_phi2 = np.array([1, 0])
            ref_grad_phi3 = np.array([0, 1])
            
            # Compute Jacobian of the transformation
            edge1 = p2 - p1
            edge2 = p3 - p1
            J = np.array([edge1, edge2]).T  # 3x2 Jacobian matrix
            
            # Compute normal and area
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            normal = normal / np.linalg.norm(normal)
            
            # Transform gradients to physical space
            grad_phi1 = J @ ref_grad_phi1
            grad_phi2 = J @ ref_grad_phi2
            grad_phi3 = J @ ref_grad_phi3
            
            # Project onto tangent plane
            grad_phi1 = grad_phi1 - np.dot(grad_phi1, normal) * normal
            grad_phi2 = grad_phi2 - np.dot(grad_phi2, normal) * normal
            grad_phi3 = grad_phi3 - np.dot(grad_phi3, normal) * normal
            
            return [grad_phi1, grad_phi2, grad_phi3], area
        
        # Main computation loop
        for t in self.triangles:
            v1, v2, v3 = self.vertices[t]
            
            # Compute mass matrix (same as before)
            area = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1))
            local_mass = (area / 12) * np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
            ])
            
            # Add to global mass matrix
            for i, vi in enumerate(t):
                for j, vj in enumerate(t):
                    M[vi, vj] += local_mass[i, j]
            
            # Compute stiffness matrix using reference element gradients
            gradients, area = compute_tangential_gradients_reference(v1, v2, v3)
            
            # Construct local stiffness matrix
            K_local = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    K_local[i, j] = np.dot(gradients[i], gradients[j]) * area
            
            # Add to global stiffness matrix
            for i, vi in enumerate(t):
                for j, vj in enumerate(t):
                    K[vi, vj] += K_local[i, j]
        
        self.reference_mass_matrix = M.tocsr()
        self.reference_stiffness_matrix = K.tocsr()
        return self.reference_mass_matrix, self.reference_stiffness_matrix
    
    def compute_matrices_stable(self):
        """
        Compute mass and stiffness matrices using a numerically stable approach.
        This implementation:
        1. Uses a more stable gradient computation
        2. Implements regularization to ensure positive semi-definiteness
        3. Uses a more robust metric tensor computation
        """
        if self.vertices is None or self.triangles is None:
            raise ValueError("Mesh must be created before computing matrices")
            
        n_vertices = self.vertices.shape[0]
        M = sparse.lil_matrix((n_vertices, n_vertices))
        K = sparse.lil_matrix((n_vertices, n_vertices))
        
        # Small regularization parameter
        epsilon = 1e-10
        
        def compute_stable_gradients(p1, p2, p3):
            """Compute gradients using a numerically stable approach."""
            # Compute triangle normal and area
            edge1 = p2 - p1
            edge2 = p3 - p1
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            
            # Normalize normal vector
            normal_norm = np.linalg.norm(normal)
            if normal_norm < epsilon:
                # Handle degenerate triangles
                return [np.zeros(3), np.zeros(3), np.zeros(3)], area
            
            normal = normal / normal_norm
            
            # Compute stable basis for tangent space
            # Use QR decomposition for better numerical stability
            basis = np.column_stack([edge1, edge2])
            Q, R = np.linalg.qr(basis)
            
            # Project normal to ensure orthogonality
            Q = Q - np.outer(normal, np.dot(normal, Q))
            
            # Normalize basis vectors
            Q = Q / np.linalg.norm(Q, axis=0)
            
            # Compute gradients in barycentric coordinates
            grad_lambda1 = np.cross(p3 - p2, normal) / (2 * area)
            grad_lambda2 = np.cross(p1 - p3, normal) / (2 * area)
            grad_lambda3 = np.cross(p2 - p1, normal) / (2 * area)
            
            # Project gradients to tangent space
            grad_lambda1 = grad_lambda1 - np.dot(grad_lambda1, normal) * normal
            grad_lambda2 = grad_lambda2 - np.dot(grad_lambda2, normal) * normal
            grad_lambda3 = grad_lambda3 - np.dot(grad_lambda3, normal) * normal
            
            # Add small regularization to ensure positive definiteness
            grad_lambda1 = grad_lambda1 + epsilon * Q[:, 0]
            grad_lambda2 = grad_lambda2 + epsilon * Q[:, 1]
            grad_lambda3 = grad_lambda3 + epsilon * Q[:, 0]
            
            return [grad_lambda1, grad_lambda2, grad_lambda3], area
        
        # Main computation loop
        for t in self.triangles:
            v1, v2, v3 = self.vertices[t]
            
            # Compute mass matrix (same as before)
            area = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1))
            local_mass = (area / 12) * np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
            ])
            
            # Add to global mass matrix
            for i, vi in enumerate(t):
                for j, vj in enumerate(t):
                    M[vi, vj] += local_mass[i, j]
            
            # Compute stiffness matrix using stable gradients
            gradients, area = compute_stable_gradients(v1, v2, v3)
            
            # Construct local stiffness matrix with regularization
            K_local = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    K_local[i, j] = np.dot(gradients[i], gradients[j]) * area
                    if i == j:
                        K_local[i, j] += epsilon * area  # Add regularization to diagonal
        
            # Add to global stiffness matrix
            for i, vi in enumerate(t):
                for j, vj in enumerate(t):
                    K[vi, vj] += K_local[i, j]
        
        self.stable_mass_matrix = M.tocsr()
        self.stable_stiffness_matrix = K.tocsr()
        return self.stable_mass_matrix, self.stable_stiffness_matrix
    
    def compute_matrices_stable_fem(self):
        """
        Compute mass and stiffness matrices using standard FEM basis functions with matrix-level regularization.
        This implementation:
        1. Uses standard FEM gradient computation
        2. Preserves partition of unity
        3. Adds regularization at the matrix level
        4. Maintains consistency with optimization
        """
        if self.vertices is None or self.triangles is None:
            raise ValueError("Mesh must be created before computing matrices")
            
        n_vertices = self.vertices.shape[0]
        M = sparse.lil_matrix((n_vertices, n_vertices))
        K = sparse.lil_matrix((n_vertices, n_vertices))
        
        # Small regularization parameter
        epsilon = 1e-10
        
        def compute_fem_gradients(p1, p2, p3):
            """Compute standard FEM gradients."""
            # Compute triangle normal and area
            edge1 = p2 - p1
            edge2 = p3 - p1
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            
            # Handle degenerate triangles
            if area < epsilon:
                return [np.zeros(3), np.zeros(3), np.zeros(3)], area
            
            normal = normal / np.linalg.norm(normal)
            
            # Standard FEM gradients
            grad_phi1 = np.cross(p3 - p2, normal) / (2 * area)
            grad_phi2 = np.cross(p1 - p3, normal) / (2 * area)
            grad_phi3 = np.cross(p2 - p1, normal) / (2 * area)
            
            # Project to tangent space (this is standard)
            grad_phi1 = grad_phi1 - np.dot(grad_phi1, normal) * normal
            grad_phi2 = grad_phi2 - np.dot(grad_phi2, normal) * normal
            grad_phi3 = grad_phi3 - np.dot(grad_phi3, normal) * normal
            
            return [grad_phi1, grad_phi2, grad_phi3], area
        
        # Main computation loop
        for t in self.triangles:
            v1, v2, v3 = self.vertices[t]
            
            # Compute mass matrix (standard)
            area = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1))
            local_mass = (area / 12) * np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
            ])
            
            # Add to global mass matrix
            for i, vi in enumerate(t):
                for j, vj in enumerate(t):
                    M[vi, vj] += local_mass[i, j]
            
            # Compute stiffness matrix using standard FEM gradients
            gradients, area = compute_fem_gradients(v1, v2, v3)
            
            # Construct local stiffness matrix
            K_local = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    K_local[i, j] = np.dot(gradients[i], gradients[j]) * area
            
            # Add regularization at the matrix level
            K_local += epsilon * area * np.eye(3)  # Add to diagonal only
            
            # Add to global stiffness matrix
            for i, vi in enumerate(t):
                for j, vj in enumerate(t):
                    K[vi, vj] += K_local[i, j]
        
        self.stable_fem_mass_matrix = M.tocsr()
        self.stable_fem_stiffness_matrix = K.tocsr()
        return self.stable_fem_mass_matrix, self.stable_fem_stiffness_matrix
    
    def get_vertex_count(self):
        """Return the number of vertices in the mesh."""
        return len(self.vertices) if self.vertices is not None else 0
    
    def get_triangle_count(self):
        """Return the number of triangles in the mesh."""
        return len(self.triangles) if self.triangles is not None else 0
    
    def _compute_triangle_statistics(self):
        """Compute and store statistics about the mesh triangles."""
        # Total number of triangles
        self.n_triangles = len(self.triangles)
        
        # Compute total surface area
        self.total_area = 4 * np.pi**2 * self.R * self.r
        
        # Compute average triangle area
        self.avg_triangle_area = self.total_area / self.n_triangles
        
        # Estimate average triangle side length
        # For a triangle of area A, the side length of an equilateral triangle would be:
        # a = sqrt(4A/sqrt(3))
        self.avg_triangle_side = np.sqrt(4 * self.avg_triangle_area / np.sqrt(3))
        
        print("\nMesh Statistics:")
        print(f"Number of vertices: {len(self.vertices)}")
        print(f"Number of triangles: {self.n_triangles}")
        print(f"Total surface area: {self.total_area:.6e}")
        print(f"Average triangle area: {self.avg_triangle_area:.6e}")
        print(f"Estimated average triangle side length: {self.avg_triangle_side:.6e}")
        
        # Store these values for easy access
        self.mesh_statistics = {
            'n_vertices': len(self.vertices),
            'n_triangles': self.n_triangles,
            'total_area': self.total_area,
            'avg_triangle_area': self.avg_triangle_area,
            'avg_triangle_side': self.avg_triangle_side
        } 