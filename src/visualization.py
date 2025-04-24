import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PartitionVisualizer:
    def __init__(self, mesh, A):
        """
        Initialize the partition visualizer.
        
        Parameters:
        -----------
        mesh : TorusMesh
            The mesh to visualize
        A : numpy.ndarray
            The partition matrix
        """
        self.mesh = mesh
        self.A = A
        self.n_phases = A.shape[1]
        
    def plot_3d(self, ax=None):
        """Create a 3D visualization of the partitioned torus."""
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        
        # Get dominant phase at each vertex
        max_phase = np.argmax(self.A, axis=1)
        
        # Create colormap
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_phases))
        
        # Color triangles by majority phase
        for t in self.mesh.triangles:
            phases = [max_phase[t[0]], max_phase[t[1]], max_phase[t[2]]]
            majority = max(set(phases), key=phases.count)
            
            ax.plot_trisurf([self.mesh.vertices[t[0], 0], self.mesh.vertices[t[1], 0], self.mesh.vertices[t[2], 0]],
                           [self.mesh.vertices[t[0], 1], self.mesh.vertices[t[1], 1], self.mesh.vertices[t[2], 1]],
                           [self.mesh.vertices[t[0], 2], self.mesh.vertices[t[1], 2], self.mesh.vertices[t[2], 2]],
                           color=colors[majority], alpha=0.8)
        
        ax.set_title(f'Torus Partition ({self.n_phases} cells)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        return ax
    
    def plot_2d(self, ax=None):
        """Create a 2D flattened visualization of the partitioned torus."""
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
        
        # Get dominant phase at each vertex
        max_phase = np.argmax(self.A, axis=1)
        
        # Create 2D grid representation
        flattened = np.zeros((self.mesh.n_theta, self.mesh.n_phi))
        for i in range(self.mesh.n_theta):
            for j in range(self.mesh.n_phi):
                idx = i * self.mesh.n_phi + j
                flattened[i, j] = max_phase[idx]
        
        # Plot with periodic boundary visualization
        extended = np.block([[flattened, flattened], [flattened, flattened]])
        
        ax.imshow(extended, cmap='tab10', interpolation='nearest')
        ax.set_title('Flattened Torus with Periodic Boundary')
        
        return ax
    
    def plot_both(self):
        """Create both 3D and 2D visualizations side by side."""
        fig = plt.figure(figsize=(15, 6))
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        self.plot_3d(ax1)
        
        # 2D plot
        ax2 = fig.add_subplot(122)
        self.plot_2d(ax2)
        
        plt.tight_layout()
        return fig 