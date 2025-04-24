import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.mesh import TorusMesh

def plot_torus_mesh(n_theta=40, n_phi=30, R=1.0, r=0.6):
    """
    Create and visualize a torus mesh with its triangulation.
    
    Parameters:
    -----------
    n_theta : int
        Number of points in the major circle direction
    n_phi : int
        Number of points in the minor circle direction
    R : float
        Major radius of the torus (distance from center to tube center)
    r : float
        Minor radius of the torus (radius of the tube)
    """
    # Create the torus mesh
    mesh = TorusMesh(n_theta, n_phi, R, r)
    
    # Get vertices and triangles
    vertices = mesh.vertices
    triangles = mesh.triangles
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the torus surface with triangulation
    surf = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          triangles=triangles, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Set labels and title
    ax.set_title('Torus Surface with Triangulation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Adjust the view angle for better visualization
    # elev is the elevation angle in degrees (0 is looking from the side)
    # azim is the azimuth angle in degrees (0 is looking from the front)
    ax.view_init(elev=20, azim=30)
    
    # Set the limits to ensure proper scaling
    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(), 
                         vertices[:, 1].max()-vertices[:, 1].min(), 
                         vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
    
    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
    
    # Print mesh statistics
    print(f"Number of vertices: {mesh.get_vertex_count()}")
    print(f"Number of triangles: {mesh.get_triangle_count()}")
    print(f"Vertex coordinates shape: {vertices.shape}")
    print(f"Triangles shape: {triangles.shape}")

if __name__ == "__main__":
    # Test with different parameters
    print("Testing torus mesh with default parameters:")
    plot_torus_mesh()
    
    # print("\nTesting torus mesh with finer resolution:")
    # plot_torus_mesh(n_theta=100, n_phi=100) 