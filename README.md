# Torus Partitioning and Matrix Analysis

This project implements and analyzes methods for computing mass and stiffness matrices on triangulated torus manifolds, based on the paper "Partitions of Minimal Length on Manifolds" by Bogosel et al. It includes tools for visualizing the torus mesh, matrix analysis, and advanced optimization techniques for partition computation.

## Installation

1.  **Prerequisites**: 
    - Install `pyenv` to manage Python versions: [pyenv installation guide](https://github.com/pyenv/pyenv#installation)
    - Install the `pyenv-virtualenv` plugin: [pyenv-virtualenv installation guide](https://github.com/pyenv/pyenv-virtualenv#installation)
    - This project uses Python 3.9.7 as specified in the `.python-version` file

2.  **Set up Python Environment**: 
    ```bash
    # Install the required Python version if you don't have it
    pyenv install 3.9.7
    
    # Create a virtualenv named 'partition' using Python 3.9.7
    pyenv virtualenv 3.9.7 partition
    
    # Navigate to the project directory
    cd /path/to/project
    
    # The .python-version file will automatically activate the 'partition' environment
    # You should see '(partition)' in your prompt
    ```

3.  **Install Dependencies**: 
    ```bash
    # With the 'partition' environment active, install dependencies
    pip install -r requirements.txt
    ```

## Usage

### Matrix Analysis and Visualization
The project includes several examples in the `examples/` directory:

-   **Visualize the Torus Mesh**: Shows the 3D triangulation of the generated torus.
    ```bash
    python -m examples.mesh_visualization
    ```

-   **Test Matrix Construction Methods**: Computes mass and stiffness matrices using different methods (Barycentric, Manifold, Stable, Stable FEM), analyzes their eigenvalue properties, and saves comparison plots (`eigenvalue_comparison.png`).
    ```bash
    python -m examples.test_matrix_construction
    ```

### Partition Optimization
New optimization capabilities have been added:

-   **Multiple Start Optimization**: Tests the partition optimization with different random initializations:
    ```bash
    python examples/test_torus_optimization_multiple.py
    ```

## Project Structure

### Core Components
-   `src/`: Contains the core implementation.
    -   `mesh.py`: `TorusMesh` class for generating torus meshes and implementing various matrix computation methods.
    -   `optimization.py`: Contains the `PartitionOptimizer` class (based on the paper, requires matrix computation).
    -   `visualization.py`: `PartitionVisualizer` class for creating 3D and 2D visualizations.
    -   `lbfgs_optimizer.py`: L-BFGS optimization with projection methods and detailed logging.
    -   `projection_iterative.py`: Iterative projection onto partition constraints.

### Examples and Tests
-   `examples/`: Contains example scripts demonstrating project features.
    -   `mesh_visualization.py`: Shows the mesh triangulation.
    -   `test_matrix_construction.py`: Compares different matrix methods.
    -   `test_torus_optimization_multiple.py`: Tests partition optimization with multiple starts.

## Features

### Matrix Analysis
- Multiple matrix construction methods (Barycentric, Manifold, Stable, Stable FEM)
- Eigenvalue analysis and comparison
- Visualization of matrix properties

### Mesh Generation and Visualization
- Torus mesh generation with configurable parameters
- 3D visualization of mesh structure
- 2D parameter space visualization

### Optimization and Partitioning
- L-BFGS optimization with constraint handling
- Multiple projection methods
- Detailed optimization logging including:
  - Energy tracking
  - Gradient norm monitoring
  - Projection distance tracking
  - Constraint violation monitoring
- Support for multiple random starts
- Constraint handling:
  - Partition constraints (sum to 1)
  - Area preservation
  - Non-negativity constraints

## Configuration and Dependencies
-   `requirements.txt`: Project dependencies for `pip`.
-   `pyproject.toml`: Project configuration and build information.
-   `.python-version`: Specifies Python version (3.9.7) and environment name.
-   `.gitignore`: Specifies intentionally untracked files.

## License

MIT License 