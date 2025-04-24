# Torus Partitioning and Matrix Analysis

This project implements and analyzes methods for computing mass and stiffness matrices on triangulated torus manifolds, based on the paper "Partitions of Minimal Length on Manifolds" by Bogosel et al. It also includes tools for visualizing the torus mesh and the results of matrix analysis.

## Installation

1.  **Prerequisites**: Ensure you have `pyenv` installed to manage Python versions. This project uses the Python version specified in the `.python-version` file.

2.  **Set up Python Environment**: Navigate to the project directory in your terminal. `pyenv` should automatically pick up the version from `.python-version`. If not, you might need to run `pyenv install $(cat .python-version)` first.

3.  **Install Dependencies**: It's recommended to create an environment specific to this project within `pyenv`. Once your Python version is set, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project includes several examples in the `examples/` directory:

-   **Visualize the Torus Mesh**: Shows the 3D triangulation of the generated torus.
    ```bash
    python -m examples.mesh_visualization
    ```

-   **Test Matrix Construction Methods**: Computes mass and stiffness matrices using different methods (Barycentric, Manifold, Stable, Stable FEM), analyzes their eigenvalue properties, and saves comparison plots (`eigenvalue_comparison.png`).
    ```bash
    python -m examples.test_matrix_construction
    ```
    *(Note: The `optimization_example.py` is present but currently excluded from the main repository state).*

## Project Structure

-   `src/`: Contains the core implementation.
    -   `mesh.py`: `TorusMesh` class for generating torus meshes and implementing various matrix computation methods.
    -   `optimization.py`: Contains the `PartitionOptimizer` class (based on the paper, requires matrix computation).
    -   `visualization.py`: `PartitionVisualizer` class for creating 3D and 2D visualizations (primarily for partition results).
-   `examples/`: Contains example scripts demonstrating project features.
    -   `mesh_visualization.py`: Shows the mesh triangulation.
    -   `test_matrix_construction.py`: Compares different matrix methods.
    -   `optimization_example.py`: (Not currently tracked in git) Demonstrates partition optimization.
-   `README.md`: This file.
-   `requirements.txt`: Project dependencies for `pip`.
-   `pyproject.toml`: Project configuration and build information.
-   `.python-version`: Specifies the Python version for `pyenv`.
-   `.gitignore`: Specifies intentionally untracked files for Git.

## License

MIT License 