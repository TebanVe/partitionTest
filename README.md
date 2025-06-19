# Manifold partition


This project implements and analyzes methods for partitioning a manifold into equal area regions (cells). Currently only implemented for the torus of revolution. It computes mass and stiffness matrices on triangulated torus manifolds, based on the paper "Partitions of Minimal Length on Manifolds" by Bogosel et al. It includes tools for visualizing the torus mesh, matrix analysis, and different optimization techniques for partition computation.

## Installation

1.  **Prerequisites**: 
    - Install `pyenv` to manage Python versions: [pyenv installation guide](https://github.com/pyenv/pyenv#installation)
    - Install the `pyenv-virtualenv` plugin: [pyenv-virtualenv installation guide](https://github.com/pyenv/pyenv-virtualenv#installation)
    - This project uses Python 3.9.7 as specified in the `.python-version` file

2.  **Set up Python Environment if Working Locally**: 
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

### Execution Environments

The code can be run in two different environments:

#### Local Development
For local development and testing, you can run the scripts directly. There are two ways to configure parameters:

1. **Direct Configuration** (Default):
   - The code uses default parameters from `src/config.py`
   - Run the script without any arguments:
   ```bash
   python examples/find_optimal_partition.py
   ```

2. **Using Input File** (For multiple simulations):
   - Create a YAML file with your parameters (see example below)
   - Use the `--input` flag to specify the file:
   ```bash
   python examples/find_optimal_partition.py --input your_parameters.yaml
   ```

   Example `parameters/input.yaml`:
   ```yaml
   n_partitions: 3
   n_theta: 8
   n_phi: 4
   R: 1.0
   r: 0.6
   lambda_penalty: 0.01
   max_iter: 15000
   tol: 1e-6
   refinement_levels: 2
   n_theta_increment: 2
   n_phi_increment: 1
   use_analytic: true
   starget: null
   seed: 42
   # Logging parameters
   log_frequency: 50  # How often to log optimization progress
   use_last_valid_iterate: true  # Whether to use last valid iterate on unsuccessful termination
   # Mesh refinement convergence criteria (used for logging and advanced control)
   refine_patience: 30
   refine_delta_energy: 1e-4
   refine_grad_tol: 1e-2
   refine_constraint_tol: 1e-2
   # Initial condition parameters
   use_custom_initial_condition: false  # Set to true to use a custom initial condition
   initial_condition_path: null  # Path to the .h5 file containing the initial condition
   allow_random_fallback: true  # Whether to allow random initialization as fallback when loading fails
   # Projection parameters for initial condition creation
   projection_max_iter: 100  # Maximum iterations for orthogonal projection algorithm
   ```

   **Note:**
   - Mesh refinement is performed at each level if `refinement_levels > 1`, with mesh resolution changes controlled by `n_theta_increment` and `n_phi_increment`.
   - Initial state logging is only performed for:
     - First level with random initialization
     - Levels where mesh resolution changes
   - The `use_last_valid_iterate` parameter controls whether to use the last valid point if optimization fails.
   - The mesh refinement criteria parameters (`refine_patience`, `refine_delta_energy`, etc.) are used for logging and advanced control.
   - The `allow_random_fallback` parameter controls whether to fall back to random initialization when loading or interpolating a custom initial condition fails. If set to `false`, the program will raise an error instead.
   - The `projection_max_iter` parameter controls the maximum number of iterations for the orthogonal projection algorithm used to create feasible initial conditions. This algorithm ensures that both partition constraints (row sums = 1) and area constraints (equal areas) are satisfied simultaneously.

   ### Initial Condition Creation with Orthogonal Projection

   The project now uses the iterative orthogonal projection algorithm from the paper "Partitions of Minimal Length on Manifolds" to create feasible initial conditions. This algorithm:

   - **Ensures feasibility**: Creates initial conditions that satisfy both partition and area constraints
   - **Improves convergence**: Starting from feasible points helps SLSQP optimization converge faster
   - **Reduces local minima**: Better initial conditions reduce the likelihood of getting stuck in poor local minima
   - **Mathematically rigorous**: Based on orthogonal projection theory

   The `projection_max_iter` parameter controls the maximum number of iterations for this algorithm. A value of 100 is typically sufficient, but you can adjust it if needed:
   - **Lower values (50-100)**: Faster initialization, suitable for most cases
   - **Higher values (200-500)**: More precise constraint satisfaction, useful for difficult cases

   ### Using External Solution as Initial Condition

   You can start the optimization from a previously computed solution by:

   1. Setting the following parameters in your YAML file:
   ```yaml
   # Initial condition parameters
   use_custom_initial_condition: true  # Enable loading from external file
   initial_condition_path: "path/to/your/solution.h5"  # Path to the solution file
   allow_random_fallback: false  # Optional: disable random fallback to ensure interpolation is used
   ```

   2. Or using the command line argument:
   ```bash
   python examples/find_optimal_partition.py --input parameters/input.yaml --initial-condition path/to/your/solution.h5
   ```

   The solution file should be an HDF5 file (.h5) containing:
   - `x_opt`: The solution vector
   - `vertices`: The mesh vertices used to compute the solution

   **Note:**
   - If the mesh resolution differs from the loaded solution, the solution will be interpolated to the new mesh
   - The initial state logging will be skipped when using a custom initial condition
   - The solution file should be compatible with the current number of partitions
   - If `allow_random_fallback` is set to `false`, the program will raise an error if loading or interpolating the solution fails, instead of falling back to random initialization

   Additional runtime options:
   - `--solution-dir`: Directory for storing solution files (optional)

#### Cluster Execution (UPPMAX)
Python version and necessary modules will be loaded with the submission script.

### Cluster Environment Setup

Before submitting jobs on UPPMAX, you need to set up your Python environment:

1. Create a virtual environment:
```bash
# Load Python module
module load python/3.9.5

# Create virtual environment in your home directory
python -m venv ~/partition

# Activate the environment
source ~/partition/bin/activate

# Install required packages
pip install -r requirements.txt
```

2. The submission script will automatically activate this environment, but you can specify a different one:
```bash
./scripts/submit.sh \
    --input parameters/input.yaml \
    --venv my_environment  # Optional: specify different environment
```

**Note:** The virtual environment is required for visualization tools that use PyVista. If you don't need visualization, you can run without a virtual environment.

For running on the UPPMAX cluster, use the provided SLURM submission script. The script supports both default parameters and custom input files:

```bash
# Basic run with default parameters from config.py
./scripts/submit.sh

The submission script provides the following options:
- `--input`: Path to input YAML file (default: parameters/input.yaml)
- `--output`: Directory for output files (default: results)
- `--solution-dir`: Directory for solution files (default: /proj/snic2020-15-36/private/LINKED_LST_MANIFOLD/PART_SOLUTION)
- `--time`: Time limit for the job (default: 12:00:00)
- `--initial-condition`: Path to HDF5 file containing initial condition (optional)
- `--venv`: Name of virtual environment to activate (default: partition)

**Note:** All mesh and optimization parameters (including mesh increments) are set in the YAML file. You do not pass mesh increments as command-line arguments.

The script will:
1. Submit your main job
2. Show you the job names and output file locations

Monitor your jobs using:
```bash
squeue -u $USER
```

Check results in:
- Main output: `results/job_logs/{job_name}/{job_name}.out`
- Main errors: `results/job_logs/{job_name}/{job_name}.err`
- Execution time and metadata: See the `results/metadata.yaml` file in the corresponding results directory.

**Important:**  
- When running on the cluster, it is important to provide the `--solution-dir` argument (or use the default set in the submission script).
- The solution file (e.g., `part<N>_nt<NTHETAINFO>_np<NPHIINFO>_lam<LAMBDA>_seed<SEED>_<TIMESTAMP>.h5`) will be saved in this directory, with a name that encodes the number of partitions, mesh parameters, lambda, seed, and timestamp (see find_optimal_partition.py and submit.sh for details).
- If `--solution-dir` is not provided, the default directory set in the submission script will be used.
- When running locally, the solution file is saved in a timestamped subdirectory of `results/`.

### Optimization Methods

The project implements two main optimization approaches:

#### SLSQP Optimization
The SLSQP (Sequential Least Squares Programming) optimizer supports both gradient computation methods:
- Finite-difference gradients (default)
- Analytic gradients (enabled by setting `use_analytic: true` in YAML)

Features:
- Flexible constraint handling (partition, area, non-negativity)
- Detailed logging and progress reporting
- Automatic plotting of optimization metrics
- Support for both local and cluster execution
- Easy switching between gradient computation methods

#### Logging and Progress Tracking
The SLSQP optimizer includes detailed logging features:
- Configurable logging frequency (`log_frequency`)
- Option to use last valid iterate on unsuccessful termination (`use_last_valid_iterate`)
- Initial state logging for new meshes and random initialization
- Detailed progress tracking including:
  - Energy values
  - Gradient norms
  - Constraint violations
  - Step sizes
  - Area evolution for each partition

Example usage:
```bash
# Using finite-difference gradients (default)
python examples/find_optimal_partition.py

# Using analytic gradients
python examples/find_optimal_partition.py --input parameters/input.yaml
```

#### L-BFGS Optimization
The L-BFGS optimizer with projection methods:
- Multiple projection methods
- Detailed optimization logging
- Support for multiple random starts
- Constraint handling for partitions

### Matrix Analysis and Visualization
The project includes tools for matrix analysis and visualization:

- **Visualize the Torus Mesh**: Shows the 3D triangulation of the generated torus.
    ```bash
    python -m examples.mesh_visualization
    ```

- **Test Matrix Construction Methods**: Computes mass and stiffness matrices using different methods.
    ```bash
    python -m examples.test_matrix_construction
    ```

## Visualization of Partitions and Mesh

You can visualize the torus mesh and overlay partition contours using the new script:

### Visualize Torus Mesh Only
```bash
python examples/torus_visualization.py --input parameters/input.yaml
```

### Visualize Partition Contours from a Solution File
```bash
python examples/torus_visualization.py --input parameters/input.yaml --solution results/<run_dir> --output-dir visualizations
```
- Replace `<run_dir>` with the directory containing your solution `.h5` file.
- The script will save a visualization image in the specified output directory.

## Public API

The following classes and functions are now available directly from the `src` package:

```python
from src import ContourAnalyzer, plot_torus_with_contours_pyvista
```

## Updated Project Structure

### Core Components
-   `src/`: Contains the core implementation.
    -   `mesh.py`: `TorusMesh` class for generating torus meshes
    -   `visualization.py`: `PartitionVisualizer` class for visualizations
    -   `lbfgs_optimizer.py`: L-BFGS optimization implementation
    -   `slsqp_optimizer.py`: SLSQP optimization implementation
    -   `projection_iterative.py`: Iterative projection methods
    -   `find_contours.py`: `ContourAnalyzer` for extracting and analyzing partition contours
    -   `plot_utils.py`: Visualization utilities using PyVista

### Examples and Scripts
-   `examples/`: Contains example scripts and optimization implementations
    -   `torus_visualization.py`: Visualize torus mesh and partition contours
    -   `mesh_visualization.py`: Mesh visualization tools
    -   `test_matrix_construction.py`: Matrix method comparisons
    -   `find_optimal_partition.py`: Main script for finding optimal partitions
-   `scripts/`: Contains utility scripts
    -   `submit.sh`: SLURM job submission script for cluster execution

### Configuration
-   `parameters/`: Contains configuration files
    -   `input.yaml`: Default input parameters
-   `results/`: Directory for output files and results

## License

MIT License 