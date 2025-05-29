# Manifold partition


This project implements and analyzes methods for partitioning a manifold into equal area regions (cells). Currently only implemented for the torus of revolution. It computes mass and stiffness matrices on triangulated torus manifolds, based on the paper "Partitions of Minimal Length on Manifolds" by Bogosel et al. It includes tools for visualizing the torus mesh, matrix analysis, and advanced optimization techniques for partition computation.

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
   ```

   **Note:** All mesh and optimization parameters (including mesh increments) are now set in the YAML file. You do not pass mesh increments as command-line arguments.

   Additional runtime options:
   - `--refinement-levels`: Override the number of refinement levels (optional)
   - `--solution-dir`: Directory for storing solution files (optional)
   - `--analytic`: Use analytic gradients (flag, optional)

#### Cluster Execution (UPPMAX)
Python version and necessary modules will be loaded with the submission script.

For running on the UPPMAX cluster, use the provided SLURM submission script. The script supports both default parameters and custom input files:

```bash
# Basic run with default parameters from config.py
./scripts/submit.sh

# Run with custom parameters file (useful for running multiple simulations)
./scripts/submit.sh \
    --input parameters/input.yaml \
    --output results \
    --refinement-levels 2 \
    --solution-dir /proj/snic2020-15-36/private/LINKED_LST_MANIFOLD/PART_SOLUTION \
    --time "12:00:00" \
    --analytic
```

The submission script provides the following options:
- `--input`: Path to input YAML file (default: parameters/input.yaml)
- `--output`: Directory for output files (default: results)
- `--refinement-levels`: Number of refinement levels (overrides YAML, optional)
- `--solution-dir`: Directory for solution files (default: /proj/snic2020-15-36/private/LINKED_LST_MANIFOLD/PART_SOLUTION)
- `--time`: Time limit for the job (default: 12:00:00)
- `--analytic`: Use analytic gradients (flag, optional)

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
- Execution time and metadata: See the `metadata.yaml` file in the corresponding results directory.

**Important:**  
- When running on the cluster, it is important to provide the `--solution-dir` argument (or use the default set in the submission script).
- The solution file (`solution.h5`) will be saved in this directory, which should be a project directory with sufficient storage and write permissions (e.g., `/proj/snic2020-15-36/private/LINKED_LST_MANIFOLD/PART_SOLUTION`).
- If `--solution-dir` is not provided, the default directory set in the submission script will be used.
- When running locally, the solution file is saved in a timestamped subdirectory of `results/`.

### Optimization Methods

The project implements two main optimization approaches:

#### SLSQP Optimization
The SLSQP (Sequential Least Squares Programming) optimizer supports both gradient computation methods:
- Finite-difference gradients (default)
- Analytic gradients (enabled with `--analytic` flag or `use_analytic: true` in YAML)

Features:
- Flexible constraint handling (partition, area, non-negativity)
- Detailed logging and progress reporting
- Automatic plotting of optimization metrics
- Support for both local and cluster execution
- Easy switching between gradient computation methods

Example usage:
```bash
# Using finite-difference gradients (default)
python examples/find_optimal_partition.py

# Using analytic gradients
python examples/find_optimal_partition.py --analytic
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

## Project Structure

### Core Components
-   `src/`: Contains the core implementation.
    -   `mesh.py`: `TorusMesh` class for generating torus meshes
    -   `optimization.py`: Contains the `PartitionOptimizer` class
    -   `visualization.py`: `PartitionVisualizer` class for visualizations
    -   `lbfgs_optimizer.py`: L-BFGS optimization implementation
    -   `slsqp_optimizer.py`: SLSQP optimization implementation
    -   `projection_iterative.py`: Iterative projection methods

### Examples and Scripts
-   `examples/`: Contains example scripts and optimization implementations
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