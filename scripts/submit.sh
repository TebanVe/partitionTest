#!/bin/bash

# Project configuration
PROJECT_FOLDER="snic2020-15-36"  # Project folder for file paths
PROJECT_ID="uppmax2025-2-192"    # Project ID for SLURM
PROJECT_BASE="/proj/${PROJECT_FOLDER}"


# Default values
INPUT_FILE="parameters/input.yaml"
OUTPUT_DIR="results"
REFINEMENT_LEVELS=1
VERTICES_INCREMENT=1000
USE_ANALYTIC=true
SOLUTIONS_DIR="${PROJECT_BASE}/private/LINKED_LST_MANIFOLD/PART_SOLUTION"  # Directory for optimization solutions
TIME_LIMIT="12:00:00"  # Default time limit

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --refinement)
            REFINEMENT_LEVELS="$2"
            shift 2
            ;;
        --vertices)
            VERTICES_INCREMENT="$2"
            shift 2
            ;;
        --solution-dir)
            SOLUTION_DIR="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --analytic)
            USE_ANALYTIC=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SOLUTION_DIR"

# Read parameters from YAML file
if [ -f "$INPUT_FILE" ]; then
    # Extract n_theta and n_phi using grep and awk
    N_THETA=$(grep "n_theta:" "$INPUT_FILE" | awk '{print $2}')
    N_PHI=$(grep "n_phi:" "$INPUT_FILE" | awk '{print $2}')
    N_PARTITIONS=$(grep "n_partitions:" "$INPUT_FILE" | awk '{print $2}')
    
    # Calculate total vertices
    TOTAL_VERTICES=$((N_THETA * N_PHI))
else
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

# Generate a unique job name with more information
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="${TIMESTAMP}_npart${N_PARTITIONS}_nvert${TOTAL_VERTICES}"

# Create job logs directory with timestamp and job name
JOB_LOGS_DIR="${OUTPUT_DIR}/job_logs/${JOB_NAME}"
mkdir -p "$JOB_LOGS_DIR"

# Create a temporary SLURM script
SLURM_SCRIPT=$(mktemp)

# Write SLURM directives
cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH -A ${PROJECT_ID}
#SBATCH -p core
#SBATCH -n 1
#SBATCH -t ${TIME_LIMIT}
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${JOB_LOGS_DIR}/${JOB_NAME}.out
#SBATCH -e ${JOB_LOGS_DIR}/${JOB_NAME}.err

# Load required modules
module load python/3.9.5

# Set up environment
export PYTHONPATH="\${PYTHONPATH}:\$(pwd)"

# Run the Python script
python examples/find_optimal_partition.py \\
    --input "${INPUT_FILE}" \\
    --refinement-levels "${REFINEMENT_LEVELS}" \\
    --vertices-increment "${VERTICES_INCREMENT}" \\
    --solution-dir "${SOLUTION_DIR}" \\
    $([ "$USE_ANALYTIC" = true ] && echo "--analytic")
EOF

# Submit the job
start_time=$(date +%s)
JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

# Clean up
rm "$SLURM_SCRIPT"

echo "Job submitted with name: ${JOB_NAME}"
echo "Job ID: ${JOB_ID}"
echo "SLURM logs will be written to: ${JOB_LOGS_DIR}"
echo "Output file: ${JOB_LOGS_DIR}/${JOB_NAME}.out"
echo "Error file: ${JOB_LOGS_DIR}/${JOB_NAME}.err"
