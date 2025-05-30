#!/bin/bash

# Project configuration
PROJECT_FOLDER="snic2020-15-36"  # Project folder for file paths
PROJECT_ID="uppmax2025-2-192"    # Project ID for SLURM
PROJECT_BASE="/proj/${PROJECT_FOLDER}"

# Default values
INPUT_FILE="parameters/input.yaml"
OUTPUT_DIR="results"
SOLUTION_DIR="${PROJECT_BASE}/private/LINKED_LST_MANIFOLD/PART_SOLUTION"  # Directory for optimization solutions
TIME_LIMIT="12:00:00"  # Default time limit

# Parse command line arguments (only those that can be passed to Python or are needed for SLURM/logs)
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
        --solution-dir)
            SOLUTION_DIR="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
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

# Extract info from YAML for job naming (not for passing to Python)
if [ -f "$INPUT_FILE" ]; then
    N_THETA=$(grep "n_theta:" "$INPUT_FILE" | awk '{print $2}')
    N_PHI=$(grep "n_phi:" "$INPUT_FILE" | awk '{print $2}')
    N_PARTITIONS=$(grep "n_partitions:" "$INPUT_FILE" | awk '{print $2}')
    LAMBDA=$(grep "lambda_penalty:" "$INPUT_FILE" | awk '{print $2}')
    SEED=$(grep "seed:" "$INPUT_FILE" | awk '{print $2}')
    N_THETA_INCREMENT=$(grep "n_theta_increment:" "$INPUT_FILE" | awk '{print $2}')
    N_PHI_INCREMENT=$(grep "n_phi_increment:" "$INPUT_FILE" | awk '{print $2}')
    REFINEMENT_LEVELS=$(grep "refinement_levels:" "$INPUT_FILE" | awk '{print $2}')
    # Calculate final n_theta and n_phi
    if [ "$REFINEMENT_LEVELS" -gt 1 ]; then
        FINAL_N_THETA=$((N_THETA + (REFINEMENT_LEVELS - 1) * N_THETA_INCREMENT))
        FINAL_N_PHI=$((N_PHI + (REFINEMENT_LEVELS - 1) * N_PHI_INCREMENT))
        N_THETA_INFO="${N_THETA}-${FINAL_N_THETA}_inct${N_THETA_INCREMENT}"
        N_PHI_INFO="${N_PHI}-${FINAL_N_PHI}_incp${N_PHI_INCREMENT}"
    else
        N_THETA_INFO="${N_THETA}"
        N_PHI_INFO="${N_PHI}"
    fi
else
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

# Generate a unique job name with more information
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="${TIMESTAMP}_npart${N_PARTITIONS}_nt${N_THETA_INFO}_np${N_PHI_INFO}_lam${LAMBDA}_seed${SEED}"

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
python examples/find_optimal_partition.py \
    --input "${INPUT_FILE}" \
    --solution-dir "${SOLUTION_DIR}"
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
