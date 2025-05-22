#!/bin/bash

# Default values
INPUT_FILE="parameters/input.yaml"
OUTPUT_DIR="results"
REFINEMENT_LEVELS=1
VERTICES_INCREMENT=1000
USE_ANALYTIC=true
SOLUTION_DIR="/proj/snic2020-15-36/private/solutions"  # Default solution directory
TIME_LIMIT="24:00:00"  # Default time limit

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
JOB_NAME="part${N_PARTITIONS}_vert${TOTAL_VERTICES}_${TIMESTAMP}"

# Create a temporary SLURM script
SLURM_SCRIPT=$(mktemp)

# Write SLURM directives
cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH -A snic2020-15-36
#SBATCH -p core
#SBATCH -n 1
#SBATCH -t ${TIME_LIMIT}
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${OUTPUT_DIR}/${JOB_NAME}.out
#SBATCH -e ${OUTPUT_DIR}/${JOB_NAME}.err

# Load required modules
module load python/3.9.5

# Set up environment
export PYTHONPATH="\${PYTHONPATH}:\$(pwd)"

# Run the Python script
python examples/slsqp_optimizer.py \\
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
echo "Output will be written to: ${OUTPUT_DIR}/${JOB_NAME}.out"
echo "Errors will be written to: ${OUTPUT_DIR}/${JOB_NAME}.err"

# Submit a dependent job that will run after the main job completes
DEPENDENT_SCRIPT=$(mktemp)
cat > "$DEPENDENT_SCRIPT" << EOF
#!/bin/bash
#SBATCH -A snic2020-15-36
#SBATCH -p core
#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH -J ${JOB_NAME}_time
#SBATCH -d afterok:${JOB_ID}
#SBATCH -o ${OUTPUT_DIR}/${JOB_NAME}_time.out
#SBATCH -e ${OUTPUT_DIR}/${JOB_NAME}_time.err

end_time=\$(date +%s)
execution_time=\$((end_time - ${start_time}))
echo "Total execution time: \${execution_time} seconds"
EOF

sbatch "$DEPENDENT_SCRIPT"
rm "$DEPENDENT_SCRIPT"
