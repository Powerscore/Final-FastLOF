#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --output=slurm_logs/{DATASET_NAME}_fastlof_%j.out
#SBATCH --error=slurm_logs/{DATASET_NAME}_fastlof_%j.err
#SBATCH --time={TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={MEMORY}
#SBATCH --partition={PARTITION}
{EMAIL_LINES}
# ============================================================================
# FastLOF Experiment - {DATASET_NAME}
# ============================================================================
# Generated automatically by slurm_submit.py
# Estimated runtime: {ESTIMATED_TIME}
# ============================================================================

# Thread limiting to prevent overhead
export OMP_NUM_THREADS={NUM_THREADS}
export MKL_NUM_THREADS={NUM_THREADS}
export OPENBLAS_NUM_THREADS={NUM_THREADS}
export NUMEXPR_NUM_THREADS={NUM_THREADS}

# Print job information
echo "========================================================================"
echo "Job Information"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: {MEMORY}"
echo "Partition: {PARTITION}"
echo "Thread limit: {NUM_THREADS}"
echo "========================================================================"
echo ""

# Change to working directory
cd {WORK_DIR}
echo "Working directory: $(pwd)"
echo ""

# Print Python and package info
echo "========================================================================"
echo "Environment Information"
echo "========================================================================"
python --version
echo ""
pip list | grep -E "numpy|pandas|scikit-learn|matplotlib"
echo "========================================================================"
echo ""

# Run the experiment script
echo "========================================================================"
echo "Starting Experiment: {DATASET_NAME} - FastLOF"
echo "========================================================================"
echo ""

{PYTHON_CMD} {SCRIPT_PATH} 2>&1 | tee -a slurm_logs/{DATASET_NAME}_fastlof_live.log

EXIT_CODE=$?

# Print completion information
echo ""
echo "========================================================================"
echo "Job Completion"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "Completed at: $(date)"
echo "========================================================================"

exit $EXIT_CODE
