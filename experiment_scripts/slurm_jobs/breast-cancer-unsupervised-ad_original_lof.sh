#!/bin/bash
#SBATCH --job-name=original_lof_breast-cancer-unsupervised-ad
#SBATCH --output=slurm_logs/breast-cancer-unsupervised-ad_original_lof_%j.out
#SBATCH --error=slurm_logs/breast-cancer-unsupervised-ad_original_lof_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --mail-user=alaa.ashraf.uni@gmail.com
#SBATCH --mail-type=FAIL,END
# ============================================================================
# Original LOF Experiment - breast-cancer-unsupervised-ad
# ============================================================================
# Generated automatically by slurm_submit.py
# Estimated runtime: ~8 hours
# ============================================================================

# Thread limiting to prevent overhead
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5

# Print job information
echo "========================================================================"
echo "Job Information"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 32G"
echo "Partition: cpu"
echo "Thread limit: 5"
echo "========================================================================"
echo ""

# Change to working directory
cd /pfs/data6/home/hu/hu_hu/hu_abdeal01/Final-FastLOF
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
echo "Starting Experiment: breast-cancer-unsupervised-ad - Original LOF"
echo "========================================================================"
echo ""

python -u experiment_scripts/breast-cancer-unsupervised-ad/run_original_lof.py 2>&1 | tee -a experiment_scripts/slurm_logs/breast-cancer-unsupervised-ad_original_lof_live.log

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
