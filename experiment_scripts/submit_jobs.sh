#!/bin/bash
#
# Convenience wrapper for SLURM job submission
# Usage: ./submit_jobs.sh [OPTIONS]
#

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}FastLOF SLURM Job Submission System${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found${NC}"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Check if required files exist
if [ ! -f "slurm_submit.py" ]; then
    echo -e "${RED}Error: slurm_submit.py not found${NC}"
    exit 1
fi

if [ ! -f "slurm_config.yaml" ]; then
    echo -e "${RED}Error: slurm_config.yaml not found${NC}"
    exit 1
fi

# Make Python script executable
chmod +x slurm_submit.py

# Parse arguments
if [ $# -eq 0 ]; then
    # No arguments - show menu
    echo "What would you like to do?"
    echo ""
    echo "  1) Generate job scripts only"
    echo "  2) Dry run (test without submitting)"
    echo "  3) Submit all jobs (4 concurrent)"
    echo "  4) Submit with custom concurrent limit"
    echo "  5) Resume previous submission"
    echo "  6) Submit specific datasets only"
    echo "  0) Exit"
    echo ""
    read -p "Enter choice [0-6]: " choice
    
    case $choice in
        1)
            echo -e "${GREEN}Generating job scripts...${NC}"
            $PYTHON_CMD slurm_submit.py --generate-only
            ;;
        2)
            echo -e "${YELLOW}Running dry run...${NC}"
            $PYTHON_CMD slurm_submit.py --dry-run
            ;;
        3)
            echo -e "${GREEN}Submitting all jobs...${NC}"
            $PYTHON_CMD slurm_submit.py
            ;;
        4)
            read -p "Enter max concurrent jobs: " max_jobs
            echo -e "${GREEN}Submitting with max $max_jobs concurrent jobs...${NC}"
            $PYTHON_CMD slurm_submit.py --max-concurrent $max_jobs
            ;;
        5)
            echo -e "${GREEN}Resuming previous submission...${NC}"
            $PYTHON_CMD slurm_submit.py --resume
            ;;
        6)
            echo "Available datasets:"
            echo "  - annthyroid-unsupervised-ad"
            echo "  - breast-cancer-unsupervised-ad"
            echo "  - creditcard"
            echo "  - dfki-artificial-3000-unsupervised-ad"
            echo "  - InternetAds_norm_02_v01"
            echo "  - kdd99-unsupervised-ad"
            echo "  - mammography"
            echo "  - pen-global-unsupervised-ad"
            echo "  - pen-local-unsupervised-ad"
            echo "  - PenDigits_withoutdupl_norm_v01"
            echo "  - satellite-unsupervised-ad"
            echo "  - shuttle-unsupervised-ad"
            echo ""
            read -p "Enter dataset names (space-separated): " datasets
            dataset_args=""
            for ds in $datasets; do
                dataset_args="$dataset_args --dataset $ds"
            done
            echo -e "${GREEN}Submitting selected datasets...${NC}"
            $PYTHON_CMD slurm_submit.py $dataset_args
            ;;
        0)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
else
    # Arguments provided - pass through to Python script
    $PYTHON_CMD slurm_submit.py "$@"
fi

echo ""
echo -e "${BLUE}======================================================================${NC}"
