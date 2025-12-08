#!/bin/bash
#
# Setup script for FastLOF experiments on SLURM cluster
# Usage: ./setup_cluster.sh
#

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}FastLOF SLURM Cluster Setup${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# 1. Check Python
echo -e "${YELLOW}[1/6] Checking Python installation...${NC}"
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python not found. Please load Python module or install Python.${NC}"
    echo "Try: module load python/3.9"
    exit 1
fi

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
echo ""

# 2. Check/Install dependencies
echo -e "${YELLOW}[2/6] Checking Python dependencies...${NC}"

# Check if packages are installed
MISSING_PACKAGES=""
for pkg in numpy pandas scikit-learn matplotlib scipy pyod numba yaml; do
    if ! $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo -e "${YELLOW}Missing packages:$MISSING_PACKAGES${NC}"
    echo "Installing from requirements_cluster.txt..."
    
    if [ -f "requirements_cluster.txt" ]; then
        pip install --user -r requirements_cluster.txt
        echo -e "${GREEN}✓ Dependencies installed${NC}"
    else
        echo -e "${RED}Error: requirements_cluster.txt not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ All dependencies already installed${NC}"
fi
echo ""

# 3. Make scripts executable
echo -e "${YELLOW}[3/6] Making scripts executable...${NC}"
chmod +x slurm_submit.py 2>/dev/null || true
chmod +x submit_jobs.sh 2>/dev/null || true
chmod +x monitor_jobs.sh 2>/dev/null || true
chmod +x check_results.py 2>/dev/null || true
chmod +x setup_cluster.sh 2>/dev/null || true
echo -e "${GREEN}✓ Scripts are now executable${NC}"
echo ""

# 4. Create directories
echo -e "${YELLOW}[4/6] Creating necessary directories...${NC}"
mkdir -p slurm_jobs
mkdir -p slurm_logs
mkdir -p ../results
echo -e "${GREEN}✓ Directories created:${NC}"
echo "  - slurm_jobs/"
echo "  - slurm_logs/"
echo "  - ../results/"
echo ""

# 5. Detect working directory and update config
echo -e "${YELLOW}[5/6] Configuring working directory...${NC}"
CURRENT_DIR=$(pwd)
WORK_DIR=$(dirname "$CURRENT_DIR")

if [ -f "slurm_config.yaml" ]; then
    # Check if work_dir needs updating
    CURRENT_WORK_DIR=$(grep "work_dir:" slurm_config.yaml | awk '{print $2}' | tr -d '"')
    
    if [ "$CURRENT_WORK_DIR" != "$WORK_DIR" ]; then
        echo -e "${YELLOW}Current config work_dir: $CURRENT_WORK_DIR${NC}"
        echo -e "${YELLOW}Detected work_dir:       $WORK_DIR${NC}"
        echo ""
        read -p "Update work_dir in config? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Backup original config
            cp slurm_config.yaml slurm_config.yaml.bak
            # Update work_dir (works on both Linux and macOS)
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s|work_dir: .*|work_dir: \"$WORK_DIR\"|" slurm_config.yaml
            else
                sed -i "s|work_dir: .*|work_dir: \"$WORK_DIR\"|" slurm_config.yaml
            fi
            echo -e "${GREEN}✓ Updated work_dir to: $WORK_DIR${NC}"
            echo "(Backup saved as slurm_config.yaml.bak)"
        else
            echo -e "${YELLOW}⚠ Remember to manually update work_dir in slurm_config.yaml${NC}"
        fi
    else
        echo -e "${GREEN}✓ work_dir already configured correctly${NC}"
    fi
else
    echo -e "${RED}Error: slurm_config.yaml not found${NC}"
    exit 1
fi
echo ""

# 6. Test setup
echo -e "${YELLOW}[6/6] Testing setup...${NC}"

# Test Python imports
echo "Testing Python imports..."
$PYTHON_CMD -c "
import sys
try:
    import numpy
    import pandas
    import sklearn
    import matplotlib
    import scipy
    import pyod
    import numba
    import yaml
    print('✓ All imports successful')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
" || exit 1

# Test slurm commands
echo "Testing SLURM commands..."
if command -v squeue &> /dev/null; then
    echo -e "${GREEN}✓ SLURM commands available${NC}"
else
    echo -e "${YELLOW}⚠ SLURM commands not found (may not be on login node)${NC}"
fi

# Test YAML config loading
echo "Testing config loading..."
$PYTHON_CMD -c "
import yaml
with open('slurm_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(f\"✓ Config loaded: {len(config['datasets'])} datasets configured\")
" || exit 1

echo ""

# Summary
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Review configuration:"
echo "   ${BLUE}cat slurm_config.yaml${NC}"
echo ""
echo "2. Test with dry run:"
echo "   ${BLUE}python slurm_submit.py --dry-run${NC}"
echo ""
echo "3. Submit a test job:"
echo "   ${BLUE}python slurm_submit.py --dataset annthyroid-unsupervised-ad${NC}"
echo ""
echo "4. Submit all jobs:"
echo "   ${BLUE}python slurm_submit.py${NC}"
echo ""
echo "5. Monitor progress:"
echo "   ${BLUE}python check_results.py${NC}"
echo ""
echo "For more information:"
echo "  - Quick start: ${BLUE}cat SLURM_QUICKSTART.md${NC}"
echo "  - Full docs:   ${BLUE}cat SLURM_README.md${NC}"
echo "  - Summary:     ${BLUE}cat SLURM_SUMMARY.md${NC}"
echo ""
echo -e "${GREEN}Happy experimenting! 🚀${NC}"
echo ""
