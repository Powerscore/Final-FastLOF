#!/bin/bash
#
# Monitor SLURM jobs for FastLOF experiments
# Usage: ./monitor_jobs.sh [interval_seconds]
#

INTERVAL=${1:-60}  # Default 60 seconds

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

clear

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}FastLOF SLURM Job Monitor${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Monitoring jobs every ${INTERVAL} seconds. Press Ctrl+C to exit."
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${BLUE}Status at ${TIMESTAMP}${NC}"
    echo "----------------------------------------------------------------------"
    
    # Count jobs by status
    RUNNING=$(squeue -u $USER -h -t RUNNING -o "%i" 2>/dev/null | wc -l)
    PENDING=$(squeue -u $USER -h -t PENDING -o "%i" 2>/dev/null | wc -l)
    TOTAL=$((RUNNING + PENDING))
    
    if [ $TOTAL -eq 0 ]; then
        echo -e "${YELLOW}No active jobs found${NC}"
    else
        echo -e "Running: ${GREEN}${RUNNING}${NC}  |  Pending: ${YELLOW}${PENDING}${NC}  |  Total: ${TOTAL}"
        echo ""
        
        # Show detailed job list
        echo "Job Details:"
        squeue -u $USER -o "  %.8i %.9P %.40j %.8T %.10M %.6D %R" 2>/dev/null
    fi
    
    echo "----------------------------------------------------------------------"
    
    # Show recent completions from logs
    if [ -d "slurm_logs" ]; then
        echo ""
        echo "Recent Activity (last 5 completed):"
        find slurm_logs -name "*.out" -type f -printf '%T+ %p\n' 2>/dev/null | \
            sort -r | head -5 | while read timestamp file; do
                dataset=$(basename "$file" | sed 's/_fastlof_.*//')
                if grep -q "Job completed at" "$file" 2>/dev/null; then
                    echo -e "  ${GREEN}✓${NC} ${dataset}"
                elif grep -q "Exit code: [^0]" "$file" 2>/dev/null; then
                    echo -e "  ${RED}✗${NC} ${dataset} (failed)"
                fi
            done
        echo "----------------------------------------------------------------------"
    fi
    
    echo ""
    echo -e "Next update in ${INTERVAL}s... (Ctrl+C to exit)"
    echo ""
    
    sleep $INTERVAL
    
    # Clear screen for next iteration
    clear
done
