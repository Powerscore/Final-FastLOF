# SLURM Job Submission System - Complete Summary

## 📦 What's Included

This automated system manages batch submission of FastLOF experiments to SLURM clusters.

### Core Files

| File | Purpose |
|------|---------|
| `slurm_config.yaml` | Resource configuration (CPUs, memory, time) per dataset |
| `slurm_template.sh` | SLURM job script template with thread limiting |
| `slurm_submit.py` | Main Python script - generates jobs, submits, monitors |
| `submit_jobs.sh` | Interactive bash wrapper with menu |
| `check_results.py` | Check experiment completion status |
| `monitor_jobs.sh` | Real-time job monitoring dashboard |

### Auto-Generated

| Directory/File | Content |
|----------------|---------|
| `slurm_jobs/` | Individual job scripts (12 files, auto-generated) |
| `slurm_logs/` | Job output/error logs and live logs |
| `.slurm_state.yaml` | State tracking for resume functionality |

## 🎯 Key Features

### 1. Intelligent Resource Allocation
- **Small datasets** (3): 16 CPUs, 32GB, 24h
- **Medium datasets** (7): 20-24 CPUs, 48-64GB, 36-48h  
- **Large datasets** (2): 32 CPUs, 128-256GB, 72h
- **Thread limiting**: Fixed at 5 threads to prevent overhead

### 2. Automatic Job Management
- Submits 4 jobs initially (configurable)
- Monitors status every 5 minutes
- Auto-submits next job when slot becomes available
- Priority-based queue (small → large)

### 3. State Management
- Saves progress continuously
- Resume from interruption with `--resume`
- Tracks completed, failed, and pending jobs

### 4. Partition Strategy
- **Primary**: `cpu` partition (68 nodes, 384GB RAM, 3-day limit)
- **Fallback**: `highmem` for very large datasets (4 nodes, 2.3TB RAM)
- **Alternative**: `cpu_il` if main partitions busy (264 nodes)

## 🚀 Usage Workflows

### Workflow 1: All-in-One (Recommended)

```bash
# Upload to cluster
scp -r experiment_scripts/ user@cluster:/path/to/fast-lof/

# Configure
cd /path/to/fast-lof/experiment_scripts
vim slurm_config.yaml  # Update work_dir path

# Test
python slurm_submit.py --dry-run

# Submit all (runs in background with monitoring)
tmux new -s fastlof
python slurm_submit.py
# Ctrl+B, D to detach

# Check progress anytime
python check_results.py
```

**Expected duration**: 3-5 days for all 12 datasets

### Workflow 2: Phased Approach (Safer)

```bash
# Phase 1: Test with small datasets (~1 day)
python slurm_submit.py \
  --dataset annthyroid-unsupervised-ad \
  --dataset breast-cancer-unsupervised-ad \
  --dataset dfki-artificial-3000-unsupervised-ad

# Verify results
python check_results.py

# Phase 2: Medium datasets (~2 days)
python slurm_submit.py \
  --dataset pen-local-unsupervised-ad \
  --dataset pen-global-unsupervised-ad \
  --dataset PenDigits_withoutdupl_norm_v01 \
  --dataset mammography \
  --dataset satellite-unsupervised-ad \
  --dataset shuttle-unsupervised-ad \
  --dataset InternetAds_norm_02_v01

# Phase 3: Large datasets (~3 days)
python slurm_submit.py \
  --dataset creditcard \
  --dataset kdd99-unsupervised-ad \
  --max-concurrent 2
```

### Workflow 3: Interactive Menu

```bash
./submit_jobs.sh

# Select option:
# 1) Generate job scripts only
# 2) Dry run (test without submitting)
# 3) Submit all jobs (4 concurrent)
# 4) Submit with custom concurrent limit
# 5) Resume previous submission
# 6) Submit specific datasets only
```

## 📊 Monitoring

### Option 1: Built-in Monitoring

The `slurm_submit.py` script shows live status:

```
================================================================================
Job Status - 2025-12-08 14:30:00
================================================================================
Running:   4
Completed: 3
Failed:    0
Pending:   5

Currently Running:
  • annthyroid-unsupervised-ad                  [12345] (~12 hours)
  • breast-cancer-unsupervised-ad               [12346] (~8 hours)
  ...
================================================================================
```

### Option 2: Standalone Monitor

```bash
# In a separate terminal
./monitor_jobs.sh 60  # Update every 60 seconds
```

### Option 3: Results Checker

```bash
python check_results.py

# Output shows completion with progress bar:
# Status Overview:
#   ✓ Completed:    3/12
#   ⚠ In Progress:  2/12
#   ✗ Not Started:  7/12
# Overall Progress: [████████████░░░░░░░░░░░░] 25.0%
```

### Option 4: Manual SLURM Commands

```bash
# View all your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# View live logs
tail -f slurm_logs/annthyroid-unsupervised-ad_fastlof_live.log

# Cancel job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## ⚙️ Configuration Reference

### slurm_config.yaml Structure

```yaml
global_settings:
  num_threads: 5              # Thread limit for all jobs
  work_dir: "/path/to/fast-lof"
  email: null                 # Optional: your@email.com
  python_cmd: "python -u"

datasets:
  dataset-name:
    priority: 1               # Lower = runs first
    partition: "cpu"
    cpus: 16
    memory: "32G"
    time: "24:00:00"         # HH:MM:SS
    estimated_time: "~12 hours"

submission:
  max_concurrent_jobs: 4      # Simultaneous jobs
  check_interval: 300         # Seconds between status checks
  retry_failed: false
```

## 🎓 Advanced Tips

### 1. Running in Background (tmux/screen)

```bash
# Using tmux (recommended)
tmux new -s fastlof
python slurm_submit.py
# Ctrl+B then D to detach

# Reattach later
tmux attach -t fastlof

# List sessions
tmux ls
```

### 2. Adjusting Resources for Specific Dataset

```bash
# Edit config
vim slurm_config.yaml

# Change for kdd99:
datasets:
  kdd99-unsupervised-ad:
    partition: "highmem"     # Switch to high-memory partition
    cpus: 64                 # More CPUs
    memory: "512G"           # More RAM
    time: "96:00:00"         # 4 days instead of 3
```

### 3. Testing Single Dataset First

```bash
# Generate job script
python slurm_submit.py --generate-only

# Edit the job script if needed
vim slurm_jobs/annthyroid-unsupervised-ad_fastlof.sh

# Submit manually
sbatch slurm_jobs/annthyroid-unsupervised-ad_fastlof.sh

# Or submit via tool
python slurm_submit.py --dataset annthyroid-unsupervised-ad
```

### 4. Handling Memory Issues

If job fails with OOM (Out of Memory):

```bash
# 1. Check error log
cat slurm_logs/dataset_fastlof_*.err

# 2. Increase memory in config
vim slurm_config.yaml
# Change memory: "64G" → "128G"

# 3. Consider sampling in dataset script
vim dataset-name/run_fastlof.py
# Change FRACTION = 1 → FRACTION = 0.5
```

### 5. Speeding Up Large Datasets

If dataset runs too slowly:

```python
# Edit dataset/run_fastlof.py

# Option 1: Reduce k values
K_VALUES = [10, 20, 30]  # Instead of [10, 20, 30, 40, 50]

# Option 2: Reduce runs
N_RUNS = 3  # Instead of 5

# Option 3: Reduce chunk range
MIN_CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 5000
CHUNK_INTERVAL = 1000

# Option 4: Sample dataset
FRACTION = 0.5  # Use 50% of data
```

## 🐛 Troubleshooting Guide

### Issue: Jobs don't start

**Check**: Partition availability
```bash
sinfo -p cpu
```

**Fix**: Switch to less busy partition in config

---

### Issue: "Module not found" error

**Check**: Python environment
```bash
which python
python -c "import sklearn; print(sklearn.__version__)"
```

**Fix**: Load module or activate venv before running
```bash
module load python/3.9
# or
source /path/to/venv/bin/activate
```

---

### Issue: Out of memory

**Check**: Memory usage in logs
```bash
sacct -j <job_id> --format=JobID,MaxRSS,State
```

**Fix**: Increase memory or use sampling

---

### Issue: Job runs forever

**Check**: Thread limiting in output
```bash
grep "OMP_NUM_THREADS" slurm_logs/dataset_*.out
```

**Fix**: Should be 5. If not set, job script may be wrong.

---

### Issue: Permission denied

**Fix**: Make scripts executable
```bash
chmod +x slurm_submit.py submit_jobs.sh monitor_jobs.sh
```

## 📈 Expected Timeline

| Time | Milestone |
|------|-----------|
| Day 0 | Upload files, configure, test with dry run |
| Day 0-1 | Small datasets complete (annthyroid, breast-cancer, dfki) |
| Day 1-2 | Medium datasets running (pen, pendigits, mammography, etc.) |
| Day 2-3 | Medium datasets complete |
| Day 3-5 | Large datasets running (creditcard, kdd99) |
| Day 5 | All experiments complete ✓ |

**Total**: ~5 days with 4 concurrent jobs

Can be faster with more concurrent jobs (6-8), but risks queue limits.

## 📁 Output Structure

```
results/
├── annthyroid-unsupervised-ad/
│   └── fastlof_experiments/
│       └── threshold_1.1/
│           ├── results_summary.csv
│           ├── detailed_results.csv
│           ├── comparison_plot.png
│           └── ... (per k-value results)
├── breast-cancer-unsupervised-ad/
│   └── fastlof_experiments/
│       └── threshold_1.1/
│           └── ...
...

slurm_logs/
├── annthyroid-unsupervised-ad_fastlof_12345.out
├── annthyroid-unsupervised-ad_fastlof_12345.err
├── annthyroid-unsupervised-ad_fastlof_live.log
...
```

## 📞 Getting Help

1. **Check logs**: `slurm_logs/` directory
2. **Check docs**: `SLURM_README.md` for detailed info
3. **Quick start**: `SLURM_QUICKSTART.md` for fast setup
4. **Test first**: Always use `--dry-run` before real submission
5. **Monitor**: Use `python check_results.py` for progress

## ✅ Final Checklist

Before submitting all jobs:

- [ ] Uploaded all experiment scripts to cluster
- [ ] Updated `work_dir` in `slurm_config.yaml`
- [ ] Tested with `--dry-run` successfully
- [ ] Verified Python environment has required packages
- [ ] Tested one small dataset manually
- [ ] Set up tmux/screen session for monitoring
- [ ] Know how to check logs and cancel jobs
- [ ] Have enough disk space for results (~10GB+)

Good luck with your experiments! 🚀
