# SLURM Job Submission System for FastLOF Experiments

This directory contains an automated system for submitting and managing FastLOF experiments on SLURM clusters.

## 📁 Files Overview

- **`slurm_config.yaml`** - Resource configuration for each dataset
- **`slurm_template.sh`** - SLURM job script template
- **`slurm_submit.py`** - Python script to manage job submission and monitoring
- **`slurm_jobs/`** - Generated job scripts (auto-created)
- **`slurm_logs/`** - Job output and error logs (auto-created)

## 🚀 Quick Start

### 1. Configure Settings

Edit `slurm_config.yaml` to adjust:
- Working directory path (`work_dir`)
- Email notifications (optional)
- Resource allocations per dataset
- Max concurrent jobs

### 2. Generate Job Scripts

```bash
python slurm_submit.py --generate-only
```

This creates individual SLURM scripts in `slurm_jobs/` directory.

### 3. Test with Dry Run

```bash
python slurm_submit.py --dry-run
```

Shows what would be submitted without actually submitting jobs.

### 4. Submit Jobs

```bash
python slurm_submit.py
```

This will:
- Submit jobs in batches (default: 4 concurrent)
- Monitor job status every 5 minutes
- Automatically submit next jobs when slots become available
- Save progress to `.slurm_state.yaml`

### 5. Resume After Interruption

If interrupted (Ctrl+C) or need to continue later:

```bash
python slurm_submit.py --resume
```

## 📊 Resource Allocation Strategy

### Small Datasets
- **Datasets**: annthyroid, breast-cancer, dfki-artificial-3000
- **Resources**: 16 CPUs, 32GB RAM, 24 hours
- **Partition**: `cpu`

### Medium Datasets
- **Datasets**: pen-local, pen-global, PenDigits, InternetAds, mammography, satellite, shuttle
- **Resources**: 20-24 CPUs, 48-64GB RAM, 36-48 hours
- **Partition**: `cpu`

### Large Datasets
- **Datasets**: creditcard, kdd99
- **Resources**: 32 CPUs, 128-256GB RAM, 72 hours (3 days)
- **Partition**: `cpu` (can fallback to `highmem`)

## 🔧 Advanced Usage

### Submit Specific Datasets Only

```bash
python slurm_submit.py --dataset annthyroid-unsupervised-ad --dataset breast-cancer-unsupervised-ad
```

### Change Max Concurrent Jobs

```bash
python slurm_submit.py --max-concurrent 5
```

### Check Job Status Manually

```bash
# View all your jobs
squeue -u $USER

# View specific job details
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# View job output in real-time
tail -f slurm_logs/<dataset>_fastlof_live.log
```

## 📈 Monitoring Progress

### Real-time Logs

Each job produces:
- **`<dataset>_fastlof_<jobid>.out`** - Standard output
- **`<dataset>_fastlof_<jobid>.err`** - Error output
- **`<dataset>_fastlof_live.log`** - Live log with tee

### Job Status Display

The submission script shows:
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
  • dfki-artificial-3000-unsupervised-ad        [12347] (~6 hours)
  • pen-local-unsupervised-ad                   [12348] (~24 hours)

Next in Queue:
  • pen-global-unsupervised-ad                  (~24 hours)
  • PenDigits_withoutdupl_norm_v01              (~20 hours)
  ... and 3 more
================================================================================
```

## ⚙️ Thread Limiting

All jobs limit threads to **5** via environment variables:
```bash
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5
```

This prevents overhead from excessive parallelization that can slow down execution.

## 📝 Customization

### Modify Resource for Specific Dataset

Edit `slurm_config.yaml`:

```yaml
datasets:
  kdd99-unsupervised-ad:
    partition: "highmem"  # Change partition
    cpus: 64              # Increase CPUs
    memory: "512G"        # Increase memory
    time: "96:00:00"      # Extend time limit
```

### Change Submission Strategy

Edit `slurm_config.yaml`:

```yaml
submission:
  max_concurrent_jobs: 6  # Submit more jobs at once
  check_interval: 180     # Check more frequently (3 minutes)
```

## 🐛 Troubleshooting

### Job Fails Immediately

1. Check error log: `slurm_logs/<dataset>_fastlof_<jobid>.err`
2. Verify paths in `slurm_config.yaml` (especially `work_dir`)
3. Ensure Python environment has required packages

### Out of Memory Error

1. Increase memory in `slurm_config.yaml`
2. For very large datasets, consider using `highmem` partition
3. Check if dataset needs sampling (edit `run_fastlof.py` to set `FRACTION < 1`)

### Job Takes Too Long

1. Check if thread limiting is applied (should see in job output)
2. Consider reducing `K_VALUES` or `N_RUNS` in dataset's `run_fastlof.py`
3. Increase time limit in `slurm_config.yaml`

### Cannot Submit More Jobs

1. Check your user's job limit: `squeue -u $USER | wc -l`
2. Reduce `max_concurrent_jobs` in config
3. Wait for some jobs to complete

## 📋 Complete Workflow Example

```bash
# 1. Review configuration
cat slurm_config.yaml

# 2. Generate and test
python slurm_submit.py --generate-only
python slurm_submit.py --dry-run

# 3. Start with small datasets first
python slurm_submit.py --dataset annthyroid-unsupervised-ad --dataset breast-cancer-unsupervised-ad

# 4. If successful, submit all
python slurm_submit.py --max-concurrent 4

# 5. Monitor in another terminal
watch -n 60 'squeue -u $USER'

# 6. If interrupted, resume
python slurm_submit.py --resume
```

## 🎯 Priority Order

Jobs are submitted in priority order (1-12):
1. annthyroid (smallest, fastest)
2. breast-cancer
3. dfki-artificial-3000
4. pen-local
5. pen-global
6. PenDigits
7. InternetAds
8. mammography
9. satellite
10. shuttle
11. creditcard
12. kdd99 (largest, slowest - runs last)

## 💾 State Management

The script saves its state to `.slurm_state.yaml` including:
- Submitted job IDs
- Completed datasets
- Failed datasets
- Pending queue

This allows resuming after interruptions or system restarts.

## 📧 Email Notifications (Optional)

To receive email notifications, edit `slurm_config.yaml`:

```yaml
global_settings:
  email: "your.email@uni.de"
  email_type: "FAIL,END"  # Options: BEGIN,END,FAIL,ALL
```

You'll receive emails when jobs:
- Complete successfully (END)
- Fail (FAIL)
- Start (BEGIN) - if enabled
