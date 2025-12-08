# SLURM System Documentation Index

Complete documentation for the automated SLURM job submission system for FastLOF experiments.

## 📚 Documentation Files

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[SLURM_QUICKSTART.md](SLURM_QUICKSTART.md)** | 5-minute setup guide | **START HERE** - First time setup |
| **[SLURM_README.md](SLURM_README.md)** | Complete documentation | Detailed reference and usage |
| **[SLURM_SUMMARY.md](SLURM_SUMMARY.md)** | System overview | Understanding the whole system |
| **[README.md](README.md)** | Main experiments readme | General experiment info |

## 🛠️ Core System Files

| File | Purpose | Type |
|------|---------|------|
| `slurm_config.yaml` | Resource configuration | Config |
| `slurm_template.sh` | Job script template | Template |
| `slurm_submit.py` | Main submission manager | Script |
| `submit_jobs.sh` | Interactive wrapper | Script |
| `check_results.py` | Progress checker | Script |
| `monitor_jobs.sh` | Live job monitor | Script |
| `setup_cluster.sh` | Initial setup helper | Script |

## 🚀 Quick Command Reference

### Setup (First Time)

```bash
# Run setup script
./setup_cluster.sh

# Or manually:
chmod +x *.sh *.py
pip install -r requirements_cluster.txt
vim slurm_config.yaml  # Update work_dir
```

### Main Operations

```bash
# Test configuration
python slurm_submit.py --dry-run

# Submit all jobs
python slurm_submit.py

# Submit specific datasets
python slurm_submit.py --dataset dataset-name

# Resume after interruption
python slurm_submit.py --resume

# Check progress
python check_results.py

# Monitor jobs
./monitor_jobs.sh
```

### Job Management

```bash
# View your jobs
squeue -u $USER

# Cancel job
scancel <job_id>

# Cancel all
scancel -u $USER

# View logs
tail -f slurm_logs/dataset_fastlof_live.log
```

## 📋 Workflows by Use Case

### Use Case 1: First-Time User

1. Read **[SLURM_QUICKSTART.md](SLURM_QUICKSTART.md)**
2. Run `./setup_cluster.sh`
3. Test: `python slurm_submit.py --dry-run`
4. Submit test job: `python slurm_submit.py --dataset annthyroid-unsupervised-ad`
5. Verify results: `python check_results.py`
6. Submit all: `python slurm_submit.py`

### Use Case 2: Troubleshooting Issues

1. Check **[SLURM_README.md](SLURM_README.md)** → Troubleshooting section
2. Review logs in `slurm_logs/`
3. Check job status: `scontrol show job <job_id>`
4. Adjust `slurm_config.yaml` if needed
5. Resume: `python slurm_submit.py --resume`

### Use Case 3: Customizing Resources

1. Identify dataset in **[SLURM_README.md](SLURM_README.md)** → Resource Allocation
2. Edit `slurm_config.yaml`:
   ```yaml
   datasets:
     your-dataset:
       cpus: 32        # Increase
       memory: "128G"  # Increase
       time: "48:00:00" # Extend
   ```
3. Regenerate: `python slurm_submit.py --generate-only`
4. Submit: `python slurm_submit.py --dataset your-dataset`

### Use Case 4: Monitoring Progress

```bash
# Option 1: Results checker
python check_results.py

# Option 2: Live monitor
./monitor_jobs.sh 60

# Option 3: SLURM commands
squeue -u $USER
sacct -u $USER --starttime=today

# Option 4: Log files
ls -lht slurm_logs/
tail -f slurm_logs/dataset_fastlof_live.log
```

## 🎯 Dataset Reference

| Dataset | Size | Priority | Est. Time | Resources |
|---------|------|----------|-----------|-----------|
| annthyroid | Small | 1 | ~12h | 16 CPUs, 32GB |
| breast-cancer | Small | 2 | ~8h | 16 CPUs, 32GB |
| dfki-artificial-3000 | Small | 3 | ~6h | 16 CPUs, 32GB |
| pen-local | Medium | 4 | ~24h | 20 CPUs, 48GB |
| pen-global | Medium | 5 | ~24h | 20 CPUs, 48GB |
| PenDigits | Medium | 6 | ~20h | 20 CPUs, 48GB |
| InternetAds | Medium | 7 | ~30h | 24 CPUs, 64GB |
| mammography | Medium | 8 | ~36h | 24 CPUs, 64GB |
| satellite | Medium | 9 | ~30h | 24 CPUs, 64GB |
| shuttle | Medium | 10 | ~36h | 24 CPUs, 64GB |
| creditcard | Large | 11 | ~48-60h | 32 CPUs, 128GB |
| kdd99 | Large | 12 | ~60-72h | 32 CPUs, 256GB |

**Total: 12 datasets, ~3-5 days with 4 concurrent jobs**

## 🔧 Configuration Reference

### slurm_config.yaml Key Settings

```yaml
global_settings:
  num_threads: 5              # IMPORTANT: Prevents thread overhead
  work_dir: "/path/to/project"  # MUST UPDATE for your cluster
  email: null                 # Optional: notifications

submission:
  max_concurrent_jobs: 4      # How many jobs at once
  check_interval: 300         # Status check frequency (seconds)
```

### Partition Options

| Partition | Nodes | MaxMem | MaxTime | Best For |
|-----------|-------|--------|---------|----------|
| `cpu` | 68 | 384GB | 3 days | Most datasets ✓ |
| `highmem` | 4 | 2.3TB | 3 days | Very large datasets |
| `cpu_il` | 264 | 256GB | 3 days | High availability |

## 🐛 Common Issues Quick Fix

| Issue | Quick Fix |
|-------|-----------|
| Job won't start | Check: `sinfo -p cpu` for availability |
| Module not found | Run: `module load python/3.9` or activate venv |
| Out of memory | Edit `slurm_config.yaml`: increase `memory` |
| Job too slow | Check logs for thread limit = 5 |
| Permission denied | Run: `chmod +x *.sh *.py` |
| Config error | Run: `./setup_cluster.sh` to reset |

## 📞 Getting Help

1. **Quick issues**: See [SLURM_README.md](SLURM_README.md) → Troubleshooting
2. **Setup problems**: Re-run `./setup_cluster.sh`
3. **Configuration**: See [SLURM_README.md](SLURM_README.md) → Customization
4. **Understanding system**: Read [SLURM_SUMMARY.md](SLURM_SUMMARY.md)
5. **Logs**: Check `slurm_logs/` directory

## 📂 Directory Structure

```
experiment_scripts/
├── Documentation
│   ├── SLURM_INDEX.md ............... (This file)
│   ├── SLURM_QUICKSTART.md .......... Quick setup
│   ├── SLURM_README.md .............. Complete docs
│   ├── SLURM_SUMMARY.md ............. System overview
│   └── README.md .................... General info
│
├── Core System
│   ├── slurm_config.yaml ............ Resource configuration
│   ├── slurm_template.sh ............ Job template
│   ├── slurm_submit.py .............. Main submission script
│   ├── submit_jobs.sh ............... Interactive wrapper
│   ├── check_results.py ............. Progress checker
│   ├── monitor_jobs.sh .............. Live monitor
│   └── setup_cluster.sh ............. Setup helper
│
├── Generated (auto-created)
│   ├── slurm_jobs/ .................. Individual job scripts
│   ├── slurm_logs/ .................. Output and error logs
│   └── .slurm_state.yaml ............ State tracking
│
└── Experiment Scripts
    ├── annthyroid-unsupervised-ad/
    │   ├── run_fastlof.py
    │   └── run_original_lof.py
    ├── breast-cancer-unsupervised-ad/
    │   └── ...
    └── ... (12 datasets total)
```

## ✅ Pre-Flight Checklist

Before submitting jobs, ensure:

- [ ] Read [SLURM_QUICKSTART.md](SLURM_QUICKSTART.md)
- [ ] Ran `./setup_cluster.sh` successfully
- [ ] Updated `work_dir` in `slurm_config.yaml`
- [ ] Tested with `python slurm_submit.py --dry-run`
- [ ] Python packages installed (`pip list | grep sklearn`)
- [ ] On correct cluster partition (check `sinfo`)
- [ ] Have disk space for results (~10GB+)
- [ ] Set up tmux/screen for long-running monitor

## 🎓 Learning Path

**Beginner**: Just want it to work
1. Read: SLURM_QUICKSTART.md
2. Run: `./setup_cluster.sh`
3. Submit: `python slurm_submit.py`

**Intermediate**: Want to understand and customize
1. Read: SLURM_QUICKSTART.md
2. Read: SLURM_SUMMARY.md
3. Customize: `slurm_config.yaml`
4. Submit phased: Small → Medium → Large datasets

**Advanced**: Need full control
1. Read: All documentation
2. Generate jobs: `python slurm_submit.py --generate-only`
3. Customize: Edit individual job scripts in `slurm_jobs/`
4. Submit: Manual `sbatch` or via submission script
5. Monitor: Custom scripts + SLURM commands

## 🎯 Success Metrics

You'll know the system is working when:

1. ✓ `--dry-run` shows expected jobs
2. ✓ Jobs appear in `squeue -u $USER`
3. ✓ Log files being created in `slurm_logs/`
4. ✓ `check_results.py` shows progress
5. ✓ Results appearing in `../results/<dataset>/fastlof_experiments/`

## 📊 Expected Outcomes

After all jobs complete (~5 days):

```
results/
├── annthyroid-unsupervised-ad/
│   └── fastlof_experiments/
│       └── threshold_1.1/
│           ├── results_summary.csv ........ Summary of all k-values
│           ├── comparison_plot.png ........ Visualization
│           └── detailed_results.csv ....... Full metrics per run
├── breast-cancer-unsupervised-ad/
│   └── ... (same structure)
...
└── kdd99-unsupervised-ad/
    └── ... (same structure)

Total: 12 datasets × multiple k-values = comprehensive results
```

## 🚀 Ready to Start?

```bash
# Your journey begins here:
cat SLURM_QUICKSTART.md

# Or jump right in:
./setup_cluster.sh
```

Good luck with your experiments! 🎉
