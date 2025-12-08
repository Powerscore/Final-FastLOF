# SLURM Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Upload to Cluster

```bash
# On your local machine
scp -r "experiment_scripts/" username@cluster:/path/to/fast-lof/
```

### Step 2: Configure Paths

Edit `slurm_config.yaml`:

```yaml
global_settings:
  work_dir: "/home/hu/hu_hu/hu_abdeal01/fast-lof"  # ← Update this path
  email: "your.email@uni.de"  # ← Optional: add your email
```

### Step 3: Test Setup

```bash
cd /path/to/fast-lof/experiment_scripts
python slurm_submit.py --dry-run
```

### Step 4: Submit Jobs

```bash
# Interactive menu
./submit_jobs.sh

# Or directly
python slurm_submit.py
```

## 📊 What Happens Next?

1. **Initial submission**: 4 jobs start immediately
2. **Monitoring**: Script checks status every 5 minutes
3. **Auto-submission**: New jobs start when slots free up
4. **Progress saved**: Can resume anytime with `--resume`

## 🎯 Recommended Workflow

### Phase 1: Test with Small Datasets (Day 1)

```bash
python slurm_submit.py \
  --dataset annthyroid-unsupervised-ad \
  --dataset breast-cancer-unsupervised-ad \
  --dataset dfki-artificial-3000-unsupervised-ad \
  --max-concurrent 3
```

**Expected completion**: ~12-24 hours

### Phase 2: Medium Datasets (Day 2)

After verifying Phase 1 works:

```bash
python slurm_submit.py \
  --dataset pen-local-unsupervised-ad \
  --dataset pen-global-unsupervised-ad \
  --dataset PenDigits_withoutdupl_norm_v01 \
  --dataset mammography \
  --max-concurrent 4
```

**Expected completion**: ~24-36 hours

### Phase 3: Large Datasets (Day 3-5)

```bash
python slurm_submit.py \
  --dataset creditcard \
  --dataset kdd99-unsupervised-ad \
  --max-concurrent 2
```

**Expected completion**: 2-3 days

### OR: Submit Everything at Once

```bash
python slurm_submit.py --max-concurrent 4
```

Jobs run in priority order (small → large).

## 📈 Monitoring

### Check Status

```bash
# Your jobs
squeue -u $USER

# Detailed view
scontrol show job <job_id>
```

### View Live Logs

```bash
# Live output
tail -f slurm_logs/annthyroid-unsupervised-ad_fastlof_live.log

# Watch for errors
tail -f slurm_logs/annthyroid-unsupervised-ad_fastlof_*.err
```

### Resume After Interruption

```bash
# Press Ctrl+C to stop the monitoring script
# Then later:
python slurm_submit.py --resume
```

## ⚠️ Common Issues & Fixes

### Issue: "Module not found"

**Fix**: Activate your Python environment before running:
```bash
module load python/3.9  # or your module
# or
source /path/to/venv/bin/activate
```

### Issue: "Out of memory"

**Fix**: Edit `slurm_config.yaml` and increase memory:
```yaml
datasets:
  your-dataset:
    memory: "256G"  # Increase this
```

### Issue: Job takes forever

**Fix**: Check thread limiting in output logs. Should see:
```
export OMP_NUM_THREADS=5
```

### Issue: Too many jobs, getting errors

**Fix**: Reduce concurrent limit:
```bash
python slurm_submit.py --max-concurrent 2
```

## 🎓 Pro Tips

1. **Start small**: Test with 1-2 datasets first
2. **Use tmux/screen**: Keep monitoring script running
   ```bash
   tmux new -s fastlof
   python slurm_submit.py
   # Ctrl+B then D to detach
   # tmux attach -t fastlof to reattach
   ```
3. **Check partition load**:
   ```bash
   sinfo -p cpu
   ```
4. **Cancel all your jobs**:
   ```bash
   scancel -u $USER
   ```

## 📞 Need Help?

1. Check logs in `slurm_logs/`
2. Review `SLURM_README.md` for detailed docs
3. Check SLURM cluster documentation

## 🎉 Success Indicators

You'll know it's working when you see:

```
✓ Submitted annthyroid-unsupervised-ad: Job ID 12345
✓ Submitted breast-cancer-unsupervised-ad: Job ID 12346

================================================================================
Job Status - 2025-12-08 14:30:00
================================================================================
Running:   4
Completed: 0
Failed:    0
Pending:   8
...
```

Results will appear in:
```
results/<dataset>/fastlof_experiments/
```

## ⏱️ Total Expected Runtime

| Phase | Datasets | Time | Strategy |
|-------|----------|------|----------|
| Small | 3 datasets | ~24h | Run first to test |
| Medium | 7 datasets | ~36-48h | Run in parallel (4 at a time) |
| Large | 2 datasets | ~60-72h | Run last, maybe separately |
| **Total** | **12 datasets** | **~3-5 days** | **With 4 concurrent jobs** |

Good luck! 🚀
