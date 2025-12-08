# Experiment Scripts

This folder contains organized scripts for running LOF experiments locally or on SLURM clusters.

## 🚀 Quick Start for SLURM Cluster

**For cluster submission, see:**
- **[SLURM_QUICKSTART.md](SLURM_QUICKSTART.md)** - 5-minute setup guide
- **[SLURM_README.md](SLURM_README.md)** - Complete documentation

```bash
# On cluster: Submit all FastLOF experiments
python slurm_submit.py

# Check progress
python check_results.py
```

## Installation

Before running experiments, install required dependencies:

```bash
pip install -r requirements.txt
```

Required packages: numpy, pandas, matplotlib, scikit-learn, scipy, pyod, numba

## Structure

Each dataset has its own subfolder with two experiment scripts:
```
experiment_scripts/
├── <dataset-name>/
│   ├── run_original_lof.py    # Experiment A: Original LOF vs Single-k LOF
│   └── run_fastlof.py          # Experiment B: FastLOF vs Standard LOF
├── ...
```

## Datasets

| Dataset | Folder Name | Format | Notes |
|---------|------------|--------|-------|
| Breast Cancer | `breast-cancer-unsupervised-ad` | .csv | |
| Satellite | `satellite-unsupervised-ad` | .csv | |
| Annthyroid | `annthyroid-unsupervised-ad` | .csv | |
| Credit Card | `creditcard` | .csv | Large - uses 10% sampling |
| DFKI Artificial | `dfki-artificial-3000-unsupervised-ad` | .csv | |
| InternetAds | `InternetAds_norm_02_v01` | .arff | |
| KDD99 | `kdd99-unsupervised-ad` | .csv | Large - uses 5% sampling |
| Mammography | `mammography` | .mat | |
| Pen Global | `pen-global-unsupervised-ad` | .csv | |
| Pen Local | `pen-local-unsupervised-ad` | .csv | |
| PenDigits | `PenDigits_withoutdupl_norm_v01` | .arff | |
| Shuttle | `shuttle-unsupervised-ad` | .csv | Large - uses 20% sampling |

## Running Scripts Locally

From the project root directory:

```bash
# Example: Run original LOF experiment on breast cancer dataset
python experiment_scripts/breast-cancer-unsupervised-ad/run_original_lof.py

# Example: Run FastLOF experiment on satellite dataset
python experiment_scripts/satellite-unsupervised-ad/run_fastlof.py
```

## Script Features

Each script:
- **Automatically imports** from `experiments.py` module
- **Sets matplotlib backend** to 'Agg' (non-interactive, cluster-compatible)
- **Loads dataset** with appropriate parameters
- **Runs experiment** with predefined configurations
- **Saves results** automatically:
  - CSV files → `results/<dataset>/lof_experiments/` or `fastlof_experiments/`
  - PNG plots → Same directory as CSV files
- **Prints progress** to stdout (captured in cluster `.out` files)

## Customizing Parameters

You can edit parameters directly in each script:

### Original LOF Scripts
- `K_MIN`, `K_MAX`, `K_STEP`: K-value range
- `N_RUNS`: Number of repetitions
- `FRACTION`: Dataset sampling fraction (for large datasets)

### FastLOF Scripts
- `K_MIN`, `K_MAX`, `K_STEP`: K-value range
- `THRESHOLD`: FastLOF threshold parameter
- `CHUNK_SIZES`: List of chunk sizes to test
- `N_RUNS`: Number of repetitions
- `FRACTION`: Dataset sampling fraction (for large datasets)

## SLURM Cluster Submission

For automated batch job submission on SLURM clusters:

### Files
- `slurm_config.yaml` - Resource configuration per dataset
- `slurm_template.sh` - SLURM job script template
- `slurm_submit.py` - Automated submission manager
- `submit_jobs.sh` - Interactive wrapper script
- `check_results.py` - Check experiment completion status

### Quick Commands

```bash
# Generate and test (dry run)
python slurm_submit.py --dry-run

# Submit all FastLOF experiments (4 concurrent jobs)
python slurm_submit.py

# Submit specific datasets only
python slurm_submit.py --dataset annthyroid-unsupervised-ad --dataset breast-cancer-unsupervised-ad

# Resume after interruption
python slurm_submit.py --resume

# Check completion status
python check_results.py
```

### Features
- ✅ Automatic job generation for all 12 datasets
- ✅ Resource allocation based on dataset size
- ✅ Batch submission with configurable concurrency (default: 4 jobs)
- ✅ Auto-queuing: submits next job when slots become available
- ✅ Thread limiting (5 threads) to prevent overhead
- ✅ State management: resume anytime with `--resume`
- ✅ Live monitoring with status updates every 5 minutes
- ✅ Email notifications (optional)

### Resource Allocation
| Dataset Size | CPUs | Memory | Time | Partition |
|--------------|------|--------|------|-----------|
| Small (<10k) | 16 | 32GB | 24h | `cpu` |
| Medium (10k-100k) | 20-24 | 48-64GB | 36-48h | `cpu` |
| Large (>100k) | 32 | 128-256GB | 72h | `cpu` |

**Estimated total runtime**: 3-5 days with 4 concurrent jobs

See **[SLURM_README.md](SLURM_README.md)** for complete documentation.

## Notes

- Large datasets (creditcard, kdd99, shuttle) use sampling by default
- All `plt.show()` calls have been removed from `experiments.py`
- Results are saved relative to the project root directory
- Scripts use relative imports, so they must be run from the project root
- SLURM system designed for FastLOF experiments (can be adapted for Original LOF)
