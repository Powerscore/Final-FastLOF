# Experiment Scripts

This folder contains organized scripts for running LOF experiments on the cluster.

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

## Notes

- Large datasets (creditcard, kdd99, shuttle) use sampling by default
- All `plt.show()` calls have been removed from `experiments.py`
- Results are saved relative to the project root directory
- Scripts use relative imports, so they must be run from the project root
