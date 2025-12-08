# Experiment Scripts - Organization Summary

## ✅ Setup Complete

Created organized experiment scripts for **12 datasets** with **24 total scripts** (2 per dataset).

## 📁 Folder Structure

```
experiment_scripts/
├── README.md                          # Documentation
├── DATASET_LIST.txt                   # Quick reference list
├── test_setup.py                      # Setup verification script
├── SUMMARY.md                         # This file
│
├── annthyroid-unsupervised-ad/
│   ├── run_original_lof.py           # Exp A: k=10-50, n=10
│   └── run_fastlof.py                # Exp B: k=10-50, t=1.1, n=10
│
├── breast-cancer-unsupervised-ad/
│   ├── run_original_lof.py           # Exp A: k=10-50, n=10
│   └── run_fastlof.py                # Exp B: k=10-50, t=1.1, n=10
│
├── creditcard/                        # ⚠️ Large dataset
│   ├── run_original_lof.py           # Exp A: fraction=0.1, n=5
│   └── run_fastlof.py                # Exp B: fraction=0.1, n=5
│
├── dfki-artificial-3000-unsupervised-ad/
│   ├── run_original_lof.py           # Exp A: k=10-50, n=10
│   └── run_fastlof.py                # Exp B: k=10-50, t=1.1, n=10
│
├── InternetAds_norm_02_v01/          # .arff format
│   ├── run_original_lof.py           # Exp A: k=10-50, n=10
│   └── run_fastlof.py                # Exp B: k=10-50, t=1.1, n=10
│
├── kdd99-unsupervised-ad/            # ⚠️ Large dataset
│   ├── run_original_lof.py           # Exp A: fraction=0.05, n=5
│   └── run_fastlof.py                # Exp B: fraction=0.05, n=5
│
├── mammography/                       # .mat format
│   ├── run_original_lof.py           # Exp A: k=10-50, n=10
│   └── run_fastlof.py                # Exp B: k=10-50, t=1.1, n=10
│
├── pen-global-unsupervised-ad/
│   ├── run_original_lof.py           # Exp A: k=10-50, n=10
│   └── run_fastlof.py                # Exp B: k=10-50, t=1.1, n=10
│
├── pen-local-unsupervised-ad/
│   ├── run_original_lof.py           # Exp A: k=10-50, n=10
│   └── run_fastlof.py                # Exp B: k=10-50, t=1.1, n=10
│
├── PenDigits_withoutdupl_norm_v01/   # .arff format
│   ├── run_original_lof.py           # Exp A: k=10-50, n=10
│   └── run_fastlof.py                # Exp B: k=10-50, t=1.1, n=10
│
├── satellite-unsupervised-ad/
│   ├── run_original_lof.py           # Exp A: k=10-50, n=10
│   └── run_fastlof.py                # Exp B: k=10-50, t=1.1, n=10
│
└── shuttle-unsupervised-ad/          # ⚠️ Large dataset
    ├── run_original_lof.py           # Exp A: fraction=0.2, n=10
    └── run_fastlof.py                # Exp B: fraction=0.2, n=10
```

## 🔧 Key Features

### Each Script Includes:
- ✅ Correct path setup (`sys.path.insert()`)
- ✅ Non-interactive matplotlib backend (`matplotlib.use('Agg')`)
- ✅ Imports from `experiments.py`
- ✅ Dataset-specific parameters
- ✅ Proper error handling
- ✅ Progress printing (captured by cluster)
- ✅ Automatic result saving

### Results Saved To:
- **CSV**: `results/<dataset>/lof_experiments/` or `fastlof_experiments/`
- **PNG**: Same directory as CSV files
- **Stdout**: Captured in cluster `.out` files

## 🧪 Testing Locally

Before submitting to cluster, test locally:

```bash
# 1. Verify setup
python experiment_scripts/test_setup.py

# 2. Test a small dataset (recommended: dfki-artificial-3000)
python experiment_scripts/dfki-artificial-3000-unsupervised-ad/run_original_lof.py

# 3. Check results
ls results/dfki-artificial-3000-unsupervised-ad/
```

## 🎯 Recommended Test Order

1. **Small dataset first**: `dfki-artificial-3000-unsupervised-ad` (~3000 samples)
2. **Medium dataset**: `breast-cancer-unsupervised-ad` or `satellite-unsupervised-ad`
3. **Large dataset**: Test with sampling (creditcard, kdd99, shuttle)

## 📊 Dataset Categories

### Small Datasets (Good for Testing)
- dfki-artificial-3000-unsupervised-ad
- breast-cancer-unsupervised-ad
- satellite-unsupervised-ad
- annthyroid-unsupervised-ad

### Medium Datasets
- pen-global-unsupervised-ad
- pen-local-unsupervised-ad
- mammography
- InternetAds_norm_02_v01
- PenDigits_withoutdupl_norm_v01

### Large Datasets (Use Sampling)
- creditcard (fraction=0.1)
- kdd99-unsupervised-ad (fraction=0.05)
- shuttle-unsupervised-ad (fraction=0.2)

## ⚙️ Parameters Summary

### Original LOF Experiments
- K range: 10-50 (step=10)
- Runs: 10 (5 for large datasets)
- Measures: AUC, Precision@k, Runtime

### FastLOF Experiments
- K range: 10-50 (step=10)
- Threshold: 1.1
- Chunk sizes: [100, 500, 1000, 2000, 5000]
- Runs: 10 (5 for large datasets)
- Measures: AUC, Precision@k, Speedup, Runtime

## 🚀 Next Steps

1. ✅ **Test locally** with a small dataset
2. ⏳ **Verify results** are saved correctly
3. ⏳ **Create cluster job scripts** (SLURM/PBS)
4. ⏳ **Submit jobs** in parallel
5. ⏳ **Collect results**

## 📝 Notes

- All scripts use **relative paths** from project root
- **matplotlib backend** set to 'Agg' (no GUI needed)
- **Results auto-save** to appropriate folders
- **Stdout captured** by cluster scheduler
- **No plt.show()** calls in experiments.py
