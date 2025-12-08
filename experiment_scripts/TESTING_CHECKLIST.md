# Testing Checklist - Before Cluster Submission

Follow these steps to verify everything works before submitting to the cluster.

## ✅ Pre-Test Checklist

### 1. Verify Setup
```bash
# From project root directory
cd "e:\Work\Thesis\Final FastLOF"

# Run setup verification
python experiment_scripts/test_setup.py
```

**Expected output:**
- ✓ Successfully imported experiments module
- ✓ All dependencies available
- ✓ Data directory exists with 12 datasets
- ✓ Results directory ready

---

## 🧪 Local Testing

### 2. Test Small Dataset (RECOMMENDED: Start Here)

**Test Original LOF:**
```bash
python experiment_scripts/dfki-artificial-3000-unsupervised-ad/run_original_lof.py
```

**What to watch for:**
- [ ] Script starts without import errors
- [ ] Dataset loads successfully
- [ ] Prints "Dataset shape: ..."
- [ ] Shows progress: "Run 1/10", "Run 2/10", etc.
- [ ] No matplotlib window appears (headless mode)
- [ ] Completes with "Experiment Complete!"

**Expected results location:**
```
results/dfki-artificial-3000-unsupervised-ad/lof_experiments/k10-50/
├── original_lof_results.csv
├── original_lof_summary.png
└── original_lof_scatter.png
```

**Test FastLOF:**
```bash
python experiment_scripts/dfki-artificial-3000-unsupervised-ad/run_fastlof.py
```

**Expected results location:**
```
results/dfki-artificial-3000-unsupervised-ad/fastlof_experiments/k10-50_t1.1/
├── fastlof_results.csv
├── fastlof_summary.png
├── fastlof_detail.png
├── fastlof_scatter.png
└── fastlof_timing.png
```

---

### 3. Verify Results

**Check CSV files:**
```bash
# View CSV headers
head results/dfki-artificial-3000-unsupervised-ad/lof_experiments/k10-50/original_lof_results.csv
```

**Check PNG files exist:**
```bash
ls -lh results/dfki-artificial-3000-unsupervised-ad/lof_experiments/k10-50/*.png
ls -lh results/dfki-artificial-3000-unsupervised-ad/fastlof_experiments/k10-50_t1.1/*.png
```

**Verification checklist:**
- [ ] CSV files are not empty
- [ ] PNG files are not corrupt (can open them)
- [ ] File sizes are reasonable (CSVs: few KB, PNGs: 50-500 KB)

---

### 4. Test Medium Dataset (Optional)

If small dataset worked, test a medium one:

```bash
python experiment_scripts/breast-cancer-unsupervised-ad/run_original_lof.py
```

---

## 🚨 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'experiments'"
**Solution:** Make sure you're running from the project root directory:
```bash
cd "e:\Work\Thesis\Final FastLOF"
python experiment_scripts/<dataset>/run_original_lof.py
```

### Issue: "FileNotFoundError: Dataset file not found"
**Solution:** Check that data files exist in `data/` folder:
```bash
ls data/*.csv
ls data/*.mat
ls data/*.arff
```

### Issue: matplotlib displays a window / script hangs
**Solution:** Verify `matplotlib.use('Agg')` is in the script before `import matplotlib.pyplot`

### Issue: Results folder not created
**Solution:** The script should auto-create it. If not, manually create:
```bash
mkdir results
```

---

## ✅ Test Success Criteria

Before moving to cluster submission, verify:

- [ ] `test_setup.py` passes all checks
- [ ] At least one dataset runs successfully (Original LOF)
- [ ] At least one dataset runs successfully (FastLOF)
- [ ] Results CSV files are created and contain data
- [ ] Results PNG files are created and can be opened
- [ ] No matplotlib windows appear during execution
- [ ] Script output is clear and informative

---

## 📋 Parameter Customization Test (Optional)

If you want to test with different parameters before cluster submission:

**Example: Reduce runs for faster testing**

Edit any script (e.g., `run_original_lof.py`):
```python
N_RUNS = 3  # Changed from 10 to 3 for quick test
```

**Example: Test different k-range**
```python
K_MIN = 20
K_MAX = 30
K_STEP = 5
```

---

## 🚀 Ready for Cluster?

Once all checks pass:
1. ✅ Restore any test parameters to desired values
2. ✅ Review SUMMARY.md for dataset categories
3. ✅ Decide which datasets to run on cluster
4. ✅ Prepare cluster job scripts (next step)

---

## 📝 Notes for Cluster Submission

When you create cluster job scripts, remember:

**Working directory:**
```bash
cd /path/to/Final\ FastLOF  # On cluster
```

**Python environment:**
```bash
# Make sure same packages are installed on cluster
pip install numpy pandas matplotlib scikit-learn scipy pyod numba
```

**Memory requirements (estimated):**
- Small datasets: 4-8 GB
- Medium datasets: 8-16 GB
- Large datasets (with sampling): 8-16 GB
- Large datasets (full): 32+ GB

**Time requirements (estimated per experiment):**
- Small datasets: 10-30 minutes
- Medium datasets: 30-60 minutes
- Large datasets: 1-3 hours

---

## 🆘 Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify all dependencies are installed
3. Ensure you're in the correct directory
4. Check file permissions
5. Review experiment_scripts/README.md

Good luck! 🎉
