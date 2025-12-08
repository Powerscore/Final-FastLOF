"""
Quick Test Script - Verify Experiment Setup
============================================
Tests that imports work correctly and paths are set up properly.
Run this before submitting cluster jobs to catch any issues.
"""

import sys
import os

# Add parent directory to path (same as in experiment scripts)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

print("=" * 80)
print("Testing Experiment Setup")
print("=" * 80)

# Test 1: Import experiments module
print("\n[1/4] Testing experiments module import...")
try:
    from experiments import load_dataset, run_original_lof_experiment, run_fastlof_experiment
    print("✓ Successfully imported experiments module")
except Exception as e:
    print(f"✗ Failed to import experiments module: {e}")
    sys.exit(1)

# Test 2: Import dependencies
print("\n[2/4] Testing dependencies...")
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from pyod.models.lof import LOF
    from fastlof import FastLOF
    from original_lof import OriginalLOF
    print("✓ All dependencies available")
except Exception as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

# Test 3: Check data directory
print("\n[3/4] Checking data directory...")
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
if os.path.exists(data_dir):
    datasets = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.mat', '.arff'))]
    print(f"✓ Data directory exists with {len(datasets)} datasets")
else:
    print(f"✗ Data directory not found at {data_dir}")
    sys.exit(1)

# Test 4: Check results directory
print("\n[4/4] Checking results directory...")
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
if not os.path.exists(results_dir):
    print(f"  Creating results directory at {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
print("✓ Results directory ready")

print("\n" + "=" * 80)
print("All tests passed! Ready to run experiments.")
print("=" * 80)
print("\nTo test a small dataset, try:")
print("  python experiment_scripts/dfki-artificial-3000-unsupervised-ad/run_original_lof.py")
print("\n")
