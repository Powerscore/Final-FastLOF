"""
FastLOF Experiment - Credit Card Dataset
=========================================
Runs Experiment B (FastLOF vs Standard LOF) on creditcard dataset.
NOTE: This is a large dataset, consider using sampling (fraction parameter).
"""

import sys
import os

# Add parent directory to path to import experiments module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

from experiments import load_dataset, run_fastlof_experiment

# Dataset configuration
DATASET_FILEPATH = "data/creditcard.csv"
DATASET_NAME = "creditcard"

# Experiment parameters
K_MIN = 10
K_MAX = 50
K_STEP = 10
THRESHOLD = 1.1
CHUNK_SIZES = [500, 1000, 2000, 5000, 10000]  # Larger chunks for large dataset
N_RUNS = 5  # Reduced for large dataset
FRACTION = 0.1  # Use 10% of data (adjust as needed)

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"FastLOF Experiment - {DATASET_NAME}")
    print(f"{'='*80}\n")
    
    # Load dataset with sampling
    print(f"Loading dataset: {DATASET_FILEPATH} (fraction={FRACTION})")
    X, y = load_dataset(DATASET_FILEPATH, fraction=FRACTION)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of anomalies: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
    
    # Run Experiment B
    results = run_fastlof_experiment(
        X, y,
        k_min=K_MIN,
        k_max=K_MAX,
        step=K_STEP,
        threshold=THRESHOLD,
        chunk_sizes=CHUNK_SIZES,
        n_runs=N_RUNS,
        dataset_filepath=DATASET_FILEPATH
    )
    
    print(f"\n{'='*80}")
    print(f"Experiment Complete!")
    print(f"{'='*80}\n")
