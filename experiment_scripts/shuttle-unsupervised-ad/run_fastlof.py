"""
FastLOF Experiment - Shuttle Dataset
=====================================
Runs Experiment B (FastLOF vs Standard LOF) on shuttle-unsupervised-ad dataset.
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
DATASET_FILEPATH = "data/shuttle-unsupervised-ad.csv"
DATASET_NAME = "shuttle-unsupervised-ad"

# Experiment parameters
K_VALUES = [10, 20, 30, 40, 50]
THRESHOLDS = [0, 1.0, 1.01, 1.1, 1.2]  # List of thresholds to test
MIN_CHUNK_SIZE = 500
MAX_CHUNK_SIZE = 10000
CHUNK_INTERVAL = 500
N_RUNS = 10
FRACTION = 1

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"FastLOF Experiment - {DATASET_NAME}")
    print(f"{'='*80}\n")
    
    # Load dataset with sampling
    print(f"Loading dataset: {DATASET_FILEPATH} (fraction={FRACTION})")
    X, y = load_dataset(DATASET_FILEPATH, fraction=FRACTION)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of anomalies: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
    
    # Run Experiment B for each threshold
    print(f"\nTesting thresholds: {THRESHOLDS}")
    for threshold in THRESHOLDS:
        print(f"\n{'-'*80}")
        print(f"Running with threshold = {threshold}")
        print(f"{'-'*80}\n")
        
        results = run_fastlof_experiment(
            X, y,
            k_values=K_VALUES,
            min_chunk_size=MIN_CHUNK_SIZE,
            max_chunk_size=MAX_CHUNK_SIZE,
            chunk_interval=CHUNK_INTERVAL,
            threshold=threshold,
            n_runs=N_RUNS,
            dataset_filepath=DATASET_FILEPATH
        )
    
    print(f"\n{'='*80}")
    print(f"All Experiments Complete!")
    print(f"{'='*80}\n")
    
    sys.exit(0)
