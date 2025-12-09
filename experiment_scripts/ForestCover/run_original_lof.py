"""
Original LOF Experiment - ForestCover Dataset
==============================================
Runs Experiment A (baseline: sklearn's LOF) on ForestCover dataset.
"""

import sys
import os

# Add parent directory to path to import experiments module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

from experiments import load_dataset, run_original_lof_experiment

# Dataset configuration
DATASET_FILEPATH = "data/ForestCover.mat"
DATASET_NAME = "ForestCover"

# Experiment parameters
K_VALUES = [10, 20, 30, 40, 50]
N_RUNS = 10

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"Original LOF Experiment - {DATASET_NAME}")
    print(f"{'='*80}\n")
    
    # Load dataset
    print(f"Loading dataset: {DATASET_FILEPATH}")
    X, y = load_dataset(DATASET_FILEPATH)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of anomalies: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
    
    # Run Experiment A
    results = run_original_lof_experiment(
        X, y,
        k_values=K_VALUES,
        n_runs=N_RUNS,
        dataset_filepath=DATASET_FILEPATH
    )
    
    print(f"\n{'='*80}")
    print(f"Experiment Complete!")
    print(f"{'='*80}\n")
    
    sys.exit(0)

