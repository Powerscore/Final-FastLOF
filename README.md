# FastLOF: Fast Local Outlier Factor Implementation

A high-performance implementation and comprehensive benchmarking framework for the Local Outlier Factor (LOF) algorithm for anomaly detection.

## Overview

This repository contains:
- **FastLOF**: An optimized implementation of the LOF algorithm with chunking for large datasets
- **Original LOF**: Multi-k range LOF implementation for robust outlier detection
- **Comprehensive Experiments**: Benchmarking framework comparing different LOF implementations
- **Multiple Datasets**: Collection of anomaly detection benchmark datasets

## Components

### Core Implementations
- `fastlof.py` - FastLOF implementation with performance optimizations
- `original_lof.py` - Original LOF algorithm with multi-k support
- `experiments.py` - Experiment framework for benchmarking
- `single_class.py` - Single-class LOF utilities

### Experiment Scripts
The `experiment_scripts/` directory contains ready-to-run scripts for each dataset:
- Original LOF experiments (`run_original_lof.py`)
- FastLOF experiments (`run_fastlof.py`)

See `experiment_scripts/README.md` for details on running experiments.

### Datasets
Multiple benchmark datasets in the `data/` directory for testing and evaluation.

### Results
Experimental results are stored in `results/` directory:
- Performance metrics (CSV files tracked with Git LFS)
- Visualization plots (PNG files tracked with Git LFS)
- Comparative analyses

## Features

- **Multi-format Support**: CSV, MAT, and ARFF dataset formats
- **Automatic Label Detection**: Smart detection of anomaly labels
- **Comprehensive Metrics**: AUC, Precision@k, and timing analysis
- **Visualization Suite**: Automated plotting and comparison charts
- **Git LFS Integration**: Efficient storage of large result files

## Installation

### Requirements

See `requirements.txt` for all dependencies:
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- pyod
- numba

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

```python
from experiments import run_original_lof_experiment, load_dataset

# Load dataset
X, y = load_dataset("data/your-dataset.csv")

# Run Original LOF experiment
results = run_original_lof_experiment(
    X, y,
    k_min=10,
    k_max=50,
    n_runs=5,
    contamination=0.1
)
```

Or use the ready-made experiment scripts:
```bash
python experiment_scripts/dataset-name/run_original_lof.py
python experiment_scripts/dataset-name/run_fastlof.py
```

## Git LFS

This repository uses Git LFS for large files:
- `*.png` - Visualization plots
- `*.csv` - Dataset and result files

Make sure Git LFS is installed before cloning:
```bash
git lfs install
```

## License

Research project for thesis work.
