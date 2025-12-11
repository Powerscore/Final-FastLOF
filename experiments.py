"""
LOF Experiment Framework
========================

This module provides a comprehensive framework for benchmarking and comparing
LOF (Local Outlier Factor) implementations:

Experiment A: Original LOF vs single-k LOF
    - Compares multi-k range original LOF algorithm implementation against pyod single-k LOF
    - Evaluates both quality (AUC, Precision@k) and runtime performance
    
Experiment B: FastLOF vs standard LOF  
    - Tests FastLOF implementation
    - Examines speed-accuracy tradeoffs across different chunk sizes
    - Aggregates results across k-value ranges

Key Features:
    - Multi-format dataset loading (.csv, .mat, .arff)
    - Automatic label detection and normalization
    - Comprehensive visualization suite
    - Statistical aggregation across multiple runs
    - Detailed timing breakdowns
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import math
import os
import scipy.io
from scipy.io import arff
import copy
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from pyod.models.lof import LOF
from fastlof import FastLOF
from original_lof import OriginalLOF


# Random seed for reproducible sampling
RANDOM_SEED = 42


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(filepath, fraction=1.0):
    """
    Load dataset from various file formats (.csv, .mat, .arff).
    
    Automatically detects and converts anomaly labels from various formats
    (text, binary, etc.). Prints preview of first 5 rows when possible.
    
    Parameters
    ----------
    filepath : str
        Path to dataset file
    fraction : float, optional (default=1.0)
        Fraction of dataset to load (0.0 to 1.0).
        Uses random sampling with fixed seed for reproducibility.
    
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,) or None
        Binary labels (1=anomaly, 0=normal) if available, else None
        
    Raises
    ------
    ValueError
        If file format unsupported or data cannot be parsed
    FileNotFoundError
        If filepath does not exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    if ext not in ['.csv', '.mat', '.arff']:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported formats are .csv, .mat, .arff."
        )
    
    X = None
    y = None
    
    # Dataset preview (first 5 rows)
    _print_dataset_preview(filepath, ext)
    
    # Load based on file type
    if ext == '.csv':
        X, y = _load_csv(filepath)
    elif ext == '.mat':
        X, y = _load_mat(filepath)
    elif ext == '.arff':
        X, y = _load_arff(filepath)
    
    # Validate loaded data
    if X is None or X.size == 0:
        raise ValueError("Loaded data is empty or could not be processed.")
    
    # Apply fraction sampling if requested
    if fraction < 1.0:
        X, y = _apply_fraction_sampling(X, y, fraction)
    
    return X, y


def _print_dataset_preview(filepath, ext):
    """Print first 5 rows of dataset for inspection."""
    try:
        if ext == '.csv':
            preview_df = pd.read_csv(filepath, nrows=5)
            print("\n--- Dataset preview (first 5 rows) ---")
            print(preview_df.head())
            print("--------------------------------------")
        elif ext == '.arff':
            preview_data, preview_meta = arff.loadarff(filepath)
            preview_df = pd.DataFrame(
                preview_data.tolist(), 
                columns=preview_data.dtype.names
            )
            print("\n--- Dataset preview (first 5 rows) ---")
            print(preview_df.head())
            print("--------------------------------------")
    except Exception as e:
        print(f"(Preview skipped due to error: {e})")


def _label_from_value(value):
    """
    Convert textual/numeric labels to binary anomaly flags.
    
    Parameters
    ----------
    value : str, bytes, int, float
        Label value from dataset
        
    Returns
    -------
    int
        1 for anomaly, 0 for normal
    """
    # Handle numeric values directly
    if isinstance(value, (int, float, np.integer, np.floating)):
        # If already 0 or 1, return as-is
        if value in [0, 0.0]:
            return 0
        elif value in [1, 1.0]:
            return 1
        # Otherwise convert to string for text matching
    
    if isinstance(value, (bytes, bytearray)):
        value = value.decode('utf-8')
    
    value_str = str(value).strip().strip('"').strip("'").lower()
    
    anomaly_labels = {
        '1', '1.0', 'anomaly', 'attack', 'outlier', 'abnormal', 'yes', 'true',
        'o', 'outliers', 'anomalous', 'attack.', 'anomaly.', 'outlier.', 'abnormal.'
    }
    
    return 1 if value_str in anomaly_labels else 0


def _load_csv(filepath):
    """
    Load CSV file using pandas for robust parsing.
    Handles quoted fields, mixed types, and label detection.
    Auto-detects whether first row is a header or data.
    """
    try:
        # Auto-detect if first row is a header
        # Read first line to check if it's numeric data or column names
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        
        # Try to parse first row values as floats (excluding last column which might be labels)
        first_row_values = first_line.split(',')
        has_header = False
        
        # Check if first row looks like column names (non-numeric values)
        for val in first_row_values[:-1]:  # Check all but last column
            try:
                float(val.strip().strip('"').strip("'"))
            except (ValueError, AttributeError):
                # Can't convert to float, likely a header
                has_header = True
                break
        
        # Read CSV with appropriate header setting
        df = pd.read_csv(filepath, header=0 if has_header else None)
        
        # Check if last column might be labels
        last_col = df.iloc[:, -1]
        last_col_name = str(df.columns[-1]).lower() if has_header else ""
        
        # Label column indicators
        label_keywords = ['class', 'label', 'target', 'outlier', 'anomaly', 'ground_truth', 'gt']
        is_label_name = any(keyword in last_col_name for keyword in label_keywords)
        
        # Check if column is binary (only 0/1 or -1/1)
        unique_vals = last_col.unique()
        is_binary = (len(unique_vals) <= 2 and 
                    all(val in [0, 1, -1, 0.0, 1.0, -1.0] for val in unique_vals))
        
        if (last_col.dtype == 'object' or 
            not pd.api.types.is_numeric_dtype(last_col) or 
            is_label_name or 
            is_binary):
            # Last column is labels
            X = df.iloc[:, :-1].values.astype(float)
            if pd.api.types.is_numeric_dtype(last_col):
                # Numeric labels (0/1 or -1/1)
                y = last_col.values.astype(np.int32)
                # Convert -1/1 to 0/1 if needed
                if -1 in y:
                    y = np.where(y == -1, 0, 1)
            else:
                # Non-numeric labels (need conversion)
                y = np.array([_label_from_value(val) for val in last_col], dtype=np.int32)
        else:
            # All numeric, no labels
            X = df.values.astype(float)
            y = None
            
        return X, y
        
    except Exception as e:
        raise Exception(f"Error loading CSV file '{filepath}': {e}")


def _load_mat(filepath):
    """Load MATLAB .mat file, searching for data and label arrays."""
    try:
        mat_data = scipy.io.loadmat(filepath)
        
        # Find data matrix
        X = None
        if 'X' in mat_data and isinstance(mat_data['X'], np.ndarray) and mat_data['X'].ndim == 2:
            X = mat_data['X']
        elif 'data' in mat_data and isinstance(mat_data['data'], np.ndarray) and mat_data['data'].ndim == 2:
            X = mat_data['data']
        else:
            # Search for largest 2D array
            potential_X = None
            max_size = 0
            for key, value in mat_data.items():
                if (isinstance(value, np.ndarray) and 
                    value.ndim == 2 and 
                    value.size > max_size and 
                    not key.startswith('__')):
                    potential_X = value
                    max_size = value.size
            
            if potential_X is not None:
                X = potential_X
            else:
                raise ValueError(
                    f"Could not find a suitable 2D data array in .mat file. "
                    f"Available keys: {list(mat_data.keys())}"
                )
        
        X = X.astype(float)
        
        # Preview .mat data
        try:
            n_cols = X.shape[1]
            col_names = [f"f{i}" for i in range(n_cols)]
            preview_df = pd.DataFrame(X[:5, :], columns=col_names)
            print("\n--- Dataset preview (first 5 rows) [from .mat X] ---")
            print(preview_df.head())
            print("--------------------------------------")
        except Exception as e:
            print(f"(Preview for .mat skipped due to error: {e})")
        
        # Find labels
        y = None
        for label_key in ['y', 'Y', 'labels', 'label']:
            if label_key in mat_data:
                label_arr = mat_data[label_key]
                label_arr = np.ravel(label_arr)
                if label_arr.size == X.shape[0]:
                    y = np.array(
                        [_label_from_value(val) for val in label_arr], 
                        dtype=np.int32
                    )
                    break
        
        return X, y
        
    except NotImplementedError:
        # Try h5py for MATLAB v7.3 files
        try:
            import h5py
            with h5py.File(filepath, 'r') as f:
                # Find data matrix
                X = None
                if 'X' in f.keys():
                    X = np.array(f['X']).T  # h5py transposes data
                elif 'data' in f.keys():
                    X = np.array(f['data']).T
                else:
                    # Search for largest 2D array
                    potential_X = None
                    max_size = 0
                    for key in f.keys():
                        if not key.startswith('__'):
                            value = f[key]
                            if isinstance(value, h5py.Dataset) and len(value.shape) == 2:
                                if value.size > max_size:
                                    potential_X = np.array(value).T
                                    max_size = value.size
                    
                    if potential_X is not None:
                        X = potential_X
                    else:
                        raise ValueError(
                            f"Could not find a suitable 2D data array in .mat file. "
                            f"Available keys: {list(f.keys())}"
                        )
                
                X = X.astype(float)
                
                # Preview .mat data
                try:
                    n_cols = X.shape[1]
                    col_names = [f"f{i}" for i in range(n_cols)]
                    preview_df = pd.DataFrame(X[:5, :], columns=col_names)
                    print("\n--- Dataset preview (first 5 rows) [from .mat X using h5py] ---")
                    print(preview_df.head())
                    print("--------------------------------------")
                except Exception as e:
                    print(f"(Preview for .mat skipped due to error: {e})")
                
                # Find labels
                y = None
                for label_key in ['y', 'Y', 'labels', 'label']:
                    if label_key in f.keys():
                        label_arr = np.array(f[label_key])
                        # Handle different shapes
                        if label_arr.ndim == 2:
                            if label_arr.shape[0] == 1:
                                label_arr = label_arr[0]
                            elif label_arr.shape[1] == 1:
                                label_arr = label_arr[:, 0]
                            else:
                                label_arr = label_arr.flatten()
                        else:
                            label_arr = label_arr.flatten()
                        
                        if label_arr.size == X.shape[0]:
                            y = np.array(
                                [_label_from_value(val) for val in label_arr], 
                                dtype=np.int32
                            )
                            break
                
                return X, y
        except ImportError:
            raise Exception(
                f"Error loading .mat file '{filepath}': "
                "This is a MATLAB v7.3 file which requires h5py. "
                "Please install h5py: pip install h5py"
            )
        except Exception as e:
            raise Exception(f"Error loading .mat file '{filepath}' with h5py: {e}")
    except Exception as e:
        raise Exception(f"Error loading .mat file '{filepath}': {e}")


def _load_arff(filepath):
    """Load ARFF file, separating numeric features from labels."""
    try:
        data, meta = arff.loadarff(filepath)
        numeric_cols = []
        labels = None
        
        for col_name in data.dtype.names:
            col_data = data[col_name]
            if np.issubdtype(col_data.dtype, np.number):
                numeric_cols.append(col_data)
            else:
                # Non-numeric column assumed to be labels
                if labels is None:
                    labels = np.array(
                        [_label_from_value(val) for val in col_data], 
                        dtype=np.int32
                    )
                else:
                    print(
                        f"Warning: Multiple non-numeric columns detected. "
                        f"Using '{col_name}' as additional label information."
                    )
                    labels = np.array(
                        [_label_from_value(val) for val in col_data], 
                        dtype=np.int32
                    )
        
        if not numeric_cols:
            raise ValueError("No numeric columns found in .arff file.")
        
        X = np.column_stack(numeric_cols).astype(float)
        y = labels if labels is not None and labels.shape[0] == X.shape[0] else None
        
        return X, y
        
    except Exception as e:
        raise Exception(f"Error loading .arff file '{filepath}': {e}")


def _apply_fraction_sampling(X, y, fraction):
    """
    Apply random sampling to reduce dataset size.
    Uses fixed random seed for reproducibility.
    """
    n_samples = int(X.shape[0] * fraction)
    n_samples = max(1, n_samples)  # Ensure at least 1 sample
    
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
    indices = np.sort(indices)  # Sort to maintain some data locality
    
    X = X[indices]
    if y is not None:
        y = y[indices]
    
    print(f"Loaded {fraction*100:.0f}% of dataset: {X.shape[0]} samples (random sampling)")
    
    return X, y


# =============================================================================
# Metrics and Statistics
# =============================================================================

def _true_contamination(y_true):
    """
    Calculate true contamination rate from labels.
    
    Parameters
    ----------
    y_true : ndarray or None
        Binary labels (1 = anomaly, 0 = normal)
        
    Returns
    -------
    float
        Proportion of anomalies in dataset
        
    Raises
    ------
    ValueError
        If labels are None, empty, or contain only one class
    """
    if y_true is None:
        raise ValueError("Labels are required for experiments")
    
    y_true = np.asarray(y_true).ravel()
    
    if y_true.size == 0:
        raise ValueError("Label array is empty")
    
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        raise ValueError("Labels must contain both classes (0 and 1)")
    
    return float(np.mean(y_true))


def _runtime_stats(times):
    """
    Compute statistics over multiple timing measurements.
    
    Parameters
    ----------
    times : list of float
        Runtime measurements in seconds
        
    Returns
    -------
    dict
        Statistics including avg, std, min, max, and raw times
    """
    times = np.array(times)
    return {
        'avg': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'times': times.tolist(),
    }


def _aggregate_metrics(metrics_list):
    """
    Aggregate metrics across multiple runs or configurations.
    
    Parameters
    ----------
    metrics_list : list of dict
        List of metric dictionaries from _compute_anomaly_metrics
        
    Returns
    -------
    dict or None
        Averaged metrics, or None if no valid metrics
    """
    if not metrics_list:
        return None
    
    keys = ['roc_auc', 'pr_auc', 'precision_at_k']
    agg = {}
    
    for key in keys:
        vals = [m[key] for m in metrics_list if m and m.get(key) is not None]
        if vals:
            agg[key] = float(np.mean(vals))
    
    return agg if agg else None


def _compute_anomaly_metrics(y_true, scores):
    """
    Compute ROC-AUC, PR-AUC, and Precision@k using true contamination.
    
    Parameters
    ----------
    y_true : array-like or None
        True binary labels (1=anomaly, 0=normal)
    scores : array-like
        Anomaly scores (higher = more anomalous)
        
    Returns
    -------
    dict or None
        Metrics dictionary with roc_auc, pr_auc, precision_at_k, k_top
        Returns None if labels unavailable or only one class present
    """
    if y_true is None:
        return None
    
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores).ravel()
    
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError(
            "Label array and scores must have the same length for metric evaluation."
        )
    
    if np.unique(y_true).size < 2:
        print("Warning: Only one class present in labels. Skipping metric computation.")
        return None
    
    try:
        roc_auc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        
        # Precision@k using true contamination rate
        contamination_true = _true_contamination(y_true)
        k_top = max(1, int(len(y_true) * contamination_true)) if contamination_true and contamination_true > 0 else 1
        
        sorted_idx = np.argsort(scores)[::-1]
        top_k_idx = sorted_idx[:k_top]
        precision_at_k = np.sum(y_true[top_k_idx]) / k_top
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision_at_k': precision_at_k,
            'k_top': k_top,
        }
    except Exception as e:
        print(f"Warning: Error computing metrics: {e}")
        return None


# =============================================================================
# File Management
# =============================================================================

def _results_dir(dataset_filepath, subfolder, k=None, threshold=None):
    """
    Create results directory with hierarchical structure.
    
    Structure: results/<dataset_name>/<subfolder>/k{k}_t{threshold}/
    
    Parameters
    ----------
    dataset_filepath : str
        Path to original dataset file
    subfolder : str
        Experiment type subfolder (e.g., 'lof_experiments', 'fastlof_experiments')
    k : int, list, or None
        K-value or range. If list, formats as k{min}-{max}
    threshold : float or None
        Threshold parameter (for FastLOF experiments)
        
    Returns
    -------
    str
        Created directory path
    """
    dataset_name = os.path.splitext(os.path.basename(dataset_filepath))[0]
    base = os.path.join(os.getcwd(), 'results', dataset_name, subfolder)
    
    # Build k_threshold folder name
    if k is not None:
        if isinstance(k, (list, tuple)) and len(k) > 1:
            k_str = f"k{min(k)}-{max(k)}"
        elif isinstance(k, (list, tuple)):
            k_str = f"k{k[0]}"
        else:
            k_str = f"k{k}"
        
        if threshold is not None:
            folder_name = f"{k_str}_t{threshold}"
        else:
            folder_name = k_str
        base = os.path.join(base, folder_name)
    
    os.makedirs(base, exist_ok=True)
    return base


def save_results_csv(rows, dataset_filepath, subfolder, filename_prefix="results", k=None, threshold=None):
    """
    Save experiment results to CSV file.
    
    Parameters
    ----------
    rows : list of dict
        Results data to save
    dataset_filepath : str
        Path to original dataset
    subfolder : str
        Experiment subfolder name
    filename_prefix : str, optional
        CSV filename prefix
    k : int, list, or None
        K-value(s) for directory structure
    threshold : float or None
        Threshold for directory structure
        
    Returns
    -------
    str or None
        Path to saved CSV file, or None if no rows to save
    """
    if not rows:
        print("Warning: No rows to save.")
        return None
    
    df = pd.DataFrame(rows)
    results_dir = _results_dir(dataset_filepath, subfolder, k=k, threshold=threshold)
    csv_path = os.path.join(results_dir, f"{filename_prefix}.csv")
    
    try:
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        return csv_path
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        return None


# =============================================================================
# Algorithm Execution Wrappers
# =============================================================================

def _run_pyod_lof(X_normalized, y, k, contamination_param, algorithm, n_runs):
    """
    Run PyOD LOF implementation multiple times and aggregate results.
    
    Parameters
    ----------
    X_normalized : ndarray
        Normalized feature matrix
    y : ndarray or None
        True labels for metric computation
    k : int
        Number of neighbors
    contamination_param : float
        Expected contamination rate
    algorithm : str
        Neighbor search algorithm ('auto', 'ball_tree', 'kd_tree', 'brute')
    n_runs : int
        Number of repetitions for timing stability
        
    Returns
    -------
    dict
        Results including k, algorithm, scores, time_stats, metrics
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not 0 < contamination_param < 1:
        raise ValueError(f"contamination must be in (0,1), got {contamination_param}")
    
    times = []
    metrics_list = []
    scores_ref = None
    actual_algorithm = None  # Store the actual algorithm chosen by 'auto'
    
    for _ in range(n_runs):
        try:
            t0 = time.perf_counter()
            lof = LOF(n_neighbors=k, contamination=contamination_param, algorithm=algorithm)
            lof.fit(X_normalized)
            scores = lof.decision_scores_
            times.append(time.perf_counter() - t0)
            metrics_list.append(_compute_anomaly_metrics(y, scores))
            scores_ref = scores
            # Capture actual algorithm chosen by 'auto' (only need to do this once)
            if actual_algorithm is None and algorithm == 'auto':
                try:
                    actual_algorithm = lof.detector_._fit_method
                except AttributeError:
                    # Fallback if attribute doesn't exist
                    actual_algorithm = 'auto'
        except Exception as e:
            print(f"Warning: LOF run failed: {e}")
            continue
    
    if not times:
        raise RuntimeError("All LOF runs failed")
    
    # If algorithm is 'auto' but we couldn't determine the actual algorithm, use 'auto'
    if algorithm == 'auto' and actual_algorithm is None:
        actual_algorithm = 'auto'
    
    return {
        'k': k,
        'algorithm': algorithm,
        'actual_algorithm': actual_algorithm if algorithm == 'auto' else algorithm,
        'scores': scores_ref,
        'time_stats': _runtime_stats(times),
        'metrics': _aggregate_metrics(metrics_list),
    }


def _run_original_lof(X_normalized, y, k_min, k_max, contamination_param, n_runs):
    """
    Run Original LOF (range-based) implementation multiple times.
    
    Parameters
    ----------
    X_normalized : ndarray
        Normalized feature matrix
    y : ndarray or None
        True labels for metric computation
    k_min : int
        Minimum k value in range
    k_max : int
        Maximum k value in range
    contamination_param : float
        Expected contamination rate
    n_runs : int
        Number of repetitions for timing stability
        
    Returns
    -------
    dict
        Results including k_range, scores, time_stats, metrics
    """
    if k_min <= 0 or k_max <= 0:
        raise ValueError(f"k_min and k_max must be positive, got {k_min}, {k_max}")
    if k_min > k_max:
        raise ValueError(f"k_min must be <= k_max, got {k_min} > {k_max}")
    
    times = []
    metrics_list = []
    scores_ref = None
    
    for _ in range(n_runs):
        try:
            t0 = time.perf_counter()
            rlof = OriginalLOF(
                n_neighbors=k_max,
                n_neighbors_lb=k_min,
                contamination=contamination_param,
                algorithm='auto',
            )
            rlof.fit(X_normalized)
            scores = rlof.decision_scores_
            times.append(time.perf_counter() - t0)
            metrics_list.append(_compute_anomaly_metrics(y, scores))
            scores_ref = scores
        except Exception as e:
            print(f"Warning: Original LOF run failed: {e}")
            continue
    
    if not times:
        raise RuntimeError("All Original LOF runs failed")
    
    return {
        'k_range': f"{k_min}-{k_max}",
        'scores': scores_ref,
        'time_stats': _runtime_stats(times),
        'metrics': _aggregate_metrics(metrics_list),
    }


def _run_fastlof(X_normalized, y, k, chunk_size, contamination_param, threshold, n_runs):
    """
    Run FastLOF implementation multiple times and aggregate results.
    
    Parameters
    ----------
    X_normalized : ndarray
        Normalized feature matrix
    y : ndarray or None
        True labels for metric computation
    k : int
        Number of neighbors
    chunk_size : int or None
        Chunk size for processing (None=use FastLOF default)
    contamination_param : float
        Expected contamination rate
    threshold : float
        Distance threshold for active set pruning
    n_runs : int
        Number of repetitions for timing stability
        
    Returns
    -------
    dict
        Results including chunk_size, time_stats, metrics, scores, timing details
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")
    
    times = []
    metrics_list = []
    timing_runs = []
    scores_ref = None
    
    for _ in range(n_runs):
        try:
            t0 = time.perf_counter()
            fastlof = FastLOF(
                n_neighbors=k,
                contamination=contamination_param,
                threshold=threshold,
                chunk_size=chunk_size,
            )
            fastlof.fit(X_normalized)
            scores = fastlof.decision_scores_
            times.append(time.perf_counter() - t0)
            metrics_list.append(_compute_anomaly_metrics(y, scores))
            timing_runs.append(copy.deepcopy(getattr(fastlof, 'timing_', {})))
            scores_ref = scores
        except Exception as e:
            print(f"Warning: FastLOF run failed: {e}")
            continue
    
    if not times:
        raise RuntimeError("All FastLOF runs failed")
    
    return {
        'chunk_size': chunk_size,
        'time_stats': _runtime_stats(times),
        'metrics': _aggregate_metrics(metrics_list),
        'scores': scores_ref,
        'timing': timing_runs[-1] if timing_runs else {},
    }


# =============================================================================
# Experiment A: Original LOF vs Single-k LOF
# =============================================================================

def run_original_lof_experiment(X, y, k_min=10, k_max=50, step=10, n_runs=5, dataset_filepath=None):
    """
    Compare Original LOF (multi-k range) against standard single-k LOF.
    
    Evaluates:
    - Runtime comparison between Original LOF and single-k variants
    - Quality metrics (ROC-AUC, Precision@k) across different k values
    - Score correlation between Original LOF and single-k LOF
    
    Note: Algorithm choice (auto/ball_tree/kd_tree/brute) is irrelevant since
    the underlying mechanism is the same between scikit-learn LOF and Original LOF.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Raw feature matrix (will be normalized internally)
    y : ndarray of shape (n_samples,) or None
        True binary labels for evaluation
    k_min : int, optional (default=10)
        Minimum k value for range
    k_max : int, optional (default=50)
        Maximum k value for range
    step : int, optional (default=10)
        Step size for k values
    n_runs : int, optional (default=5)
        Number of repetitions per configuration
    dataset_filepath : str or None, optional
        Path to dataset for results organization
        
    Returns
    -------
    dict
        Results dictionary with keys:
        - 'baseline': List of LOF results per k
        - 'original': Original LOF results
    """
    print("\n=== Experiment A: Original LOF Quality & Robustness ===")
    
    # Normalize features
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    # Determine contamination parameter
    contamination_param = _true_contamination(y)
    print(f"Using contamination parameter: {contamination_param:.4f}")
    
    # Generate k values
    k_values = list(range(k_min, k_max + 1, step))
    print(f"Testing k values: {k_values}")
    
    # Run baseline LOF (algorithm choice is irrelevant)
    print("\nRunning baseline LOF...")
    baseline = []
    for k in k_values:
        print(f"  k={k}...", end='', flush=True)
        result = _run_pyod_lof(X_norm, y, k, contamination_param, 'auto', n_runs)
        baseline.append(result)
        print(f" done (avg: {result['time_stats']['avg']:.4f}s)")
    
    # Run Original LOF
    print(f"\nRunning Original LOF (k_range={k_min}-{k_max})...")
    original_result = _run_original_lof(X_norm, y, k_min, k_max, contamination_param, n_runs)
    print(f"  done (avg: {original_result['time_stats']['avg']:.4f}s)")
    
    # Prepare CSV results
    rows = []
    for res in baseline:
        rows.append({
            'Algorithm': 'LOF',
            'k': res['k'],
            'k_range': '',
            'Chunk_Size': 'N/A',
            'Runtime_Avg': res['time_stats']['avg'],
            'Runtime_Std': res['time_stats']['std'],
            'Runtime_Min': res['time_stats']['min'],
            'Runtime_Max': res['time_stats']['max'],
            'ROC_AUC': (res['metrics'] or {}).get('roc_auc'),
            'Precision_at_k': (res['metrics'] or {}).get('precision_at_k'),
        })
    
    rows.append({
        'Algorithm': 'OriginalLOF',
        'k': '',
        'k_range': f"{k_min}-{k_max}",
        'Chunk_Size': 'N/A',
        'Runtime_Avg': original_result['time_stats']['avg'],
        'Runtime_Std': original_result['time_stats']['std'],
        'Runtime_Min': original_result['time_stats']['min'],
        'Runtime_Max': original_result['time_stats']['max'],
        'ROC_AUC': (original_result['metrics'] or {}).get('roc_auc'),
        'Precision_at_k': (original_result['metrics'] or {}).get('precision_at_k'),
    })
    
    # Save results
    output_dir = _results_dir(dataset_filepath, 'lof_experiments', k=k_values) if dataset_filepath else os.getcwd()
    save_results_csv(rows, dataset_filepath, 'lof_experiments', 
                    filename_prefix='original_lof_results', k=k_values)
    
    # Generate plots
    plot_original_lof_results(dataset_filepath or 'dataset', baseline, 
                             original_result, output_dir,
                             n_runs=n_runs, k_values=k_values)
    
    print("\n=== Experiment A Complete ===")
    
    return {
        'baseline': baseline,
        'original': original_result,
    }
def plot_original_lof_results(dataset_filepath, baseline, 
                              original_result, output_dir, n_runs=None, k_values=None):
    """
    Full visualization panel for Original LOF experiment.
    Produces:
      1. Summary plot with:
         - Runtime comparison (bar)
         - AUC vs k (with Original LOF ref + best-k marker)
         - Precision@k vs k (with Original LOF ref + best-k marker)
         - Correlation vs k (single-k LOF vs Original LOF)
      2. Scatter plots: Original LOF scores vs each k LOF

    Parameters
    ----------
    dataset_filepath : str
        Dataset path
    baseline : list of dict
        Results for LOF for each k
    original_result : dict
        Results for Original LOF
    output_dir : str
        Directory to save the plots
    n_runs : int or None
        Number of runs for title
    k_values : list or None
        K values tested (extracted if None)
    """

    # ----------------------------------------------------------------------
    # Extract metrics
    # ----------------------------------------------------------------------
    if k_values is None:
        k_values = [r['k'] for r in baseline]

    lof_aucs = np.array([r['metrics']['roc_auc'] if r['metrics'] else np.nan for r in baseline])
    lof_prec = np.array([r['metrics']['precision_at_k'] if r['metrics'] else np.nan for r in baseline])

    orig_auc = original_result['metrics']['roc_auc'] if original_result['metrics'] else np.nan
    orig_prec = original_result['metrics']['precision_at_k'] if original_result['metrics'] else np.nan
    orig_scores = np.array(original_result['scores'])

    # ----------------------------------------------------------------------
    # Compute correlation per k between Original LOF and Single-k LOF
    # ----------------------------------------------------------------------
    correlations = []
    for r in baseline:
        s_lof = np.array(r['scores'])
        # Filter extreme outliers (>100) for visualization clarity
        mask = (orig_scores < 100) & (s_lof < 100)
        if not np.any(mask):
            mask = np.ones_like(orig_scores, dtype=bool)

        correlations.append(np.corrcoef(orig_scores[mask], s_lof[mask])[0, 1]
                           if s_lof.size > 1 else np.nan)

    correlations = np.array(correlations)

    # ----------------------------------------------------------------------
    # Identify Best-k
    # ----------------------------------------------------------------------
    best_idx = int(np.nanargmax(lof_aucs))
    best_k = k_values[best_idx]

    # ----------------------------------------------------------------------
    # Runtime aggregation
    # ----------------------------------------------------------------------
    def _agg_time(list_stats):
        avg = np.mean([s['time_stats']['avg'] for s in list_stats])
        min_v = np.min([s['time_stats']['min'] for s in list_stats])
        max_v = np.max([s['time_stats']['max'] for s in list_stats])
        return avg, avg - min_v, max_v - avg

    lof_avg, lof_err_low, lof_err_up = _agg_time(baseline)

    orig_avg = original_result['time_stats']['avg']
    orig_err_low = orig_avg - original_result['time_stats']['min']
    orig_err_up = original_result['time_stats']['max'] - orig_avg

    # ----------------------------------------------------------------------
    # FIGURE 1: Full summary panel
    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Build title with runs and k range
    title_parts = [f"Original LOF vs Single-k LOF — {os.path.basename(dataset_filepath)}"]
    if n_runs is not None:
        title_parts.append(f"runs={n_runs}")
    if k_values is not None:
        k_range_str = f"k={min(k_values)}-{max(k_values)}"
        title_parts.append(k_range_str)
    title = " | ".join(title_parts)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # ----------------------------------------------------------------------
    # Panel 1: Runtime comparison
    # ----------------------------------------------------------------------
    ax = axes[0, 0]
    labels = ["LOF (avg over k)", "Original LOF"]
    values = [lof_avg, orig_avg]
    yerr = [[lof_err_low, orig_err_low],
            [lof_err_up, orig_err_up]]

    ax.bar(labels, values, yerr=yerr, alpha=0.75, capsize=6)
    ax.set_title("Runtime Comparison")
    ax.set_ylabel("Time (seconds)")
    ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0 (time cannot be negative)
    ax.grid(True, axis='y', alpha=0.3)

    # ----------------------------------------------------------------------
    # Panel 2: AUC vs k
    # ----------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(k_values, lof_aucs, "o-", label="LOF", color="blue")
    ax.axhline(orig_auc, color="red", linestyle="--", linewidth=2,
               label=f"Original LOF AUC = {orig_auc:.4f}")

    # Mark best-k
    ax.scatter(best_k, lof_aucs[best_idx], color="blue", s=80, edgecolor="black")

    ax.set_title("AUC vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("ROC AUC")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ----------------------------------------------------------------------
    # Panel 3: Precision@k vs k
    # ----------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(k_values, lof_prec, "o-", label="LOF", color="blue")
    ax.axhline(orig_prec, color="red", linestyle="--", linewidth=2,
               label=f"Original LOF P@k = {orig_prec:.4f}")

    # Mark best-k
    ax.scatter(best_k, lof_prec[best_idx], color="blue", s=80, edgecolor="black")

    ax.set_title("Precision@k vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ----------------------------------------------------------------------
    # Panel 4: Correlation vs k
    # ----------------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(k_values, correlations, "o-", label="Correlation (LOF vs Original LOF)", color="blue")
    ax.set_title("Correlation with Original LOF vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Pearson r")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ----------------------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------------------
    out_path = os.path.join(output_dir, "original_lof_summary.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ----------------------------------------------------------------------
    # FIGURE 2: Scatter plots — Original LOF vs each k LOF
    # ----------------------------------------------------------------------
    n_k = len(k_values)
    n_cols = min(3, n_k)
    n_rows = int(math.ceil(n_k / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    
    for idx, (k_val, res) in enumerate(zip(k_values, baseline)):
        ax = axes_flat[idx]
        lof_scores = np.array(res["scores"])
        
        mask = (orig_scores < 100) & (lof_scores < 100)
        if not np.any(mask):
            mask = np.ones_like(orig_scores, dtype=bool)
        
        x = lof_scores[mask]
        y = orig_scores[mask]
        corr = np.corrcoef(x, y)[0, 1] if x.size > 2 else np.nan
        
        ax.scatter(x, y, s=10, alpha=0.5)
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.2)
        
        ax.set_title(f"Original LOF vs k={k_val} LOF\nCorr={corr:.4f}")
        ax.set_xlabel(f"k={k_val} LOF")
        ax.set_ylabel("Original LOF")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_k, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    scatter_path = os.path.join(output_dir, "original_lof_scatter.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {scatter_path}")


# =============================================================================
# Experiment B: FastLOF vs Standard LOF
# =============================================================================

def run_fastlof_experiment(
    X,
    y,
    k_values=[10, 20, 30, 40, 50],
    min_chunk_size=2048,
    max_chunk_size=8192,
    chunk_interval=2048,
    n_runs=5,
    threshold=1.2,
    dataset_filepath=None,
    skip_ball_tree=False,
    skip_kd_tree=False,
):
    """
    Compare FastLOF against standard LOF across chunk sizes and k values.
    
    Results are averaged across k values to show chunk size effects independent
    of k selection. This provides a robust view of FastLOF's speed-accuracy 
    tradeoff characteristics.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Raw feature matrix (will be normalized internally)
    y : ndarray of shape (n_samples,) or None
        True binary labels for evaluation
    k_values : list of int, optional (default=[10,20,30,40,50])
        K-values to test (results will be averaged)
    min_chunk_size : int, optional (default=2048)
        Minimum chunk size to test
    max_chunk_size : int, optional (default=8192)
        Maximum chunk size to test
    chunk_interval : int, optional (default=2048)
        Step size between chunk sizes
    n_runs : int, optional (default=5)
        Number of repetitions per configuration
    threshold : float, optional (default=1.2)
        Distance threshold for FastLOF active set pruning
    dataset_filepath : str or None, optional
        Path to dataset for results organization
    skip_ball_tree : bool, optional (default=False)
        If True, skip ball_tree algorithm in baseline LOF tests
    skip_kd_tree : bool, optional (default=False)
        If True, skip kd_tree algorithm in baseline LOF tests
        
    Returns
    -------
    dict
        Results dictionary with keys:
        - 'baseline_auto': Averaged LOF auto results
        - 'baseline_brute': Averaged LOF brute results
        - 'fast_results': List of FastLOF results per chunk size
    """
    print("\n=== Experiment B: FastLOF Speed & Scalability ===")
    
    # Normalize features
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    # Determine contamination parameter
    contamination_param = _true_contamination(y)
    print(f"Using contamination parameter: {contamination_param:.4f}")
    print(f"Testing k values: {k_values}")
    print(f"Chunk sizes: {max_chunk_size} to {min_chunk_size} (step={chunk_interval})")
    
    # Run baseline LOF for each k value with multiple algorithms
    print("\n--- Running baseline LOF ---")
    
    # Build list of algorithms to test based on skip parameters
    algorithms_to_test = []
    if not skip_ball_tree:
        algorithms_to_test.append('ball_tree')
    if not skip_kd_tree:
        algorithms_to_test.append('kd_tree')
    algorithms_to_test.append('brute')  # Always include brute
    
    if skip_ball_tree or skip_kd_tree:
        skipped = []
        if skip_ball_tree:
            skipped.append('ball_tree')
        if skip_kd_tree:
            skipped.append('kd_tree')
        print(f"Skipping algorithms: {', '.join(skipped)}")
    
    baseline_lists = {alg: [] for alg in algorithms_to_test}
    baseline_scores_lists = {alg: [] for alg in algorithms_to_test}
    failed_algorithms = set()
    
    for k in k_values:
        print(f"  k={k}", end='', flush=True)
        for algorithm in algorithms_to_test:
            if algorithm in failed_algorithms:
                continue  # Skip algorithms that have already failed
            print(f" ({algorithm})...", end='', flush=True)
            try:
                res = _run_pyod_lof(X_norm, y, k, contamination_param, algorithm, n_runs)
                baseline_lists[algorithm].append(res)
                baseline_scores_lists[algorithm].append(res['scores'])
                print(f" {res['time_stats']['avg']:.4f}s", end='')
            except Exception as e:
                print(f" FAILED", end='')
                failed_algorithms.add(algorithm)
        print()  # Newline after all algorithms for this k
    
    # Aggregate baseline results across k values
    def _agg_baseline(results_list, scores_list):
        """Aggregate results across k values."""
        times_all = []
        metrics_list_all = []
        actual_algorithm = None
        for res in results_list:
            times_all.extend(res['time_stats']['times'])
            if res['metrics']:
                metrics_list_all.append(res['metrics'])
            # Capture actual_algorithm from first result (should be same for all k)
            if actual_algorithm is None and 'actual_algorithm' in res:
                actual_algorithm = res['actual_algorithm']
        result = {
            'time_stats': _runtime_stats(times_all),
            'metrics': _aggregate_metrics(metrics_list_all),
            'scores': np.mean(scores_list, axis=0) if scores_list else None,
        }
        if actual_algorithm:
            result['actual_algorithm'] = actual_algorithm
        return result
    
    # Aggregate results for each algorithm
    baseline_agg = {}
    for algorithm in algorithms_to_test:
        if algorithm not in failed_algorithms and baseline_lists[algorithm]:
            baseline_agg[algorithm] = _agg_baseline(baseline_lists[algorithm], baseline_scores_lists[algorithm])
            baseline_agg[algorithm]['algorithm'] = algorithm
        else:
            print(f"\n  WARNING: {algorithm} failed for all k values, excluding from comparison")
            baseline_agg[algorithm] = None
    
    # Extract individual baselines for compatibility
    baseline_ball_tree = baseline_agg.get('ball_tree')
    baseline_kd_tree = baseline_agg.get('kd_tree')
    baseline_brute = baseline_agg.get('brute')
    
    # Choose brute as primary baseline for FastLOF comparison (most accurate)
    successful_algorithms = [alg for alg, res in baseline_agg.items() if res is not None]
    if not successful_algorithms:
        raise RuntimeError("All baseline algorithms failed, cannot proceed with experiment")
    
    # Use brute for comparisons (if available), otherwise fallback to first successful
    if baseline_brute is not None:
        baseline_reference = baseline_brute
    else:
        baseline_reference = baseline_agg[successful_algorithms[0]]
    
    print(f"\nBaseline algorithms:")
    for algorithm in algorithms_to_test:
        result = baseline_agg.get(algorithm)
        if result:
            print(f"  LOF {algorithm} (averaged): {result['time_stats']['avg']:.4f}s")
    print(f"\nUsing LOF brute as reference for FastLOF comparisons")
    
    # Build chunk sizes to test
    chunk_sizes = list(range(max_chunk_size, min_chunk_size - 1, -chunk_interval))
    print(f"\nTesting chunk sizes: {chunk_sizes}")
    
    # Run FastLOF for each chunk size (averaged across k values)
    print("\n--- Running FastLOF ---")
    fast_results = []
    
    for cs in chunk_sizes:
        print(f"Chunk size {cs}:")
        
        # Collect results for all k values for this chunk size
        chunk_results = []
        chunk_scores_list = []
        chunk_times_all = []
        chunk_metrics_all = []
        chunk_timing = None
        
        for k in k_values:
            print(f"  k={k}...", end='', flush=True)
            res = _run_fastlof(X_norm, y, k, cs, contamination_param, threshold, n_runs)
            chunk_results.append(res)
            chunk_scores_list.append(res['scores'])
            chunk_times_all.extend(res['time_stats']['times'])
            if res['metrics']:
                chunk_metrics_all.append(res['metrics'])
            if res.get('timing'):
                chunk_timing = res['timing']  # Keep last timing for reference
            print(f" {res['time_stats']['avg']:.4f}s")
        
        # Average results across k values for this chunk size
        avg_scores = np.mean(chunk_scores_list, axis=0) if chunk_scores_list else None
        avg_result = {
            'chunk_size': cs,
            'chunk_count': (chunk_timing or {}).get('n_chunks') if chunk_timing else None,
            'time_stats': _runtime_stats(chunk_times_all),
            'metrics': _aggregate_metrics(chunk_metrics_all),
            'scores': avg_scores,
            'timing': chunk_timing,
        }
        
        # Calculate correlation with brute baseline (most accurate reference)
        if avg_scores is not None and baseline_reference['scores'] is not None:
            avg_result['correlation'] = np.corrcoef(baseline_reference['scores'], avg_scores)[0, 1]
        else:
            avg_result['correlation'] = np.nan
        
        fast_results.append(avg_result)
        print(f"  Averaged: {avg_result['time_stats']['avg']:.4f}s, "
              f"corr={avg_result['correlation']:.4f}")
    
    # Prepare CSV results
    k_range_str = f"{min(k_values)}-{max(k_values)}"
    rows = []
    
    # Baseline results - include all successful algorithms
    for algorithm_name, base_res in [('ball_tree', baseline_ball_tree),
                                      ('kd_tree', baseline_kd_tree),
                                      ('brute', baseline_brute)]:
        if base_res is not None:
            rows.append({
                'Algorithm': f"LOF_{algorithm_name}",
                'k': '',
                'k_range': k_range_str,
                'Chunk_Size': 'N/A',
                'Runtime_Avg': base_res['time_stats']['avg'],
                'Runtime_Std': base_res['time_stats']['std'],
                'Runtime_Min': base_res['time_stats']['min'],
                'Runtime_Max': base_res['time_stats']['max'],
                'ROC_AUC': (base_res['metrics'] or {}).get('roc_auc'),
                'Precision_at_k': (base_res['metrics'] or {}).get('precision_at_k'),
            })
    
    # FastLOF results
    for res in fast_results:
        rows.append({
            'Algorithm': 'FastLOF',
            'k': '',
            'k_range': k_range_str,
            'Chunk_Size': res['chunk_size'],
            'Runtime_Avg': res['time_stats']['avg'],
            'Runtime_Std': res['time_stats']['std'],
            'Runtime_Min': res['time_stats']['min'],
            'Runtime_Max': res['time_stats']['max'],
            'ROC_AUC': (res['metrics'] or {}).get('roc_auc'),
            'Precision_at_k': (res['metrics'] or {}).get('precision_at_k'),
        })
    
    # Save results
    output_dir = _results_dir(dataset_filepath, 'fastlof_experiments', 
                              k=k_values, threshold=threshold) if dataset_filepath else os.getcwd()
    save_results_csv(rows, dataset_filepath, 'fastlof_experiments', 
                    filename_prefix='fastlof_results', k=k_values, threshold=threshold)
    
    # Generate plots
    plot_fastlof_results(dataset_filepath or 'dataset', baseline_ball_tree, 
                        baseline_kd_tree, baseline_brute, 
                        fast_results, output_dir, n_runs=n_runs, 
                        threshold=threshold, k_values=k_values)
    
    print("\n=== Experiment B Complete ===")
    
    return {
        'baseline_ball_tree': baseline_ball_tree,
        'baseline_kd_tree': baseline_kd_tree,
        'baseline_brute': baseline_brute,
        'baseline_reference': baseline_reference,  # The baseline used for comparisons
        'fast_results': fast_results,
    }


def plot_fastlof_results(dataset_filepath, baseline_ball_tree, baseline_kd_tree, 
                        baseline_brute, fast_results, output_dir, n_runs=None, 
                        threshold=None, k_values=None):
    """
    Generate comprehensive visualization suite for FastLOF experiment.
    
    Creates:
    1. Main summary (4 subplots):
       - Score correlation vs chunk size
       - Baseline runtime comparison
       - FastLOF runtime vs chunk size with baseline references
       - ROC-AUC vs chunk size with baseline reference
    2. Detail plot (3 subplots):
       - Precision@k vs chunk size
       - Dual-axis runtime + AUC
       - PR-AUC vs chunk size
    3. Scatter plots: FastLOF vs LOF scores per chunk size
    4. Timing breakdown pie charts per chunk size
    
    Parameters
    ----------
    dataset_filepath : str
        Dataset name for plot titles
    baseline_ball_tree : dict or None
        Aggregated results from LOF ball_tree (None if failed)
    baseline_kd_tree : dict or None
        Aggregated results from LOF kd_tree (None if failed)
    baseline_brute : dict or None
        Aggregated results from LOF brute (None if failed)
    fast_results : list of dict
        Results per chunk size from FastLOF
    output_dir : str
        Directory to save plots
    n_runs : int or None
        Number of runs (for title)
    threshold : float or None
        Threshold parameter (for title)
    k_values : list or None
        K-values tested (for title)
    """
    # Organize baseline results
    baselines = {
        'ball_tree': baseline_ball_tree,
        'kd_tree': baseline_kd_tree,
        'brute': baseline_brute
    }
    successful_baselines = {name: res for name, res in baselines.items() if res is not None}
    
    if not successful_baselines:
        print("ERROR: All baseline algorithms failed, cannot generate plots")
        return
    
    # Use brute as reference for quality comparisons (most accurate), fallback if not available
    if baseline_brute is not None:
        baseline_reference = baseline_brute
    else:
        baseline_reference = list(successful_baselines.values())[0]
    
    chunk_labels = [res['chunk_size'] for res in fast_results]
    chunk_numeric = [lbl for lbl in chunk_labels]
    
    corr_vals = [r.get('correlation') for r in fast_results]
    auc_vals = [(r['metrics'] or {}).get('roc_auc') for r in fast_results]
    prec_vals = [(r['metrics'] or {}).get('precision_at_k') for r in fast_results]
    pr_auc_vals = [(r['metrics'] or {}).get('pr_auc') for r in fast_results]
    runtimes = [r['time_stats']['avg'] for r in fast_results]
    
    # Use brute as baseline for quality metrics
    baseline_auc = (baseline_reference['metrics'] or {}).get('roc_auc')
    baseline_prec = (baseline_reference['metrics'] or {}).get('precision_at_k')
    baseline_pr_auc = (baseline_reference['metrics'] or {}).get('pr_auc')
    
    # ========== Main summary (4 plots) ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Build title with runs, threshold, and k range
    title_parts = [f"FastLOF Speed & Fidelity ({os.path.basename(dataset_filepath)})"]
    if n_runs is not None:
        title_parts.append(f"runs={n_runs}")
    if threshold is not None:
        title_parts.append(f"threshold={threshold}")
    if k_values is not None:
        k_range_str = f"k={min(k_values)}-{max(k_values)}"
        title_parts.append(k_range_str)
    title = " | ".join(title_parts)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 1. Correlation vs chunk size
    axes[0, 0].plot(chunk_numeric, corr_vals, 'o-', color='teal')
    axes[0, 0].set_xlabel('Chunk size')
    axes[0, 0].set_ylabel('Correlation with LOF (brute)')
    axes[0, 0].set_title('FastLOF correlation vs chunk size')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Baseline runtimes comparison
    colors_map = {'ball_tree': 'blue', 'kd_tree': 'orange', 'brute': 'green'}
    
    baseline_names = [f'LOF {name}' for name in successful_baselines.keys()]
    baseline_avgs = [successful_baselines[name]['time_stats']['avg'] for name in successful_baselines.keys()]
    baseline_stds = [successful_baselines[name]['time_stats']['std'] for name in successful_baselines.keys()]
    bar_colors = [colors_map.get(name, 'gray') for name in successful_baselines.keys()]
    
    bars_base = axes[0, 1].bar(
        baseline_names, baseline_avgs, yerr=baseline_stds,
        color=bar_colors, alpha=0.7, capsize=6
    )
    
    bars_top = []
    for bar, name in zip(bars_base, successful_baselines.keys()):
        base = successful_baselines[name]
        val = base['time_stats']['avg']
        hi = base['time_stats']['std']
        text_y = val + hi + 0.06 * (val + hi)
        bars_top.append(text_y)
        axes[0, 1].text(
            bar.get_x() + bar.get_width()/2, text_y,
            f"min: {base['time_stats']['min']:.4f}s\n"
            f"avg: {val:.4f}s\n"
            f"max: {base['time_stats']['max']:.4f}s",
            ha='center', va='bottom', fontsize=8, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='black', alpha=0.9)
        )
    
    axes[0, 1].set_ylim(bottom=0, top=max(bars_top) * 1.10 if bars_top else None)
    axes[0, 1].set_ylabel('Runtime (s)')
    axes[0, 1].set_title('Baseline runtimes')
    axes[0, 1].grid(True, axis='y', alpha=0.3)
    axes[0, 1].legend(loc='upper left', fontsize=8)
    # 3. Runtime vs chunk size with reference lines (outlier-resistant y-axis)
    # Sort by chunk size (ascending order) - handle None/'default' by putting it at the end
    sorted_indices = sorted(range(len(chunk_numeric)), 
                           key=lambda i: float('inf') if (chunk_numeric[i] is None or chunk_numeric[i] == 'default') else chunk_numeric[i])
    sorted_chunk_labels = [chunk_labels[i] for i in sorted_indices]
    sorted_chunk_numeric = [chunk_numeric[i] for i in sorted_indices]
    sorted_runtimes = [runtimes[i] for i in sorted_indices]
    sorted_fast_results = [fast_results[i] for i in sorted_indices]
    sorted_runtime_err_low = [r['time_stats']['avg'] - r['time_stats']['min'] for r in sorted_fast_results]
    sorted_runtime_err_up = [r['time_stats']['max'] - r['time_stats']['avg'] for r in sorted_fast_results]
    
    bars_runtime = axes[1, 0].bar(
        range(len(sorted_chunk_labels)), sorted_runtimes, color='red', alpha=0.7,
        yerr=[sorted_runtime_err_low, sorted_runtime_err_up], capsize=6
    )
    
    # Calculate all bar tops (avg + error)
    bar_tops = [val + err for val, err in zip(sorted_runtimes, sorted_runtime_err_up)]
    max_runtime_top = max(bar_tops) if bar_tops else 0
    
    # Use 95th percentile for y-axis if there are outliers
    if len(bar_tops) > 2:
        p95 = np.percentile(bar_tops, 95)
        # If max is much larger than p95, cap at p95
        if max_runtime_top > p95 * 2:
            y_max = p95 * 1.15  # Add 15% margin above p95
            axes[1, 0].set_ylim(bottom=0, top=y_max)
            
            # Count and annotate clipped bars
            clipped = [(i, sorted_chunk_labels[i], bar_tops[i]) 
                      for i, top in enumerate(bar_tops) if top > y_max]
            if clipped:
                clipped_str = ', '.join([f"{lbl} ({val:.2f}s)" for _, lbl, val in clipped])
                axes[1, 0].text(0.98, 0.98, f'Clipped (max runtime):\n{clipped_str}',
                               transform=axes[1, 0].transAxes, fontsize=7,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                                        alpha=0.5, edgecolor='red'))
        else:
            axes[1, 0].set_ylim(bottom=0, top=max_runtime_top * 1.05)
    else:
        axes[1, 0].set_ylim(bottom=0, top=max_runtime_top * 1.05)
    
    # Add reference lines for all successful baseline algorithms
    for name in successful_baselines.keys():
        baseline = successful_baselines[name]
        color = colors_map.get(name, 'gray')
        linestyle_map = {'ball_tree': '-', 'kd_tree': '-.', 'brute': '--'}
        axes[1, 0].axhline(baseline['time_stats']['avg'], color=color, 
                          linestyle=linestyle_map.get(name, '--'), 
                          label=f'LOF {name} avg = {baseline["time_stats"]["avg"]:.4f}s')
    
    axes[1, 0].set_xticks(range(len(sorted_chunk_labels)))
    axes[1, 0].set_xticklabels(sorted_chunk_labels, rotation=45)
    axes[1, 0].set_xlim(-0.5, len(sorted_chunk_labels) - 0.5)
    axes[1, 0].set_ylabel('Runtime (s)')
    axes[1, 0].set_title('FastLOF runtime by chunk size')
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    axes[1, 0].legend(loc='upper left', fontsize=8)
    
    # 4. ROC AUC vs chunk size with baseline (adaptive y-axis)
    axes[1, 1].plot(chunk_numeric, auc_vals, 'o-', color='purple', label='FastLOF AUC')
    if baseline_auc is not None:
        axes[1, 1].axhline(baseline_auc, color='green', linestyle='--', 
                          label=f'LOF brute AUC = {baseline_auc:.4f}')
    axes[1, 1].set_xlabel('Chunk size')
    axes[1, 1].set_ylabel('ROC AUC')
    
    # Adaptive y-axis: zoom in if variation is small
    valid_aucs = [v for v in auc_vals if v is not None]
    if baseline_auc is not None:
        valid_aucs.append(baseline_auc)
    
    if valid_aucs:
        auc_min, auc_max = min(valid_aucs), max(valid_aucs)
        auc_range = auc_max - auc_min
        
        if auc_range < 0.1:  # Small variation - zoom in
            margin = max(0.02, auc_range * 0.3)  # At least 2% margin
            y_min = max(0.0, auc_min - margin)
            y_max = min(1.0, auc_max + margin)
            axes[1, 1].set_ylim(y_min, y_max)
            # Add context annotation
            axes[1, 1].text(0.02, 0.98, f'Zoomed view\n(range: {auc_range:.4f})', 
                           transform=axes[1, 1].transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', 
                           facecolor='yellow', alpha=0.3))
        else:  # Large variation - show full range
            axes[1, 1].set_ylim(0.0, 1.05)
    else:
        axes[1, 1].set_ylim(0.0, 1.05)
    
    axes[1, 1].set_title('AUC vs chunk size')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='lower right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(output_dir, 'fastlof_summary.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Experiment B plots to {out_path}")
    
    # ========== Detail figure (3 plots) ==========
    fig_d, axes_d = plt.subplots(1, 3, figsize=(18, 5))
    fig_d.suptitle(f"FastLOF Detail ({os.path.basename(dataset_filepath)})", 
                   fontsize=14, fontweight='bold')
    
    # 1. Precision@k vs chunk size
    axes_d[0].plot(chunk_numeric, prec_vals, 'o-', color='orange', label='FastLOF P@k')
    if baseline_prec is not None:
        axes_d[0].axhline(baseline_prec, color='green', linestyle='--', 
                         label=f'LOF brute P@k = {baseline_prec:.4f}')
    axes_d[0].set_xlabel('Chunk size')
    axes_d[0].set_ylabel('Precision@k')
    axes_d[0].set_ylim(0.0, 1.05)
    axes_d[0].set_title('Precision@k vs chunk size')
    axes_d[0].grid(True, alpha=0.3)
    axes_d[0].legend(loc='lower right', fontsize=8)
    
    # 2. Dual-axis runtime + AUC vs chunk size
    ax_rt = axes_d[1]
    ax_auc = ax_rt.twinx()
    ax_rt.plot(chunk_numeric, runtimes, 'o-', color='red', label='Runtime (s)')
    ax_auc.plot(chunk_numeric, auc_vals, 's--', color='purple', label='ROC AUC')
    if baseline_auc is not None:
        ax_auc.axhline(baseline_auc, color='green', linestyle=':', linewidth=1.5,
                      label=f'LOF brute AUC = {baseline_auc:.4f}')
    ax_rt.set_xlabel('Chunk size')
    ax_rt.set_ylabel('Runtime (s)', color='red')
    ax_auc.set_ylabel('ROC AUC', color='purple')
    ax_rt.set_title('Runtime & AUC vs chunk size')
    ax_rt.grid(True, alpha=0.3)
    lines = ax_rt.get_lines() + ax_auc.get_lines()
    labels = [l.get_label() for l in lines]
    ax_rt.legend(lines, labels, loc='lower right', fontsize=8)
    
    # 3. PR-AUC vs chunk size
    axes_d[2].plot(chunk_numeric, pr_auc_vals, 'o-', color='teal', label='FastLOF PR-AUC')
    if baseline_pr_auc is not None:
        axes_d[2].axhline(baseline_pr_auc, color='green', linestyle='--', 
                         label=f'LOF brute PR-AUC = {baseline_pr_auc:.4f}')
    axes_d[2].set_xlabel('Chunk size')
    axes_d[2].set_ylabel('PR-AUC')
    axes_d[2].set_ylim(0.0, 1.05)
    axes_d[2].set_title('PR-AUC vs chunk size')
    axes_d[2].grid(True, alpha=0.3)
    axes_d[2].legend(loc='lower right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path_d = os.path.join(output_dir, 'fastlof_detail.png')
    plt.savefig(out_path_d, dpi=300, bbox_inches='tight')
    plt.close(fig_d)
    print(f"Saved detail plots to {out_path_d}")
    
    # ========== Scatter plots: FastLOF vs LOF scores ==========
    if fast_results:
        sorted_results = sorted(fast_results, key=lambda r: r['chunk_size'])
        n_results = len(sorted_results)
        n_cols = min(3, n_results)
        n_rows = int(math.ceil(n_results / n_cols))
        
        scatter_fig, axes_s = plt.subplots(n_rows, n_cols, 
                                          figsize=(6 * n_cols, 4.5 * n_rows), 
                                          squeeze=False)
        axes_flat = axes_s.flatten()
        
        for idx, res in enumerate(sorted_results):
            ax = axes_flat[idx]
            fast_scores = res['scores']
            ref_scores = baseline_reference['scores']
            
            # Filter outliers for better visualization
            mask = (ref_scores < 100) & (fast_scores < 100)
            if not np.any(mask):
                mask = np.ones_like(ref_scores, dtype=bool)
            
            x = ref_scores[mask]
            y = fast_scores[mask]
            corr = np.corrcoef(x, y)[0, 1] if x.size > 1 else np.nan
            
            mn = min(x.min(), y.min())
            mx = max(x.max(), y.max())
            
            ax.scatter(x, y, s=8, alpha=0.5, color='teal')
            ax.plot([mn, mx], [mn, mx], '--', color='red', linewidth=1.2, label='y = x')
            ax.set_xlabel('LOF (brute) score')
            ax.set_ylabel('FastLOF score')
            ax.set_title(f"Chunk {res['chunk_size']} corr={corr:.4f}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
        
        # Hide unused subplots
        for idx in range(n_results, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        scatter_fig.tight_layout(rect=[0, 0, 1, 0.96])
        scatter_path = os.path.join(output_dir, 'fastlof_scatter.png')
        scatter_fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close(scatter_fig)
        print(f"Saved FastLOF scatter plots to {scatter_path}")
    
    # ========== Timing pie charts per chunk size ==========
    timing_results = [r for r in fast_results if r.get('timing')]
    if timing_results:
        n_cols = min(3, len(timing_results))
        n_rows = int(math.ceil(len(timing_results) / n_cols))
        
        t_fig, t_axes = plt.subplots(n_rows, n_cols, 
                                     figsize=(6 * n_cols, 5 * n_rows), 
                                     squeeze=False)
        t_flat = t_axes.flatten()
        
        for idx, res in enumerate(timing_results):
            ax = t_flat[idx]
            timing = res.get('timing', {}) or {}
            details = timing.get('chunk_processing_details', {})
            
            dist_t = details.get('distance_computation', 0.0)
            neigh_t = details.get('neighbor_updates', 0.0)
            lof_t = timing.get('lof_calculation', 0.0)
            other_t = (timing.get('initialization', 0.0) + 
                      details.get('self_distance_handling', 0.0) + 
                      timing.get('active_set_updates', 0.0) + 
                      timing.get('finalization', 0.0))
            
            values = [dist_t, neigh_t, lof_t, other_t]
            labels = ['Distance', 'Neighbor', 'LOF calc', 'Others']
            total = sum(values)
            
            if total <= 0:
                ax.text(0.5, 0.5, 'No timing data', ha='center', va='center')
                ax.axis('off')
                continue
            
            def autopct(pct):
                return f"{(pct*total/100):.2f}s\n({pct:.1f}%)" if total > 0 else ''
            
            ax.pie(values, labels=labels, autopct=autopct, startangle=90, 
                  textprops={'fontsize': 8})
            ax.axis('equal')
            
            # Add distances_computed percentage if available
            distances_info = timing.get('distances_computed', 'N/A')
            if distances_info != 'N/A':
                ax.text(0.5, 1.15, f"Distances computed: {distances_info}",
                       transform=ax.transAxes, ha='center', va='center', 
                       fontsize=9, fontweight='bold')
            
            chunk_count = res.get('chunk_count', '?')
            ax.set_title(f"Chunk {res['chunk_size']} (chunks={chunk_count})\n"
                        f"Total: {total:.2f}s", fontsize=10)
        
        # Hide unused subplots
        for idx in range(len(timing_results), len(t_flat)):
            t_flat[idx].axis('off')
        
        t_fig.tight_layout(rect=[0, 0, 1, 0.97])
        t_path = os.path.join(output_dir, 'fastlof_timing.png')
        t_fig.savefig(t_path, dpi=300, bbox_inches='tight')
        plt.close(t_fig)
        print(f"Saved FastLOF timing pies to {t_path}")

