"""
Generate visualization plots for FastLOF summary tables.
Creates per-threshold plots showing performance comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os
from experiments import load_dataset

def get_dataset_sizes():
    """Get dataset sizes (n_samples, n_features) for all datasets."""
    dataset_sizes = {}
    data_dir = Path('data')
    
    # Map dataset names in results to file names
    dataset_file_map = {
        'annthyroid-unsupervised-ad': 'annthyroid-unsupervised-ad.csv',
        'breast-cancer-unsupervised-ad': 'breast-cancer-unsupervised-ad.csv',
        'creditcard': 'creditcard.csv',
        'dfki-artificial-3000-unsupervised-ad': 'dfki-artificial-3000-unsupervised-ad.csv',
        'ForestCover': 'ForestCover.mat',
        'http': 'http.mat',
        'InternetAds_norm_02_v01': 'InternetAds_norm_02_v01.arff',
        'kdd99-unsupervised-ad': 'kdd99-unsupervised-ad.csv',
        'mammography': 'mammography.mat',
        'mulcross': 'mulcross.arff',
        'pen-global-unsupervised-ad': 'pen-global-unsupervised-ad.csv',
        'pen-local-unsupervised-ad': 'pen-local-unsupervised-ad.csv',
        'PenDigits_withoutdupl_norm_v01': 'PenDigits_withoutdupl_norm_v01.arff',
        'satellite-unsupervised-ad': 'satellite-unsupervised-ad.csv',
        'shuttle-unsupervised-ad': 'shuttle-unsupervised-ad.csv',
    }
    
    for dataset_name, filename in dataset_file_map.items():
        filepath = data_dir / filename
        if filepath.exists():
            try:
                X, y = load_dataset(str(filepath))
                dataset_sizes[dataset_name] = X.shape
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                dataset_sizes[dataset_name] = (0, 0)
        else:
            dataset_sizes[dataset_name] = (0, 0)
    
    return dataset_sizes

def get_clean_dataset_name(dataset_name):
    """Map full dataset names to cleaner, shorter names."""
    name_mapping = {
        'breast-cancer-unsupervised-ad': 'Breast Cancer',
        'pen-global-unsupervised-ad': 'Pen (Global)',
        'InternetAds_norm_02_v01': 'InternetAds',
        'dfki-artificial-3000-unsupervised-ad': 'DFKI (Artif.)',
        'satellite-unsupervised-ad': 'Satellite',
        'pen-local-unsupervised-ad': 'Pen (Local)',
        'annthyroid-unsupervised-ad': 'Annthyroid',
        'PenDigits_withoutdupl_norm_v01': 'PenDigits',
        'mammography': 'Mammography',
        'shuttle-unsupervised-ad': 'Shuttle',
        'mulcross': 'Mulcross',
        'creditcard': 'CreditCard',
        'ForestCover': 'ForestCover',
        'http': 'HTTP',
        'kdd99-unsupervised-ad': 'KDD99',
    }
    return name_mapping.get(dataset_name, dataset_name)

def format_dataset_label(dataset_name, dataset_sizes):
    """Format dataset name with shape information in brackets."""
    clean_name = get_clean_dataset_name(dataset_name)
    if dataset_name in dataset_sizes:
        n_samples, n_features = dataset_sizes[dataset_name]
        return f"{clean_name} ({n_samples:,}, {n_features})"
    return clean_name

def parse_mean_std(value_str):
    """Parse 'mean ± std' string into (mean, std) tuple."""
    if pd.isna(value_str) or value_str == "N/A":
        return None, None
    try:
        # Match pattern: "number ± number"
        match = re.match(r'([\d.]+)\s*±\s*([\d.]+)', str(value_str))
        if match:
            mean = float(match.group(1))
            std = float(match.group(2))
            return mean, std
        else:
            # Try to parse as single number
            return float(value_str), 0.0
    except:
        return None, None

def load_fastlof_data(threshold, dataset_sizes):
    """Load and parse FastLOF summary data for a threshold."""
    csv_path = Path("results_summary") / f"summary_fastlof_{threshold}.csv"
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    
    # Sort by dataset size (number of samples)
    if 'Dataset' in df.columns:
        # Add size column for sorting
        df['_sort_size'] = df['Dataset'].apply(
            lambda x: dataset_sizes.get(x, (0, 0))[0]
        )
        df = df.sort_values('_sort_size').reset_index(drop=True)
        df = df.drop('_sort_size', axis=1)
    
    # Parse all time and AUC columns
    parsed_data = {'Dataset': df['Dataset'].values}
    
    # Baseline LOF methods
    for method in ['LOF_ball_tree', 'LOF_kd_tree', 'LOF_brute']:
        time_col = f'{method}_Time'
        auc_col = f'{method}_ROC-AUC'
        
        if time_col in df.columns:
            times = [parse_mean_std(v) for v in df[time_col]]
            aucs = [parse_mean_std(v) for v in df[auc_col]]
            parsed_data[f'{method}_time'] = [t[0] if t[0] is not None else np.nan for t in times]
            parsed_data[f'{method}_time_std'] = [t[1] if t[1] is not None else np.nan for t in times]
            parsed_data[f'{method}_auc'] = [a[0] if a[0] is not None else np.nan for a in aucs]
            parsed_data[f'{method}_auc_std'] = [a[1] if a[1] is not None else np.nan for a in aucs]
    
    # FastLOF variants
    for variant in ['Fastest', 'Slowest', 'BestTradeoff', 'Avg']:
        time_col = f'FastLOF_{variant}_Time'
        auc_col = f'FastLOF_{variant}_ROC-AUC'
        
        if time_col in df.columns:
            times = [parse_mean_std(v) for v in df[time_col]]
            aucs = [parse_mean_std(v) for v in df[auc_col]]
            parsed_data[f'FastLOF_{variant}_time'] = [t[0] if t[0] is not None else np.nan for t in times]
            parsed_data[f'FastLOF_{variant}_time_std'] = [t[1] if t[1] is not None else np.nan for t in times]
            parsed_data[f'FastLOF_{variant}_auc'] = [a[0] if a[0] is not None else np.nan for a in aucs]
            parsed_data[f'FastLOF_{variant}_auc_std'] = [a[1] if a[1] is not None else np.nan for a in aucs]
    
    return pd.DataFrame(parsed_data)

def plot_speedup_comparison(df, threshold, output_dir, dataset_sizes):
    """Plot speedup comparison: FastLOF vs baseline LOF methods."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    datasets = df['Dataset'].values
    # Format dataset labels with size information
    dataset_labels = [format_dataset_label(d, dataset_sizes) for d in datasets]
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    width = 0.15
    
    # Get baseline times (use brute as reference, fallback to others)
    baseline_times = []
    baseline_name = None
    
    for method in ['LOF_brute', 'LOF_kd_tree', 'LOF_ball_tree']:
        time_col = f'{method}_time'
        if time_col in df.columns:
            times = df[time_col].values
            if not np.all(np.isnan(times)):
                baseline_times = times
                baseline_name = method.replace('LOF_', '')
                break
    
    if baseline_times is None or len(baseline_times) == 0:
        print(f"  Warning: No baseline data for {threshold}")
        plt.close(fig)
        return
    
    # Calculate speedups for FastLOF variants
    variants = ['Fastest', 'Avg', 'BestTradeoff']
    colors = ['#2ecc71', '#3498db', '#9b59b6']  # Green, Blue, Purple
    labels = ['Fastest', 'Average', 'Best Tradeoff']
    
    bars = []
    for i, (variant, color, label) in enumerate(zip(variants, colors, labels)):
        time_col = f'FastLOF_{variant}_time'
        if time_col in df.columns:
            fastlof_times = df[time_col].values
            speedups = baseline_times / fastlof_times
            speedups = np.where(np.isnan(speedups) | np.isinf(speedups), 0, speedups)
            
            bar = ax.bar(x + i*width, speedups, width, label=label, color=color, alpha=0.8)
            bars.append(bar)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Speedup vs {baseline_name} (×)', fontsize=12, fontweight='bold')
    ax.set_title(f'FastLOF Speedup Comparison (Threshold = {threshold})', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (1×)')
    
    plt.tight_layout()
    output_path = output_dir / f'fastlof_{threshold}_speedup.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path.name}")

def plot_time_vs_auc_scatter(df, threshold, output_dir, dataset_sizes):
    """Plot Time vs ROC-AUC scatter for all methods."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Baseline methods
    baseline_methods = [
        ('LOF_ball_tree', 'Ball Tree', 'red', '^'),
        ('LOF_kd_tree', 'KD Tree', 'orange', 's'),
        ('LOF_brute', 'Brute Force', 'darkred', 'D'),
    ]
    
    for method_key, method_name, color, marker in baseline_methods:
        time_col = f'{method_key}_time'
        auc_col = f'{method_key}_auc'
        time_std_col = f'{method_key}_time_std'
        auc_std_col = f'{method_key}_auc_std'
        
        if time_col in df.columns and auc_col in df.columns:
            times = df[time_col].values
            aucs = df[auc_col].values
            time_stds = df[time_std_col].values if time_std_col in df.columns else None
            auc_stds = df[auc_std_col].values if auc_std_col in df.columns else None
            
            # Filter out NaN values
            mask = ~(np.isnan(times) | np.isnan(aucs))
            if np.any(mask):
                ax.scatter(times[mask], aucs[mask], 
                          label=method_name, color=color, marker=marker, 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # FastLOF variants
    fastlof_variants = [
        ('Fastest', 'FastLOF Fastest', '#2ecc71', 'o'),
        ('Slowest', 'FastLOF Slowest', '#e74c3c', 'o'),
        ('BestTradeoff', 'FastLOF Best Tradeoff', '#9b59b6', 'o'),
        ('Avg', 'FastLOF Average', '#3498db', 'o'),
    ]
    
    for variant_key, variant_name, color, marker in fastlof_variants:
        time_col = f'FastLOF_{variant_key}_time'
        auc_col = f'FastLOF_{variant_key}_auc'
        
        if time_col in df.columns and auc_col in df.columns:
            times = df[time_col].values
            aucs = df[auc_col].values
            
            # Filter out NaN values
            mask = ~(np.isnan(times) | np.isnan(aucs))
            if np.any(mask):
                ax.scatter(times[mask], aucs[mask], 
                          label=variant_name, color=color, marker=marker, 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
    ax.set_title(f'Time vs ROC-AUC: All Methods (Threshold = {threshold})', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Use log scale for time if values span large range
    times_all = []
    for col in df.columns:
        if '_time' in col and col != 'Dataset':
            times_all.extend(df[col].dropna().values)
    if times_all:
        time_max = max(times_all)
        time_min = min([t for t in times_all if t > 0])
        if time_max / time_min > 100:
            ax.set_xscale('log')
    
    plt.tight_layout()
    output_path = output_dir / f'fastlof_{threshold}_time_vs_auc.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path.name}")

def plot_performance_bars(df, threshold, output_dir, dataset_sizes):
    """Plot grouped bar chart comparing performance metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    datasets = df['Dataset'].values
    # Format dataset labels with size information
    dataset_labels = [format_dataset_label(d, dataset_sizes) for d in datasets]
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    width = 0.12
    
    # Time comparison (left subplot)
    methods = [
        ('LOF_brute', 'Brute', '#e74c3c'),
        ('LOF_kd_tree', 'KD Tree', '#f39c12'),
        ('LOF_ball_tree', 'Ball Tree', '#c0392b'),
        ('FastLOF_Fastest', 'FastLOF Fastest', '#2ecc71'),
        ('FastLOF_Avg', 'FastLOF Avg', '#3498db'),
        ('FastLOF_BestTradeoff', 'FastLOF Best', '#9b59b6'),
    ]
    
    # First, plot all bars with actual values (excluding NaN)
    for i, (method_key, method_name, color) in enumerate(methods):
        time_col = f'{method_key}_time'
        if time_col in df.columns:
            times = df[time_col].values
            # Only plot non-NaN values
            mask = ~np.isnan(times)
            if np.any(mask):
                ax1.bar(x[mask] + i*width, times[mask], width, label=method_name, color=color, alpha=0.8)
    
    ax1.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Runtime (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Runtime Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width * 2.5)
    ax1.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=7)
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set log scale if needed
    if np.any(df[[f'{m[0]}_time' for m in methods if f'{m[0]}_time' in df.columns]].values > 0):
        ax1.set_yscale('log')
    
    # Get y-axis limits after matplotlib sets them
    ax1.relim()
    ax1.autoscale()
    y_min, y_max = ax1.get_ylim()
    
    # Now draw capped bars for missing/failed values at the top of the plot
    for i, (method_key, method_name, color) in enumerate(methods):
        time_col = f'{method_key}_time'
        if time_col in df.columns:
            times = df[time_col].values
            # Find NaN positions (missing/failed runs)
            nan_mask = np.isnan(times)
            if np.any(nan_mask):
                # Draw capped bars at y_max height with distinctive styling
                ax1.bar(x[nan_mask] + i*width, y_max, width, color=color, alpha=0.6, 
                        edgecolor='red', linewidth=2.5, hatch='///', label='_nolegend_')
                # Add horizontal line at top to clearly indicate "capped"
                for idx in np.where(nan_mask)[0]:
                    bar_x = x[idx] + i*width
                    # Draw horizontal line across the top of the bar
                    ax1.plot([bar_x - width/2, bar_x + width/2], 
                            [y_max, y_max], 
                            'r-', linewidth=3, solid_capstyle='round')
    
    # Ensure y-axis extends exactly to the top border where capped bars are
    ax1.set_ylim([y_min, y_max])
    
    # AUC comparison (right subplot)
    for i, (method_key, method_name, color) in enumerate(methods):
        auc_col = f'{method_key}_auc'
        if auc_col in df.columns:
            aucs = df[auc_col].values
            aucs = np.where(np.isnan(aucs), 0, aucs)
            ax2.bar(x + i*width, aucs, width, label=method_name, color=color, alpha=0.8)
    
    ax2.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ROC-AUC', fontsize=11, fontweight='bold')
    ax2.set_title('ROC-AUC Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * 2.5)
    ax2.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=7)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.1])
    
    plt.suptitle(f'Performance Comparison: All Methods (Threshold = {threshold})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / f'fastlof_{threshold}_performance_bars.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path.name}")

def main():
    output_dir = Path("results_summary") / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Get dataset sizes first
    print("Loading dataset sizes...")
    dataset_sizes = get_dataset_sizes()
    print(f"  Loaded sizes for {len(dataset_sizes)} datasets")
    
    thresholds = ['t0', 't1.0', 't1.01', 't1.1', 't1.2']
    
    print("\nGenerating FastLOF summary plots...")
    
    for threshold in thresholds:
        print(f"\n  Processing threshold {threshold}...")
        
        df = load_fastlof_data(threshold, dataset_sizes)
        if df is None or len(df) == 0:
            print(f"    No data found for {threshold}")
            continue
        
        # Generate plots
        plot_speedup_comparison(df, threshold, output_dir, dataset_sizes)
        plot_time_vs_auc_scatter(df, threshold, output_dir, dataset_sizes)
        plot_performance_bars(df, threshold, output_dir, dataset_sizes)
    
    print(f"\nDone! All plots saved to {output_dir}")

if __name__ == "__main__":
    main()

