"""
Parse experiment results CSV files and create summary tables.
- One table for Original LOF experiment (one row per dataset, columns for each k)
- Five tables for FastLOF experiments (one per threshold, one row per dataset, columns for each algorithm/metric)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def format_value_with_std(avg, std, decimals=4):
    """Format value as 'avg ± std'."""
    if pd.isna(avg) or pd.isna(std):
        return "N/A"
    return f"{avg:.{decimals}f} ± {std:.{decimals}f}"

def parse_original_lof_csv(csv_path):
    """Parse Original LOF results CSV and return all rows."""
    try:
        df = pd.read_csv(csv_path)
        rows = []
        
        for _, row in df.iterrows():
            k_val = row.get('k', '')
            k_range = row.get('k_range', '')
            
            # Check if ROC_AUC_Std exists
            roc_auc_std = row.get('ROC_AUC_Std', 0.0)
            if pd.isna(roc_auc_std):
                roc_auc_std = 0.0
            
            rows.append({
                'Algorithm': row['Algorithm'],
                'k': k_val if pd.notna(k_val) and k_val != '' else None,
                'k_range': k_range if pd.notna(k_range) and k_range != '' else None,
                'runtime_avg': row['Runtime_Avg'],
                'runtime_std': row['Runtime_Std'],
                'roc_auc': row['ROC_AUC'],
                'roc_auc_std': roc_auc_std,
            })
        
        return rows
    except Exception as e:
        print(f"Error parsing {csv_path}: {e}")
        return []

def parse_fastlof_csv(csv_path):
    """Parse FastLOF results CSV and return all rows."""
    try:
        df = pd.read_csv(csv_path)
        rows = []
        
        for _, row in df.iterrows():
            algorithm = row['Algorithm']
            chunk_size = row.get('Chunk_Size', 'N/A')
            
            # Check if ROC_AUC_Std exists
            roc_auc_std = row.get('ROC_AUC_Std', 0.0)
            if pd.isna(roc_auc_std):
                roc_auc_std = 0.0
            
            rows.append({
                'Algorithm': algorithm,
                'Chunk_Size': chunk_size,
                'runtime_avg': row['Runtime_Avg'],
                'runtime_std': row['Runtime_Std'],
                'roc_auc': row['ROC_AUC'],
                'roc_auc_std': roc_auc_std,
            })
        
        return rows
    except Exception as e:
        print(f"Error parsing {csv_path}: {e}")
        return []

def aggregate_fastlof_metrics(fastlof_rows):
    """Aggregate FastLOF rows to get fastest, slowest, best tradeoff, and average."""
    if not fastlof_rows:
        return None
    
    # Fastest (lowest runtime)
    fastest_idx = min(range(len(fastlof_rows)), key=lambda i: fastlof_rows[i]['runtime_avg'])
    fastest = fastlof_rows[fastest_idx]
    
    # Slowest (highest runtime)
    slowest_idx = max(range(len(fastlof_rows)), key=lambda i: fastlof_rows[i]['runtime_avg'])
    slowest = fastlof_rows[slowest_idx]
    
    # Best tradeoff (highest AUC/time ratio)
    tradeoff_scores = []
    for row in fastlof_rows:
        if row['runtime_avg'] > 0:
            ratio = row['roc_auc'] / row['runtime_avg']
            tradeoff_scores.append(ratio)
        else:
            tradeoff_scores.append(0)
    best_tradeoff_idx = max(range(len(fastlof_rows)), key=lambda i: tradeoff_scores[i])
    best_tradeoff = fastlof_rows[best_tradeoff_idx]
    
    # Average (mean across all chunk sizes)
    runtime_avgs = [r['runtime_avg'] for r in fastlof_rows]
    runtime_stds = [r['runtime_std'] for r in fastlof_rows]
    roc_auc_avgs = [r['roc_auc'] for r in fastlof_rows]
    roc_auc_stds = [r['roc_auc_std'] for r in fastlof_rows]
    
    # For average, compute mean of means and std of means
    avg_runtime_avg = np.mean(runtime_avgs)
    avg_runtime_std = np.std(runtime_avgs)  # Std of the means
    avg_roc_auc_avg = np.mean(roc_auc_avgs)
    avg_roc_auc_std = np.std(roc_auc_avgs)  # Std of the means
    
    return {
        'fastest': fastest,
        'slowest': slowest,
        'best_tradeoff': best_tradeoff,
        'avg': {
            'runtime_avg': avg_runtime_avg,
            'runtime_std': avg_runtime_std,
            'roc_auc': avg_roc_auc_avg,
            'roc_auc_std': avg_roc_auc_std,
        }
    }

def main():
    results_dir = Path("results")
    
    # Store Original LOF results: {dataset: [rows]}
    original_lof_results = defaultdict(list)
    
    # Store FastLOF results: {threshold: {dataset: [rows]}}
    fastlof_results = defaultdict(lambda: defaultdict(list))
    
    # Get all dataset directories
    dataset_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    print("Parsing CSV files...")
    
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f"  Processing {dataset_name}...")
        
        # Parse Original LOF results
        original_lof_csv = dataset_dir / "lof_experiments" / "k10-50" / "original_lof_results.csv"
        if original_lof_csv.exists():
            rows = parse_original_lof_csv(original_lof_csv)
            if rows:
                original_lof_results[dataset_name] = rows
        
        # Parse FastLOF results for each threshold
        fastlof_dir = dataset_dir / "fastlof_experiments"
        if fastlof_dir.exists():
            for threshold_dir in fastlof_dir.iterdir():
                if threshold_dir.is_dir():
                    threshold = threshold_dir.name.split('_')[-1]  # Extract 't0', 't1.0', etc.
                    fastlof_csv = threshold_dir / "fastlof_results.csv"
                    
                    if fastlof_csv.exists():
                        rows = parse_fastlof_csv(fastlof_csv)
                        if rows:
                            fastlof_results[threshold][dataset_name] = rows
    
    # Create output directory
    output_dir = Path("results_summary")
    output_dir.mkdir(exist_ok=True)
    
    # ============================================================================
    # Create Original LOF summary table (one row per dataset)
    # ============================================================================
    print("\nGenerating Original LOF summary table...")
    
    original_lof_rows = []
    all_datasets = sorted(original_lof_results.keys())
    
    for dataset in all_datasets:
        rows_data = original_lof_results[dataset]
        
        # Build row with columns for each k value
        row = {'Dataset': dataset}
        
        # Extract data for each k value
        for k_val in [10, 20, 30, 40, 50]:
            k_rows = [r for r in rows_data if r['k'] == k_val and r['Algorithm'] == 'LOF']
            if k_rows:
                k_data = k_rows[0]
                row[f'LOF_k{k_val}_Time'] = format_value_with_std(
                    k_data['runtime_avg'], k_data['runtime_std'], decimals=4
                )
                row[f'LOF_k{k_val}_ROC-AUC'] = format_value_with_std(
                    k_data['roc_auc'], k_data['roc_auc_std'], decimals=4
                )
            else:
                row[f'LOF_k{k_val}_Time'] = "N/A"
                row[f'LOF_k{k_val}_ROC-AUC'] = "N/A"
        
        # Extract OriginalLOF data
        orig_rows = [r for r in rows_data if r['Algorithm'] == 'OriginalLOF']
        if orig_rows:
            orig_data = orig_rows[0]
            row['OriginalLOF_Time'] = format_value_with_std(
                orig_data['runtime_avg'], orig_data['runtime_std'], decimals=4
            )
            row['OriginalLOF_ROC-AUC'] = format_value_with_std(
                orig_data['roc_auc'], orig_data['roc_auc_std'], decimals=4
            )
        else:
            row['OriginalLOF_Time'] = "N/A"
            row['OriginalLOF_ROC-AUC'] = "N/A"
        
        original_lof_rows.append(row)
    
    if original_lof_rows:
        # Define column order
        column_order = ['Dataset']
        for k in [10, 20, 30, 40, 50]:
            column_order.extend([f'LOF_k{k}_Time', f'LOF_k{k}_ROC-AUC'])
        column_order.extend(['OriginalLOF_Time', 'OriginalLOF_ROC-AUC'])
        
        df_original = pd.DataFrame(original_lof_rows)
        # Reorder columns
        df_original = df_original.reindex(columns=column_order, fill_value="N/A")
        output_file = output_dir / "summary_original_lof.csv"
        df_original.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  Saved to {output_file}")
        print(f"  Total rows: {len(original_lof_rows)}")
    
    # ============================================================================
    # Create FastLOF summary tables (one row per dataset)
    # ============================================================================
    print("\nGenerating FastLOF summary tables...")
    
    thresholds = sorted(fastlof_results.keys())
    
    for threshold in thresholds:
        print(f"  Creating table for {threshold}...")
        
        fastlof_summary_rows = []
        datasets_for_threshold = sorted(fastlof_results[threshold].keys())
        
        for dataset in datasets_for_threshold:
            rows_data = fastlof_results[threshold][dataset]
            
            # Separate baseline LOF and FastLOF rows
            baseline_rows = {r['Algorithm']: r for r in rows_data if r['Algorithm'].startswith('LOF_')}
            fastlof_rows = [r for r in rows_data if r['Algorithm'] == 'FastLOF']
            
            # Build row
            row = {'Dataset': dataset}
            
            # Baseline LOF algorithms
            for algo_name in ['LOF_ball_tree', 'LOF_kd_tree', 'LOF_brute']:
                if algo_name in baseline_rows:
                    algo_data = baseline_rows[algo_name]
                    row[f'{algo_name}_Time'] = format_value_with_std(
                        algo_data['runtime_avg'], algo_data['runtime_std'], decimals=4
                    )
                    row[f'{algo_name}_ROC-AUC'] = format_value_with_std(
                        algo_data['roc_auc'], algo_data['roc_auc_std'], decimals=4
                    )
                else:
                    row[f'{algo_name}_Time'] = "N/A"
                    row[f'{algo_name}_ROC-AUC'] = "N/A"
            
            # FastLOF aggregations
            if fastlof_rows:
                aggregated = aggregate_fastlof_metrics(fastlof_rows)
                
                # Fastest
                row['FastLOF_Fastest_Time'] = format_value_with_std(
                    aggregated['fastest']['runtime_avg'],
                    aggregated['fastest']['runtime_std'],
                    decimals=4
                )
                row['FastLOF_Fastest_ROC-AUC'] = format_value_with_std(
                    aggregated['fastest']['roc_auc'],
                    aggregated['fastest']['roc_auc_std'],
                    decimals=4
                )
                
                # Slowest
                row['FastLOF_Slowest_Time'] = format_value_with_std(
                    aggregated['slowest']['runtime_avg'],
                    aggregated['slowest']['runtime_std'],
                    decimals=4
                )
                row['FastLOF_Slowest_ROC-AUC'] = format_value_with_std(
                    aggregated['slowest']['roc_auc'],
                    aggregated['slowest']['roc_auc_std'],
                    decimals=4
                )
                
                # Best tradeoff
                row['FastLOF_BestTradeoff_Time'] = format_value_with_std(
                    aggregated['best_tradeoff']['runtime_avg'],
                    aggregated['best_tradeoff']['runtime_std'],
                    decimals=4
                )
                row['FastLOF_BestTradeoff_ROC-AUC'] = format_value_with_std(
                    aggregated['best_tradeoff']['roc_auc'],
                    aggregated['best_tradeoff']['roc_auc_std'],
                    decimals=4
                )
                
                # Average
                row['FastLOF_Avg_Time'] = format_value_with_std(
                    aggregated['avg']['runtime_avg'],
                    aggregated['avg']['runtime_std'],
                    decimals=4
                )
                row['FastLOF_Avg_ROC-AUC'] = format_value_with_std(
                    aggregated['avg']['roc_auc'],
                    aggregated['avg']['roc_auc_std'],
                    decimals=4
                )
            else:
                # No FastLOF data
                for metric in ['Fastest', 'Slowest', 'BestTradeoff', 'Avg']:
                    row[f'FastLOF_{metric}_Time'] = "N/A"
                    row[f'FastLOF_{metric}_ROC-AUC'] = "N/A"
            
            fastlof_summary_rows.append(row)
        
        if fastlof_summary_rows:
            # Define column order
            column_order = ['Dataset']
            column_order.extend([
                'LOF_ball_tree_Time', 'LOF_ball_tree_ROC-AUC',
                'LOF_kd_tree_Time', 'LOF_kd_tree_ROC-AUC',
                'LOF_brute_Time', 'LOF_brute_ROC-AUC',
                'FastLOF_Fastest_Time', 'FastLOF_Fastest_ROC-AUC',
                'FastLOF_Slowest_Time', 'FastLOF_Slowest_ROC-AUC',
                'FastLOF_BestTradeoff_Time', 'FastLOF_BestTradeoff_ROC-AUC',
                'FastLOF_Avg_Time', 'FastLOF_Avg_ROC-AUC',
            ])
            
            df_fastlof = pd.DataFrame(fastlof_summary_rows)
            # Reorder columns
            df_fastlof = df_fastlof.reindex(columns=column_order, fill_value="N/A")
            output_file = output_dir / f"summary_fastlof_{threshold}.csv"
            df_fastlof.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"    Saved to {output_file}")
            print(f"    Total rows: {len(fastlof_summary_rows)}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
