#!/usr/bin/env python3
"""
Check completion status of FastLOF experiments
==============================================

This script checks which experiments have completed successfully
by looking for expected output files in the results directory.
"""

import os
from pathlib import Path
from datetime import datetime
import yaml

# Dataset list
DATASETS = [
    "annthyroid-unsupervised-ad",
    "breast-cancer-unsupervised-ad",
    "creditcard",
    "dfki-artificial-3000-unsupervised-ad",
    "InternetAds_norm_02_v01",
    "kdd99-unsupervised-ad",
    "mammography",
    "pen-global-unsupervised-ad",
    "pen-local-unsupervised-ad",
    "PenDigits_withoutdupl_norm_v01",
    "satellite-unsupervised-ad",
    "shuttle-unsupervised-ad",
]

def check_experiment_completion(dataset_name: str, base_path: Path) -> dict:
    """Check if experiment has completed and return status."""
    result_dir = base_path / "results" / dataset_name / "fastlof_experiments"
    
    status = {
        'dataset': dataset_name,
        'completed': False,
        'has_results': False,
        'has_plots': False,
        'threshold_dirs': [],
        'latest_file': None,
        'latest_time': None
    }
    
    if not result_dir.exists():
        return status
    
    # Check for threshold directories
    threshold_dirs = [d for d in result_dir.iterdir() if d.is_dir() and d.name.startswith('threshold_')]
    status['threshold_dirs'] = [d.name for d in threshold_dirs]
    
    if not threshold_dirs:
        return status
    
    # Check latest threshold directory
    latest_threshold = max(threshold_dirs, key=lambda d: d.stat().st_mtime)
    
    # Check for expected files
    expected_files = [
        'results_summary.csv',
        'comparison_plot.png',
        'detailed_results.csv'
    ]
    
    found_files = []
    for fname in expected_files:
        fpath = latest_threshold / fname
        if fpath.exists():
            found_files.append(fname)
            if status['latest_time'] is None or fpath.stat().st_mtime > status['latest_time']:
                status['latest_file'] = fpath
                status['latest_time'] = fpath.stat().st_mtime
    
    status['has_results'] = 'results_summary.csv' in found_files or 'detailed_results.csv' in found_files
    status['has_plots'] = 'comparison_plot.png' in found_files
    status['completed'] = status['has_results'] and status['has_plots']
    
    return status

def format_time_ago(timestamp):
    """Format timestamp as 'X hours/days ago'."""
    if timestamp is None:
        return "Never"
    
    delta = datetime.now().timestamp() - timestamp
    
    if delta < 60:
        return "Just now"
    elif delta < 3600:
        minutes = int(delta / 60)
        return f"{minutes}m ago"
    elif delta < 86400:
        hours = int(delta / 3600)
        return f"{hours}h ago"
    else:
        days = int(delta / 86400)
        return f"{days}d ago"

def main():
    # Determine base path
    script_dir = Path(__file__).parent
    base_path = script_dir.parent
    
    print("=" * 80)
    print("FastLOF Experiment Completion Status")
    print("=" * 80)
    print(f"Checking results in: {base_path / 'results'}")
    print()
    
    # Check all datasets
    all_status = []
    for dataset in DATASETS:
        status = check_experiment_completion(dataset, base_path)
        all_status.append(status)
    
    # Count completion
    completed = sum(1 for s in all_status if s['completed'])
    partial = sum(1 for s in all_status if s['has_results'] and not s['completed'])
    not_started = sum(1 for s in all_status if not s['has_results'])
    
    # Print summary
    print(f"Status Overview:")
    print(f"  ✓ Completed:   {completed:2d}/{len(DATASETS)}")
    print(f"  ⚠ In Progress: {partial:2d}/{len(DATASETS)}")
    print(f"  ✗ Not Started: {not_started:2d}/{len(DATASETS)}")
    print()
    
    # Print detailed status
    print("Detailed Status:")
    print("-" * 80)
    
    for status in all_status:
        dataset = status['dataset']
        
        # Status symbol
        if status['completed']:
            symbol = "✓"
            color = "\033[0;32m"  # Green
        elif status['has_results']:
            symbol = "⚠"
            color = "\033[1;33m"  # Yellow
        else:
            symbol = "✗"
            color = "\033[0;31m"  # Red
        
        nc = "\033[0m"  # No color
        
        # Format output
        dataset_display = dataset[:45].ljust(45)
        time_display = format_time_ago(status['latest_time']).rjust(10)
        
        print(f"{color}{symbol}{nc} {dataset_display} {time_display}", end="")
        
        # Additional info
        if status['threshold_dirs']:
            print(f"  ({len(status['threshold_dirs'])} thresholds)")
        else:
            print()
    
    print("-" * 80)
    print()
    
    # Show in-progress experiments
    in_progress = [s for s in all_status if s['has_results'] and not s['completed']]
    if in_progress:
        print("In Progress (may still be running):")
        for status in in_progress:
            print(f"  • {status['dataset']}")
            if status['threshold_dirs']:
                print(f"    Thresholds: {', '.join(status['threshold_dirs'])}")
        print()
    
    # Show next to run
    not_started_list = [s for s in all_status if not s['has_results']]
    if not_started_list:
        print(f"Not Yet Started ({len(not_started_list)}):")
        for status in not_started_list[:5]:  # Show first 5
            print(f"  • {status['dataset']}")
        if len(not_started_list) > 5:
            print(f"  ... and {len(not_started_list) - 5} more")
        print()
    
    # Progress bar
    progress = completed / len(DATASETS) * 100
    bar_length = 50
    filled = int(bar_length * completed / len(DATASETS))
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"Overall Progress: [{bar}] {progress:.1f}%")
    print()
    
    # Estimate completion
    if partial > 0 or not_started > 0:
        # Try to estimate from config
        config_path = script_dir / "slurm_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            remaining_time_str = []
            for s in not_started_list + in_progress:
                if s['dataset'] in config['datasets']:
                    est = config['datasets'][s['dataset']].get('estimated_time', '')
                    if est:
                        remaining_time_str.append(f"{s['dataset']}: {est}")
            
            if remaining_time_str:
                print("Estimated time remaining for pending jobs:")
                for line in remaining_time_str[:5]:
                    print(f"  • {line}")
                if len(remaining_time_str) > 5:
                    print(f"  ... and {len(remaining_time_str) - 5} more")
                print()
    
    print("=" * 80)
    
    # Exit code: 0 if all complete, 1 otherwise
    return 0 if completed == len(DATASETS) else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
