"""
Generate a comprehensive table of dataset information.
Includes: Dataset Name, Domain, Samples (N), Features (D), Anomalies, Anomaly Rate (%)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from experiments import load_dataset

def get_dataset_domain(dataset_name):
    """Infer domain from dataset name."""
    name_lower = dataset_name.lower()
    
    # Network/Intrusion detection
    if any(x in name_lower for x in ['http', 'kdd', 'internetads']):
        return 'Network/Intrusion'
    
    # Medical/Healthcare
    if any(x in name_lower for x in ['breast', 'cancer', 'mammography', 'annthyroid']):
        return 'Medical/Healthcare'
    
    # Financial
    if 'creditcard' in name_lower:
        return 'Financial'
    
    # Handwriting/Image
    if any(x in name_lower for x in ['pen', 'pendigits']):
        return 'Handwriting/Image'
    
    # Satellite/Remote Sensing
    if 'satellite' in name_lower:
        return 'Satellite/Remote Sensing'
    
    # Space/Shuttle
    if 'shuttle' in name_lower:
        return 'Space/Shuttle'
    
    # Forest/Cover
    if 'forest' in name_lower:
        return 'Forest/Cover'
    
    # Artificial/Synthetic
    if any(x in name_lower for x in ['artificial', 'dfki', 'mulcross']):
        return 'Artificial/Synthetic'
    
    return 'Other'

def main():
    data_dir = Path('data')
    output_dir = Path('results_summary')
    output_dir.mkdir(exist_ok=True)
    
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
    
    print("Loading datasets and collecting information...")
    print("=" * 80)
    
    dataset_info = []
    
    for dataset_name, filename in sorted(dataset_file_map.items()):
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        try:
            print(f"Loading {dataset_name}...", end=' ')
            X, y = load_dataset(str(filepath))
            
            n_samples, n_features = X.shape
            n_anomalies = int(np.sum(y)) if y is not None else 0
            anomaly_rate = (n_anomalies / n_samples * 100) if n_samples > 0 else 0.0
            
            domain = get_dataset_domain(dataset_name)
            
            dataset_info.append({
                'Dataset Name': dataset_name,
                'Domain': domain,
                'Samples (N)': n_samples,
                'Features (D)': n_features,
                'Anomalies': n_anomalies,
                'Anomaly Rate (%)': f"{anomaly_rate:.2f}"
            })
            
            print(f"OK ({n_samples:,} samples, {n_features} features, {n_anomalies} anomalies, {anomaly_rate:.2f}%)")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(dataset_info)
    
    # Sort by number of samples
    df = df.sort_values('Samples (N)').reset_index(drop=True)
    
    # Save as CSV
    csv_path = output_dir / 'dataset_info.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 80)
    print(f"Saved dataset table to: {csv_path}")
    
    # Also create LaTeX table format
    latex_path = output_dir / 'dataset_info.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Dataset Information}\n")
        f.write("\\label{tab:datasets}\n")
        f.write("\\begin{tabular}{l l r r r r}\n")
        f.write("\\toprule\n")
        f.write("Dataset Name & Domain & Samples ($N$) & Features ($D$) & Anomalies & Anomaly Rate (\\%)\\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df.iterrows():
            dataset_name = row['Dataset Name'].replace('_', '\\_')
            domain = row['Domain']
            n_samples = f"{row['Samples (N)']:,}"
            n_features = row['Features (D)']
            n_anomalies = f"{row['Anomalies']:,}"
            anomaly_rate = row['Anomaly Rate (%)']
            
            f.write(f"{dataset_name} & {domain} & {n_samples} & {n_features} & {n_anomalies} & {anomaly_rate}\\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved LaTeX table to: {latex_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Dataset Summary:")
    print(f"  Total datasets: {len(df)}")
    print(f"  Total samples: {df['Samples (N)'].sum():,}")
    print(f"  Total anomalies: {df['Anomalies'].sum():,}")
    print(f"  Average anomaly rate: {df['Anomaly Rate (%)'].astype(float).mean():.2f}%")
    print(f"  Min samples: {df['Samples (N)'].min():,}")
    print(f"  Max samples: {df['Samples (N)'].max():,}")
    print("\nTable preview:")
    print(df.to_string(index=False))
    
    print("\nDone!")

if __name__ == "__main__":
    main()

