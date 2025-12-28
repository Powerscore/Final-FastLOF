"""
Generate visualization plots for Bounded LOF summary table.
Uses `results_summary/summary_original_lof.csv` and:
- Sorts datasets by size (number of samples)
- Adds dataset shape to labels: "<name> (n_samples, n_features)"
- Compares average single-k LOF (with std) vs Bounded LOF in terms of runtime and ROC-AUC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

from experiments import load_dataset


def get_dataset_sizes():
    """Get dataset sizes (n_samples, n_features) for all datasets."""
    dataset_sizes = {}
    data_dir = Path("data")

    # Map dataset names in results to file names
    dataset_file_map = {
        "annthyroid-unsupervised-ad": "annthyroid-unsupervised-ad.csv",
        "breast-cancer-unsupervised-ad": "breast-cancer-unsupervised-ad.csv",
        "creditcard": "creditcard.csv",
        "dfki-artificial-3000-unsupervised-ad": "dfki-artificial-3000-unsupervised-ad.csv",
        "ForestCover": "ForestCover.mat",
        "http": "http.mat",
        "InternetAds_norm_02_v01": "InternetAds_norm_02_v01.arff",
        "kdd99-unsupervised-ad": "kdd99-unsupervised-ad.csv",
        "mammography": "mammography.mat",
        "mulcross": "mulcross.arff",
        "pen-global-unsupervised-ad": "pen-global-unsupervised-ad.csv",
        "pen-local-unsupervised-ad": "pen-local-unsupervised-ad.csv",
        "PenDigits_withoutdupl_norm_v01": "PenDigits_withoutdupl_norm_v01.arff",
        "satellite-unsupervised-ad": "satellite-unsupervised-ad.csv",
        "shuttle-unsupervised-ad": "shuttle-unsupervised-ad.csv",
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
        match = re.match(r"([\d.]+)\s*±\s*([\d.]+)", str(value_str))
        if match:
            mean = float(match.group(1))
            std = float(match.group(2))
            return mean, std
        else:
            # Try to parse as single number
            return float(value_str), 0.0
    except Exception:
        return None, None


def load_original_lof_summary(dataset_sizes):
    """
    Load and parse Bounded LOF summary table.

    For each dataset, compute:
      - Average single-k LOF across k = {10,20,30,40,50} with std
      - Bounded LOF metrics
    """
    csv_path = Path("results_summary") / "summary_original_lof.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)

    # Sort by dataset size (number of samples)
    if "Dataset" in df.columns:
        df["_sort_size"] = df["Dataset"].apply(
            lambda x: dataset_sizes.get(x, (0, 0))[0]
        )
        df = df.sort_values("_sort_size").reset_index(drop=True)
        df = df.drop("_sort_size", axis=1)

    records = []
    k_values = [10, 20, 30, 40, 50]

    for _, row in df.iterrows():
        dataset = row["Dataset"]

        # Collect per-k metrics
        k_stats = []
        for k in k_values:
            time_col = f"LOF_k{k}_Time"
            auc_col = f"LOF_k{k}_ROC-AUC"
            if time_col in df.columns and auc_col in df.columns:
                t_mean, t_std = parse_mean_std(row[time_col])
                a_mean, a_std = parse_mean_std(row[auc_col])
                if t_mean is not None and a_mean is not None:
                    k_stats.append(
                        {
                            "k": k,
                            "time_mean": t_mean,
                            "time_std": t_std,
                            "auc_mean": a_mean,
                            "auc_std": a_std,
                        }
                    )

        if not k_stats:
            continue

        # Average single-k LOF across all k values
        avg_time_mean = float(np.mean([s["time_mean"] for s in k_stats]))
        avg_time_std = float(np.std([s["time_mean"] for s in k_stats]))
        avg_auc_mean = float(np.mean([s["auc_mean"] for s in k_stats]))
        avg_auc_std = float(np.std([s["auc_mean"] for s in k_stats]))

        # Bounded LOF metrics
        bounded_time_mean, bounded_time_std = parse_mean_std(row["OriginalLOF_Time"])
        bounded_auc_mean, bounded_auc_std = parse_mean_std(row["OriginalLOF_ROC-AUC"])

        records.append(
            {
                "Dataset": dataset,
                "Avg_k_Time": avg_time_mean,
                "Avg_k_Time_Std": avg_time_std,
                "Avg_k_AUC": avg_auc_mean,
                "Avg_k_AUC_Std": avg_auc_std,
                "Bounded_Time": bounded_time_mean,
                "Bounded_Time_Std": bounded_time_std,
                "Bounded_AUC": bounded_auc_mean,
                "Bounded_AUC_Std": bounded_auc_std,
            }
        )

    return pd.DataFrame(records)


def plot_original_lof_comparison(df, dataset_sizes, output_dir):
    """Plot grouped bar chart comparing average single-k LOF (with std) vs Bounded LOF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    datasets = df["Dataset"].values
    labels = [format_dataset_label(d, dataset_sizes) for d in datasets]
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    width = 0.35

    # Runtime comparison
    avg_times = df["Avg_k_Time"].values
    avg_times_std = df["Avg_k_Time_Std"].values
    bounded_times = df["Bounded_Time"].values

    # Average single-k LOF with error bars
    ax1.bar(x - width/2, avg_times, width, yerr=avg_times_std, 
            label="Average single-k LOF", color="#3498db", alpha=0.8, 
            capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    # Bounded LOF
    ax1.bar(x + width/2, bounded_times, width, 
            label="Bounded LOF (k=10-50)", color="#e74c3c", alpha=0.8)

    ax1.set_xlabel("Dataset", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Runtime (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Runtime: Average single-k LOF vs Bounded LOF",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    if np.any(avg_times > 0) or np.any(bounded_times > 0):
        ax1.set_yscale("log")

    # ROC-AUC comparison
    avg_aucs = df["Avg_k_AUC"].values
    avg_aucs_std = df["Avg_k_AUC_Std"].values
    bounded_aucs = df["Bounded_AUC"].values

    # Average single-k LOF with error bars
    ax2.bar(x - width/2, avg_aucs, width, yerr=avg_aucs_std,
            label="Average single-k LOF", color="#3498db", alpha=0.8,
            capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    # Bounded LOF
    ax2.bar(x + width/2, bounded_aucs, width,
            label="Bounded LOF (k=10-50)", color="#e74c3c", alpha=0.8)

    ax2.set_xlabel("Dataset", fontsize=11, fontweight="bold")
    ax2.set_ylabel("ROC-AUC", fontsize=11, fontweight="bold")
    ax2.set_title(
        "ROC-AUC: Average single-k LOF vs Bounded LOF",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_ylim([0, 1.1])

    plt.suptitle(
        "Bounded LOF Experiment Summary\n"
        "(Comparison of average single-k LOF vs Bounded LOF across datasets)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    output_path = output_dir / "original_lof_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = Path("results_summary") / "plots"
    output_dir.mkdir(exist_ok=True)

    print("Loading dataset sizes...")
    dataset_sizes = get_dataset_sizes()
    print(f"  Loaded sizes for {len(dataset_sizes)} datasets")

    print("\nLoading Bounded LOF summary...")
    df = load_original_lof_summary(dataset_sizes)
    if df is None or df.empty:
        print("No Bounded LOF summary data found or parsed.")
        return

    print(f"  Parsed {len(df)} datasets")

    print("\nGenerating Bounded LOF comparison plot...")
    plot_original_lof_comparison(df, dataset_sizes, output_dir)

    print("\nDone! Plot saved in results_summary/plots")


if __name__ == "__main__":
    main()


