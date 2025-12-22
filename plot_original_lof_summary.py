"""
Generate visualization plots for Original LOF summary table.
Uses `results_summary/summary_original_lof.csv` and:
- Sorts datasets by size (number of samples)
- Adds dataset shape to labels: "<name> (n_samples, n_features)"
- Compares best single-k LOF vs OriginalLOF in terms of runtime and ROC-AUC
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


def format_dataset_label(dataset_name, dataset_sizes):
    """Format dataset name with shape information in brackets."""
    if dataset_name in dataset_sizes:
        n_samples, n_features = dataset_sizes[dataset_name]
        return f"{dataset_name} ({n_samples:,}, {n_features})"
    return dataset_name


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
    Load and parse Original LOF summary table.

    For each dataset, compute:
      - Best single-k LOF (by ROC-AUC)
      - Average single-k LOF across k = {10,20,30,40,50}
      - OriginalLOF metrics
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

        # Best single-k by ROC-AUC (break ties by smaller k)
        best_k_entry = max(
            k_stats, key=lambda s: (s["auc_mean"], -s["k"])
        )

        # Average single-k LOF across all k values
        avg_time_mean = float(np.mean([s["time_mean"] for s in k_stats]))
        avg_time_std = float(np.std([s["time_mean"] for s in k_stats]))
        avg_auc_mean = float(np.mean([s["auc_mean"] for s in k_stats]))
        avg_auc_std = float(np.std([s["auc_mean"] for s in k_stats]))

        # OriginalLOF metrics
        orig_time_mean, orig_time_std = parse_mean_std(row["OriginalLOF_Time"])
        orig_auc_mean, orig_auc_std = parse_mean_std(row["OriginalLOF_ROC-AUC"])

        records.append(
            {
                "Dataset": dataset,
                "Best_k": best_k_entry["k"],
                "Best_k_Time": best_k_entry["time_mean"],
                "Best_k_Time_Std": best_k_entry["time_std"],
                "Best_k_AUC": best_k_entry["auc_mean"],
                "Best_k_AUC_Std": best_k_entry["auc_std"],
                "Avg_k_Time": avg_time_mean,
                "Avg_k_Time_Std": avg_time_std,
                "Avg_k_AUC": avg_auc_mean,
                "Avg_k_AUC_Std": avg_auc_std,
                "Original_Time": orig_time_mean,
                "Original_Time_Std": orig_time_std,
                "Original_AUC": orig_auc_mean,
                "Original_AUC_Std": orig_auc_std,
            }
        )

    return pd.DataFrame(records)


def plot_original_lof_comparison(df, dataset_sizes, output_dir):
    """Plot grouped bar chart comparing single-k LOF (best & average) vs OriginalLOF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    datasets = df["Dataset"].values
    labels = [format_dataset_label(d, dataset_sizes) for d in datasets]
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    width = 0.28

    # Runtime comparison
    best_times = df["Best_k_Time"].values
    avg_times = df["Avg_k_Time"].values
    orig_times = df["Original_Time"].values

    ax1.bar(x - width, best_times, width, label="Best single-k LOF", color="#3498db")
    ax1.bar(x, avg_times, width, label="Average single-k LOF", color="#95a5a6")
    ax1.bar(x + width, orig_times, width, label="Original LOF (k=10-50)", color="#e74c3c")

    ax1.set_xlabel("Dataset", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Runtime (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Runtime: Single-k LOF (best & avg) vs Original LOF",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    if np.any(best_times > 0) or np.any(orig_times > 0):
        ax1.set_yscale("log")

    # ROC-AUC comparison
    best_aucs = df["Best_k_AUC"].values
    avg_aucs = df["Avg_k_AUC"].values
    orig_aucs = df["Original_AUC"].values

    ax2.bar(x - width, best_aucs, width, label="Best single-k LOF", color="#3498db")
    ax2.bar(x, avg_aucs, width, label="Average single-k LOF", color="#95a5a6")
    ax2.bar(x + width, orig_aucs, width, label="Original LOF (k=10-50)", color="#e74c3c")

    ax2.set_xlabel("Dataset", fontsize=11, fontweight="bold")
    ax2.set_ylabel("ROC-AUC", fontsize=11, fontweight="bold")
    ax2.set_title(
        "ROC-AUC: Single-k LOF (best & avg) vs Original LOF",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_ylim([0, 1.1])

    plt.suptitle(
        "Original LOF Experiment Summary\n"
        "(Comparison of single-k LOF (best & average) vs Original LOF across datasets)",
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

    print("\nLoading Original LOF summary...")
    df = load_original_lof_summary(dataset_sizes)
    if df is None or df.empty:
        print("No Original LOF summary data found or parsed.")
        return

    print(f"  Parsed {len(df)} datasets")

    print("\nGenerating Original LOF comparison plot...")
    plot_original_lof_comparison(df, dataset_sizes, output_dir)

    print("\nDone! Plot saved in results_summary/plots")


if __name__ == "__main__":
    main()


