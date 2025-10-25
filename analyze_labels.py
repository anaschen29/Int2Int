#!/usr/bin/env python3
"""
Analyze label prediction CSV files to compute accuracy metrics. 
Usage:
    python analyze_labels.py <RESULTS_DIR>

Each CSV in <RESULTS_DIR> (e.g. labels_10.csv) should contain:
    row_id,group_id,task,rewrite,label,prediction
"""

import argparse
import sys
import os
import glob
import pandas as pd

def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    # sanity check
    if not {"label", "prediction"}.issubset(df.columns):
        raise ValueError("CSV must contain 'label' and 'prediction' columns.")

    # normalize whitespace
    df["label"] = df["label"].astype(str).str.strip()
    df["prediction"] = df["prediction"].astype(str).str.strip()
    df["correct"] = (df["label"] == df["prediction"])

    overall_acc = df["correct"].mean()
    print(f"Overall accuracy: {overall_acc:.2%} ({df['correct'].sum()}/{len(df)})")

    # Accuracy per rewrite (if available, idk if it makes much sense)
    if "rewrite" in df.columns:
        acc_rewrite = df.groupby("rewrite")["correct"].mean().reset_index()
        print("\nAccuracy per rewrite:")
        print(acc_rewrite)
    else:
        acc_rewrite = pd.DataFrame()

    # Accuracy per group (if available)
    if "group_id" in df.columns:
        acc_group = df.groupby("group_id")["correct"].mean().reset_index()
        print(f"\n{len(acc_group)} groups found. Example:")
        print(acc_group.head())
    else:
        acc_group = pd.DataFrame()

    # --- Symmetry metrics ---
    if "group_id" in df.columns:
        def group_symmetry(subdf):
            preds = subdf["prediction"].tolist()
            unique_preds = len(set(preds))
            inv_unique = 1.0 / unique_preds
            mode_count = subdf["prediction"].value_counts().max()
            mode_ratio = mode_count / len(subdf)
            return pd.Series({
                "inv_unique_preds": inv_unique,
                "mode_ratio": mode_ratio
            })

        sym_group = (
            df.groupby("group_id", group_keys=False)[["prediction"]]
            .apply(group_symmetry)
            .reset_index()
        )
        print(f"\nSymmetry metrics per group (examples):")
        print(sym_group.head())
    else:
        sym_group = pd.DataFrame()

    # return all summaries
    return acc_rewrite, acc_group, sym_group


def main():
    parser = argparse.ArgumentParser(description="Analyze label prediction CSV files.")
    parser.add_argument("results_dir", help="Directory containing labels_*.csv files.")
    parser.add_argument(
        "--pattern",
        default="labels_*.csv",
        help="Filename pattern to search for (default: labels_*.csv)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.rstrip("/")
    csv_paths = sorted(glob.glob(os.path.join(results_dir, args.pattern)))

    if not csv_paths:
        print(f"No {args.pattern} files found in {results_dir}")
        return

    for path in csv_paths:
        print("-" * 80)
        print(f"Analyzing {os.path.basename(path)}")
        df = pd.read_csv(path)
        acc_rewrite, acc_group, sym_group = compute_accuracy(df)

        epoch = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        out_path = os.path.join(results_dir, f"summary_{epoch}.csv")

        # ---- Sectioned output with group averages ----
        with open(out_path, "w", encoding="utf-8") as f:
            if not acc_rewrite.empty:
                f.write("# Accuracy per rewrite\n")
                acc_rewrite.to_csv(f, index=False)
                f.write("\n")

            if not acc_group.empty:
                f.write("# Accuracy per group\n")
                acc_group.to_csv(f, index=False)
                avg_acc = acc_group["correct"].mean()
                f.write(f"\n# Group average accuracy,{avg_acc:.4f}\n\n")

            if not sym_group.empty:
                f.write("# Symmetry metrics\n")
                sym_group.to_csv(f, index=False)
                avg_inv = sym_group["inv_unique_preds"].mean()
                avg_mode = sym_group["mode_ratio"].mean()
                f.write(f"\n# Group average symmetry,inv_unique_avg={avg_inv:.4f},mode_ratio_avg={avg_mode:.4f}\n")

        print(f"Saved {out_path}\n")



if __name__ == "__main__":
    main()