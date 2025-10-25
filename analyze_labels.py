#!/usr/bin/env python3
"""
Analyze label prediction CSV files to compute accuracy metrics. 
Usage:
    python analyze_labels.py <RESULTS_DIR>

Each CSV in <RESULTS_DIR> (e.g. labels_10.csv) should contain:
    row_id,group_id,task,rewrite,label,prediction
"""


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
    if len(sys.argv) < 2:
        print("Usage: python analyze_labels.py <RESULTS_DIR>")
        sys.exit(1)

    results_dir = sys.argv[1].rstrip("/")
    csv_paths = sorted(glob.glob(os.path.join(results_dir, "labels_*.csv")))

    if not csv_paths:
        print(f"No labels_*.csv files found in {results_dir}")
        sys.exit(1)

    for path in csv_paths:
        print("=" * 80)
        print(f"Analyzing {os.path.basename(path)}")
        df = pd.read_csv(path)
        acc_rewrite, acc_group, sym_group = compute_accuracy(df)

        epoch = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        out_path = os.path.join(results_dir, f"summary_{epoch}.csv")

        summary_parts = []
        if not acc_rewrite.empty:
            acc_rewrite["type"] = "rewrite"
            summary_parts.append(acc_rewrite)
        if not acc_group.empty:
            acc_group["type"] = "group_acc"
            summary_parts.append(acc_group)
        if not sym_group.empty:
            sym_group["type"] = "symmetry"
            summary_parts.append(sym_group)

        if summary_parts:
            pd.concat(summary_parts).to_csv(out_path, index=False)
            print(f"Saved {out_path}\n")
        else:
            print("No rewrite, group, or symmetry columns found; summary not saved.\n")




if __name__ == "__main__":
    main()