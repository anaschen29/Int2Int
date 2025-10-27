#!/usr/bin/env python3
"""
Analyze label prediction CSV files to compute accuracy metrics (machine-readable outputs).

Usage:
    python analyze_labels.py <RESULTS_DIR> [--pattern labels_*.csv] [--no-aggregates]

Outputs per epoch (E):
  - summary_E_overview.csv              (total, correct, overall_accuracy)
  - summary_E_accuracy_rewrite.csv      (rewrite, accuracy)                [if rewrite present]
  - summary_E_accuracy_group.csv        (group_id, accuracy + __AVG__)     [if group_id present]
  - summary_E_symmetry_group.csv        (group_id, inv_unique_preds, mode_ratio + __AVG__) [if group_id present]

Also writes all-epochs aggregates when >1 input unless --no-aggregates is set:
  - all_overview.csv
  - all_accuracy_rewrite.csv
  - all_accuracy_group.csv
  - all_symmetry_group.csv
"""

import argparse
import os
import glob
import pandas as pd

def _safe_round(series, ndigits=4):
    return series.astype(float).round(ndigits)

def compute_tables(df: pd.DataFrame):
    """Return four DataFrames: overview, per-rewrite, per-group acc, per-group symmetry."""
    # sanity check
    if not {"label", "prediction"}.issubset(df.columns):
        raise ValueError("CSV must contain 'label' and 'prediction' columns.")

    df = df.copy()
    # normalize whitespace
    df["label"] = df["label"].astype(str).str.strip()
    df["prediction"] = df["prediction"].astype(str).str.strip()
    df["correct"] = (df["label"] == df["prediction"])

    total = len(df)
    correct = int(df["correct"].sum())
    overall_acc = correct / total if total else float("nan")
    overview = pd.DataFrame([{
        "total": total,
        "correct": correct,
        "overall_accuracy": round(overall_acc, 4)
    }])

    # per-rewrite
    if "rewrite" in df.columns:
        acc_rewrite = (df.groupby("rewrite", dropna=False)["correct"]
                         .mean()
                         .reset_index()
                         .rename(columns={"correct": "accuracy"}))
        acc_rewrite["accuracy"] = _safe_round(acc_rewrite["accuracy"])
        acc_rewrite = acc_rewrite.sort_values(["rewrite"]).reset_index(drop=True)
    else:
        acc_rewrite = pd.DataFrame(columns=["rewrite", "accuracy"])

    # per-group + symmetry
    if "group_id" in df.columns:
        acc_group = (df.groupby("group_id", dropna=False)["correct"]
                       .mean()
                       .reset_index()
                       .rename(columns={"correct": "accuracy"}))
        acc_group["accuracy"] = _safe_round(acc_group["accuracy"])
        acc_group = acc_group.sort_values(["group_id"]).reset_index(drop=True)

        # symmetry metrics
        def group_symmetry(subdf):
            n = len(subdf)
            if n == 0:
                return pd.Series({"inv_unique_preds": 0.0, "mode_ratio": 0.0})
            preds = subdf["prediction"].tolist()
            unique_preds = len(set(preds))
            inv_unique = 1.0 / unique_preds if unique_preds > 0 else 0.0
            mode_count = subdf["prediction"].value_counts().max()
            mode_ratio = mode_count / n
            return pd.Series({"inv_unique_preds": inv_unique, "mode_ratio": mode_ratio})

        sym_group = (df.groupby("group_id", dropna=False)[["prediction"]]
                       .apply(group_symmetry)
                       .reset_index())
        sym_group["inv_unique_preds"] = _safe_round(sym_group["inv_unique_preds"])
        sym_group["mode_ratio"] = _safe_round(sym_group["mode_ratio"])
        sym_group = sym_group.sort_values(["group_id"]).reset_index(drop=True)
    else:
        acc_group = pd.DataFrame(columns=["group_id", "accuracy"])
        sym_group = pd.DataFrame(columns=["group_id", "inv_unique_preds", "mode_ratio"])

    # add AVG rows (easy to parse, no comments)
    if not acc_group.empty:
        acc_group = pd.concat(
            [acc_group,
             pd.DataFrame([{"group_id": "__AVG__", "accuracy": round(acc_group["accuracy"].mean(), 4)}])],
            ignore_index=True
        )
    if not sym_group.empty:
        sym_group = pd.concat(
            [sym_group,
             pd.DataFrame([{
                 "group_id": "__AVG__",
                 "inv_unique_preds": round(sym_group["inv_unique_preds"].mean(), 4),
                 "mode_ratio": round(sym_group["mode_ratio"].mean(), 4),
             }])],
            ignore_index=True
        )

    return overview, acc_rewrite, acc_group, sym_group

def main():
    parser = argparse.ArgumentParser(description="Analyze label prediction CSV files (machine-readable outputs).")
    parser.add_argument("results_dir", help="Directory containing labels_*.csv files.")
    parser.add_argument("--pattern", default="labels_*.csv",
                        help="Filename pattern to search for (default: labels_*.csv)")
    parser.add_argument("--no-aggregates", action="store_true",
                        help="Do not write all-epochs aggregate CSVs.")
    args = parser.parse_args()

    results_dir = args.results_dir.rstrip("/")
    csv_paths = sorted(glob.glob(os.path.join(results_dir, args.pattern)))
    if not csv_paths:
        print(f"No {args.pattern} files found in {results_dir}")
        return

    # holders for all-epochs aggregates
    agg_overview = []
    agg_rewrite = []
    agg_group = []
    agg_sym = []

    for path in csv_paths:
        fname = os.path.basename(path)
        # epoch = trailing token after last underscore (fallback to full stem)
        stem = os.path.splitext(fname)[0]
        parts = stem.split("_")
        epoch = parts[-1] if len(parts) > 1 else stem

        print("-" * 80)
        print(f"Analyzing {fname} (epoch={epoch})")

        df = pd.read_csv(path)
        overview, acc_rewrite, acc_group, sym_group = compute_tables(df)

        # per-epoch outputs (pure CSVs, one table per file)
        pd.options.display.float_format = "{:0.4f}".format  # for any printouts (doesn't affect CSV types)
        base = os.path.join(results_dir, f"summary_{epoch}")

        overview.to_csv(base + "_overview.csv", index=False)
        if not acc_rewrite.empty:
            acc_rewrite.to_csv(base + "_accuracy_rewrite.csv", index=False)
        if not acc_group.empty:
            acc_group.to_csv(base + "_accuracy_group.csv", index=False)
        if not sym_group.empty:
            sym_group.to_csv(base + "_symmetry_group.csv", index=False)

        print(f"Saved summary CSVs for epoch {epoch}")

        # stash for aggregates
        o = overview.copy(); o.insert(0, "epoch", epoch); agg_overview.append(o)
        if not acc_rewrite.empty:
            r = acc_rewrite.copy(); r.insert(0, "epoch", epoch); agg_rewrite.append(r)
        if not acc_group.empty:
            g = acc_group.copy(); g.insert(0, "epoch", epoch); agg_group.append(g)
        if not sym_group.empty:
            s = sym_group.copy(); s.insert(0, "epoch", epoch); agg_sym.append(s)

    # all-epochs aggregates (unless disabled or only one file)
    multiple = len(csv_paths) > 1
    if multiple and not args.no-aggregates:
        if agg_overview:
            pd.concat(agg_overview, ignore_index=True) \
              .sort_values(["epoch"]).to_csv(os.path.join(results_dir, "all_overview.csv"), index=False)
        if agg_rewrite:
            pd.concat(agg_rewrite, ignore_index=True) \
              .sort_values(["epoch", "rewrite"]).to_csv(os.path.join(results_dir, "all_accuracy_rewrite.csv"), index=False)
        if agg_group:
            pd.concat(agg_group, ignore_index=True) \
              .sort_values(["epoch", "group_id"]).to_csv(os.path.join(results_dir, "all_accuracy_group.csv"), index=False)
        if agg_sym:
            pd.concat(agg_sym, ignore_index=True) \
              .sort_values(["epoch", "group_id"]).to_csv(os.path.join(results_dir, "all_symmetry_group.csv"), index=False)
        print("Saved all-epochs aggregate CSVs")

if __name__ == "__main__":
    main()
