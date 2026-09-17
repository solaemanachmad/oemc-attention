"""
aggregate_results.py

Scans results/kfold/<dataset>/<model_type>/fold_*/ for the *_metrics.csv
file with the MOST rows (since train_model()'s all_metrics list grows
across folds and is re-saved every fold, the file in the LAST fold's
folder already contains all folds' rows — no need to merge multiple
files manually).

For each (dataset, model_type) found, computes mean +/- std across
folds for every numeric metric column, and assembles one combined
table ready for the paper's main Results section.

Usage (from project root):
    python aggregate_results.py
    python aggregate_results.py --base_dir results --dataset gazecom
    python aggregate_results.py --out results_summary.csv
"""

import argparse
import glob
import os

import pandas as pd
import numpy as np


def _find_best_metrics_csv(dataset_model_dir):
    """
    Within results/kfold/<dataset>/<model_type>/, find the fold_N
    subfolder whose *_metrics.csv has the most rows (= the most
    complete cumulative record), and return its path + row count.
    """
    candidates = glob.glob(os.path.join(dataset_model_dir, "fold_*", "*_metrics.csv"))
    if not candidates:
        return None, 0

    best_path, best_rows = None, -1
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if len(df) > best_rows:
            best_path, best_rows = path, len(df)

    return best_path, best_rows


def discover_runs(base_dir="results"):
    """
    Yields (dataset, model_type, metrics_csv_path, n_rows) for every
    dataset/model_type combination found under base_dir/kfold/.
    """
    kfold_root = os.path.join(base_dir, "kfold")
    if not os.path.isdir(kfold_root):
        return

    for dataset in sorted(os.listdir(kfold_root)):
        dataset_dir = os.path.join(kfold_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for model_type in sorted(os.listdir(dataset_dir)):
            model_dir = os.path.join(dataset_dir, model_type)
            if not os.path.isdir(model_dir):
                continue
            path, n_rows = _find_best_metrics_csv(model_dir)
            if path is not None:
                yield dataset, model_type, path, n_rows


# Columns to summarize as mean +/- std, in the order they should
# appear in the final table.
METRIC_COLS = [
    "F1_avg", "F1_Fixation", "F1_Saccade", "F1_Pursuit", "F1_Blink",
    "ev_F1_avg", "ev_F1_Fixation", "ev_F1_Saccade", "ev_F1_Pursuit", "ev_F1_Blink",
    "roc_auc_macro", "roc_auc_micro",
    "val_loss", "train_loss", "epochs_run",
]


def summarize(df, n_expected_folds=None):
    """Return a dict of {col: (mean, std)} for every metric column present."""
    summary = {}
    for col in METRIC_COLS:
        if col in df.columns:
            summary[col] = (df[col].mean(), df[col].std())
    summary["_n_folds"] = len(df)
    if n_expected_folds is not None and len(df) != n_expected_folds:
        summary["_incomplete"] = True
    else:
        summary["_incomplete"] = False
    return summary


def build_summary_table(base_dir="results", dataset_filter=None,
                        model_filter=None, n_expected_folds=5):
    rows = []
    for dataset, model_type, path, n_rows in discover_runs(base_dir):
        if dataset_filter and dataset not in dataset_filter:
            continue
        if model_filter and model_type not in model_filter:
            continue

        df = pd.read_csv(path)
        summary = summarize(df, n_expected_folds)

        row = {"dataset": dataset, "model_type": model_type,
               "n_folds_found": summary["_n_folds"],
               "incomplete": summary["_incomplete"],
               "source_file": path}
        for col in METRIC_COLS:
            if col in summary:
                mean, std = summary[col]
                row[f"{col}_mean"] = mean
                row[f"{col}_std"] = std
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["dataset", "model_type"]).reset_index(drop=True)
    return result


def print_readable(df):
    """Print a compact, human-readable view (macro F1 sample+event, per model)."""
    if df.empty:
        print("No results found.")
        return

    print("=" * 100)
    print("SUMMARY (mean ± std across folds)")
    print("=" * 100)
    for _, r in df.iterrows():
        flag = "  [INCOMPLETE]" if r.get("incomplete") else ""
        print(f"\n[{r['dataset']}] {r['model_type']}{flag} "
              f"(n_folds={r['n_folds_found']})")
        print(f"  Source: {r['source_file']}")
        if "F1_avg_mean" in r:
            print(f"  F1-macro (sample): {r['F1_avg_mean']*100:.2f}% ± {r['F1_avg_std']*100:.2f}")
        if "ev_F1_avg_mean" in r:
            print(f"  F1-macro (event) : {r['ev_F1_avg_mean']*100:.2f}% ± {r['ev_F1_avg_std']*100:.2f}")
        if "F1_Pursuit_mean" in r:
            print(f"  F1 SP (sample)   : {r['F1_Pursuit_mean']*100:.2f}% ± {r['F1_Pursuit_std']*100:.2f}")
        if "ev_F1_Pursuit_mean" in r:
            print(f"  F1 SP (event)    : {r['ev_F1_Pursuit_mean']*100:.2f}% ± {r['ev_F1_Pursuit_std']*100:.2f}")
        if "roc_auc_macro_mean" in r:
            print(f"  ROC-AUC (macro)  : {r['roc_auc_macro_mean']:.4f} ± {r['roc_auc_macro_std']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate k-fold results across all models/datasets "
                    "into one mean±std summary table for the paper."
    )
    parser.add_argument("--base_dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="Only include these dataset(s)")
    parser.add_argument("--model_type", type=str, nargs="+", default=None,
                        help="Only include these model_type(s)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Expected number of folds, used to flag incomplete runs")
    parser.add_argument("--out", type=str, default="results_summary.csv",
                        help="Where to save the full summary CSV")
    args = parser.parse_args()

    df = build_summary_table(args.base_dir, args.dataset, args.model_type, args.n_splits)
    print_readable(df)

    if not df.empty:
        df.to_csv(args.out, index=False, float_format="%.4f")
        print(f"\nFull summary saved to: {args.out}")

        incomplete = df[df["incomplete"]]
        if not incomplete.empty:
            print("\n⚠️  WARNING — incomplete runs detected (fewer folds than expected):")
            for _, r in incomplete.iterrows():
                print(f"   [{r['dataset']}] {r['model_type']}: "
                      f"only {r['n_folds_found']}/{args.n_splits} folds found")


if __name__ == "__main__":
    main()