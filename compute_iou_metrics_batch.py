"""
compute_iou_metrics_batch.py

Retroactively computes IoU-based event-level metrics (Hooge et al.
earliest-match method, per Wang et al. 2025 Section 2.3.3) for every
already-saved *_results.pt found under results/kfold/ — NO RETRAINING
REQUIRED, since raw predictions (preds, labels) are already stored.

Usage (from project root):
    python compute_iou_metrics_batch.py
    python compute_iou_metrics_batch.py --dataset gazecom --model_type skip_attseqnet
    python compute_iou_metrics_batch.py --iou_threshold 0.5
    python compute_iou_metrics_batch.py --out iou_summary.csv
"""

import argparse
import glob
import os
import re

import pandas as pd
import torch

from iou_event_matching import compute_iou_event_metrics

CLASS_NAMES = ["Fixation", "Saccade", "Pursuit", "Blink"]


def _parse_dataset_model_fold(results_path, base_dir):
    """
    results/kfold/<dataset>/<model_type>/fold_N/<prefix>_results.pt
    -> (dataset, model_type, fold_number)
    """
    rel = os.path.relpath(results_path, os.path.join(base_dir, "kfold"))
    parts = rel.split(os.sep)
    dataset, model_type = parts[0], parts[1]
    fname = os.path.basename(results_path)
    m = re.match(r"fold(\d+)_", fname)
    fold_num = int(m.group(1)) if m else None
    return dataset, model_type, fold_num


def find_all_results_pt(base_dir="results"):
    pattern = os.path.join(base_dir, "kfold", "*", "*", "fold_*", "*_results.pt")
    return sorted(glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="Compute IoU-based event-level metrics for every "
                    "saved *_results.pt, without retraining."
    )
    parser.add_argument("--base_dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, nargs="+", default=None)
    parser.add_argument("--model_type", type=str, nargs="+", default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.0,
                        help="0.0 for 'any overlap' matching (Table 4-style), "
                             "0.5 for strict matching (Table 5-style)")
    parser.add_argument("--out", type=str, default="iou_summary.csv")
    args = parser.parse_args()

    all_paths = find_all_results_pt(args.base_dir)
    rows = []

    for path in all_paths:
        dataset, model_type, fold_num = _parse_dataset_model_fold(path, args.base_dir)
        if args.dataset and dataset not in args.dataset:
            continue
        if args.model_type and model_type not in args.model_type:
            continue

        results = torch.load(path, map_location="cpu")
        preds, labels = results["preds"], results["labels"]

        metrics = compute_iou_event_metrics(
            preds, labels, CLASS_NAMES, iou_threshold=args.iou_threshold
        )

        row = {
            "dataset": dataset, "model_type": model_type, "fold": fold_num,
            "iou_threshold": args.iou_threshold,
            "macro_f1_iou": metrics["macro_f1"],
            "macro_iou": metrics["macro_iou"],
        }
        for cname in CLASS_NAMES:
            row[f"f1_iou_{cname}"] = metrics[cname]["f1"]
            row[f"iou_{cname}"] = metrics[cname]["avg_iou"]
            row[f"precision_iou_{cname}"] = metrics[cname]["precision"]
            row[f"recall_iou_{cname}"] = metrics[cname]["recall"]
        rows.append(row)
        print(f"  [{dataset}/{model_type}/fold{fold_num}] "
              f"macro F1(IoU>{args.iou_threshold})={metrics['macro_f1']*100:.2f}%  "
              f"macro IoU={metrics['macro_iou']:.3f}")

    if not rows:
        print("No results.pt files found matching the given filters.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, float_format="%.4f")
    print(f"\nSaved per-fold IoU metrics to: {args.out}")

    # Also print a mean±std summary per (dataset, model_type)
    print("\n" + "=" * 80)
    print(f"SUMMARY (mean ± std across folds, IoU threshold = {args.iou_threshold})")
    print("=" * 80)
    for (dataset, model_type), g in df.groupby(["dataset", "model_type"]):
        print(f"\n[{dataset}] {model_type}  (n_folds={len(g)})")
        print(f"  Macro F1 (IoU): {g['macro_f1_iou'].mean()*100:.2f}% ± {g['macro_f1_iou'].std()*100:.2f}")
        print(f"  Macro IoU     : {g['macro_iou'].mean():.3f} ± {g['macro_iou'].std():.3f}")
        print(f"  F1 SP (IoU)   : {g['f1_iou_Pursuit'].mean()*100:.2f}% ± {g['f1_iou_Pursuit'].std()*100:.2f}")


if __name__ == "__main__":
    main()
