"""
regenerate_plot.py

Batch version of regenerate_plots.py — walks results/kfold/ (and
results/single/ if present) and regenerates confusion matrix (+
optionally ROC/PR) plots for EVERY *_results.pt file found, inferring
dataset / model_type / fold_idx from the folder structure itself:

    results/kfold/<dataset>/<model_type>/fold_N/<prefix>_results.pt
    results/single/<dataset>/<model_type>/<prefix>_results.pt

No retraining, no model loading — reuses the saved predictions.

Usage (from project root):
    python regenerate_plot.py
    python regenerate_plot.py --base_dir results --skip_roc_pr
    python regenerate_plot.py --dataset gazecom          # only GazeCom
    python regenerate_plot.py --model_type tcn cnn_lstm  # only these models
"""

import argparse
import glob
import os
import re
import sys

import torch

sys.path.insert(0, os.getcwd())

from utils.metrics import plot_cmcount, plot_cmpercent, plot_roc, plot_pr


def _derive_prefix_and_fold(results_path):
    fname = os.path.basename(results_path)
    prefix = fname[:-len("_results.pt")] if fname.endswith("_results.pt") else fname
    m = re.match(r"fold(\d+)_", prefix)
    fold_idx = int(m.group(1)) - 1 if m else None
    return prefix, fold_idx


def _find_all_results_pt(base_dir):
    """
    Discover every *_results.pt under base_dir/kfold/ and base_dir/single/,
    inferring (use_kfold, dataset, model_type, fold_idx) from the path.

    Expected structure:
        <base_dir>/kfold/<dataset>/<model_type>/fold_N/<prefix>_results.pt
        <base_dir>/single/<dataset>/<model_type>/<prefix>_results.pt

    Yields dicts: {path, use_kfold, dataset, model_type, prefix, fold_idx}
    """
    jobs = []

    for use_kfold, subdir in [(True, "kfold"), (False, "single")]:
        root = os.path.join(base_dir, subdir)
        if not os.path.isdir(root):
            continue

        pattern = os.path.join(root, "*", "*", "**", "*_results.pt")
        for path in glob.glob(pattern, recursive=True):
            rel = os.path.relpath(path, root)
            parts = rel.split(os.sep)
            # parts = [dataset, model_type, (fold_N,) filename]
            if len(parts) < 3:
                continue
            dataset, model_type = parts[0], parts[1]
            prefix, fold_idx = _derive_prefix_and_fold(path)
            jobs.append({
                "path": path, "use_kfold": use_kfold,
                "dataset": dataset, "model_type": model_type,
                "prefix": prefix, "fold_idx": fold_idx,
            })

    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate confusion matrix / ROC / PR plots for "
                    "every *_results.pt found under results/kfold/ and "
                    "results/single/, without retraining."
    )
    parser.add_argument("--base_dir", type=str, default="results")
    parser.add_argument("--class_names", nargs="+",
                        default=["Fixation", "Saccade", "Pursuit", "Blink"])
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="Only process these dataset(s), e.g. --dataset gazecom")
    parser.add_argument("--model_type", type=str, nargs="+", default=None,
                        help="Only process these model_type(s), e.g. --model_type tcn cnn_lstm")
    parser.add_argument("--skip_roc_pr", action="store_true",
                        help="Only regenerate confusion matrices (faster)")
    parser.add_argument("--dry_run", action="store_true",
                        help="List what would be processed without actually regenerating")
    args = parser.parse_args()

    jobs = _find_all_results_pt(args.base_dir)

    if args.dataset:
        jobs = [j for j in jobs if j["dataset"] in args.dataset]
    if args.model_type:
        jobs = [j for j in jobs if j["model_type"] in args.model_type]

    if not jobs:
        print(f"No *_results.pt files found under {args.base_dir}/kfold/ or "
              f"{args.base_dir}/single/ matching your filters.")
        return

    print(f"Found {len(jobs)} results.pt file(s) to process:\n")
    for j in jobs:
        print(f"  [{j['dataset']:<8}] [{j['model_type']:<15}] "
              f"fold_idx={j['fold_idx']}  {j['path']}")

    if args.dry_run:
        print("\n--dry_run set — nothing was regenerated.")
        return

    print()
    ok, failed = 0, 0
    for j in jobs:
        try:
            results = torch.load(j["path"], map_location="cpu")
            labels, preds, probs = results["labels"], results["preds"], results["probs"]

            plot_cmcount(j["prefix"], labels, preds, args.class_names,
                        use_kfold=j["use_kfold"], fold_idx=j["fold_idx"],
                        base_dir=args.base_dir, model_type=j["model_type"],
                        dataset=j["dataset"])
            plot_cmpercent(j["prefix"], labels, preds, args.class_names,
                          use_kfold=j["use_kfold"], fold_idx=j["fold_idx"],
                          base_dir=args.base_dir, model_type=j["model_type"],
                          dataset=j["dataset"])

            if not args.skip_roc_pr:
                plot_roc(j["prefix"], labels, probs, args.class_names,
                        use_kfold=j["use_kfold"], fold_idx=j["fold_idx"],
                        base_dir=args.base_dir, model_type=j["model_type"],
                        dataset=j["dataset"])
                plot_pr(j["prefix"], labels, probs, args.class_names,
                       use_kfold=j["use_kfold"], fold_idx=j["fold_idx"],
                       base_dir=args.base_dir, model_type=j["model_type"],
                       dataset=j["dataset"])

            print(f"  OK   : {j['path']}")
            ok += 1
        except Exception as e:
            print(f"  FAIL : {j['path']} — {e}")
            failed += 1

    print(f"\nDone — {ok} regenerated, {failed} failed.")


if __name__ == "__main__":
    main()