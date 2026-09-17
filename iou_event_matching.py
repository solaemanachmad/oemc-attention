"""
iou_event_matching.py

IoU-based event-level matching, following the "earliest match" method
of Hooge et al. (2018) as described in Wang et al. (2025, BSPC),
Section 2.3.3 and Fig. 4:

  - Ground-truth and predicted label sequences are each segmented into
    contiguous events (runs of the same class label).
  - For each ground-truth event, the EARLIEST (first-starting)
    predicted event of the SAME class that has not yet been matched
    and that overlaps it in time is selected as its candidate match.
  - IoU = (overlap duration) / (union duration) is computed for that
    matched pair.
  - A match is only counted as a true positive ("hit") if
    IoU > iou_threshold; otherwise it counts as a miss (false
    negative) for the ground-truth event, and the predicted event (if
    any) remains available to be matched to the next ground-truth event.
  - Predicted events never matched to any ground-truth event are false
    positives.
  - This one-to-one greedy matching (a predicted event is "used up"
    once matched) prevents a single predicted event from being counted
    as a hit for two different ground-truth events.

This is a POST-HOC metric computed directly from already-saved
predictions (`*_results.pt`: preds + labels) — no retraining is
required to add this metric to already-completed experiments.

NOTE: this is a best-effort reimplementation of the method described
in the literature (Hooge et al. 2018; used by Startsev et al. and
Wang et al.); no public reference implementation was available to
verify against line-by-line. Treat results as approximate, and state
this in the Method section if reported.
"""

import numpy as np
import torch


def _segment_into_events(labels):
    """
    Run-length-encode a 1D label sequence into a list of
    (start_idx, end_idx_exclusive, label) events.
    """
    labels = np.asarray(labels)
    events = []
    n = len(labels)
    i = 0
    while i < n:
        j = i
        lbl = labels[i]
        while j < n and labels[j] == lbl:
            j += 1
        events.append((i, j, int(lbl)))
        i = j
    return events


def _iou(a_start, a_end, b_start, b_end):
    """IoU between two half-open intervals [start, end)."""
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = max(0, inter_end - inter_start)
    union = (a_end - a_start) + (b_end - b_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compute_iou_event_metrics(preds, labels, class_names, iou_threshold=0.0):
    """
    preds, labels : 1D array-like of predicted / ground-truth class
                     indices (same length, sample-level, in temporal
                     order — e.g. concatenated across a fold's
                     validation set in the same order they were
                     collected).
    class_names   : list of class names, index-aligned with label ids.
    iou_threshold : minimum IoU for a match to count as a hit
                    (paper reports both IoU > 0 and IoU > 0.5).

    Returns a dict with per-class precision/recall/F1, the average
    matched IoU per class, and macro averages — mirroring the
    structure of Table 4/5 in Wang et al. (2025).
    """
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    gt_events = _segment_into_events(labels)
    pred_events = _segment_into_events(preds)

    n_classes = len(class_names)
    results = {}

    for c in range(n_classes):
        gt_c = [e for e in gt_events if e[2] == c]
        pred_c = [e for e in pred_events if e[2] == c]
        pred_available = list(pred_c)  # will be consumed as matched

        tp = 0
        matched_ious = []

        for (gs, ge, _) in gt_c:
            # Earliest-starting, not-yet-matched predicted event of the
            # same class that overlaps this ground-truth event.
            best_idx, best_iou = None, -1.0
            for idx, (ps, pe, _) in enumerate(pred_available):
                if pe <= gs or ps >= ge:
                    continue  # no temporal overlap at all
                # "earliest match": take the first candidate found
                # (pred_events already sorted by start time); compute
                # its IoU and stop searching further candidates.
                best_idx = idx
                best_iou = _iou(gs, ge, ps, pe)
                break

            if best_idx is not None and best_iou > iou_threshold:
                tp += 1
                matched_ious.append(best_iou)
                pred_available.pop(best_idx)
            # else: this ground-truth event is a miss (false negative);
            # any overlapping-but-below-threshold predicted event is
            # left in pred_available and may still match a later
            # ground-truth event or end up counted as a false positive.

        fn = len(gt_c) - tp
        fp = len(pred_available)  # predicted events never matched

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        avg_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

        results[class_names[c]] = {
            "precision": precision, "recall": recall, "f1": f1,
            "avg_iou": avg_iou, "n_gt_events": len(gt_c),
            "n_pred_events": len(pred_c), "tp": tp, "fp": fp, "fn": fn,
        }

    results["macro_f1"] = float(np.mean([results[c]["f1"] for c in class_names]))
    results["macro_iou"] = float(np.mean(
        [results[c]["avg_iou"] for c in class_names if results[c]["n_gt_events"] > 0]
    )) if any(results[c]["n_gt_events"] > 0 for c in class_names) else 0.0

    return results
