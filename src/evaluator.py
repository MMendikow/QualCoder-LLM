from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd


def merge_gold_pred_by_id(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    empty_error: str = "No overlapping ids between predictions and hand-coded validation",
) -> pd.DataFrame:
    """
    Shared alignment helper used by evaluators (and optionally by callers).
    Preserves existing behavior:
      - subset to ["id","labels"]
      - cast id to str
      - inner merge on id
      - raise if empty
    """
    g = gold_df[["id", "labels"]].copy()
    p = pred_df[["id", "labels"]].copy()
    g["id"] = g["id"].astype(str)
    p["id"] = p["id"].astype(str)

    merged = pd.merge(g, p, on="id", how="inner", suffixes=("_gold", "_pred"))
    if merged.empty:
        raise ValueError(empty_error)
    return merged


# ---------- Multi-label evaluator ----------


def _to_set_list(x) -> Set[str]:
    if isinstance(x, (list, tuple, set)):
        return {str(v) for v in x if str(v).strip()}
    s = str(x).strip()
    return {s} if s else set()


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def compute_metrics_multilabel(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label_universe: List[str],
    *,
    per_label: bool = False,
) -> Dict[str, float]:
    """
    Compute multi-label metrics. Always returns overall metrics; per-label
    metrics are only emitted when per_label=True.
    """
    merged = merge_gold_pred_by_id(gold_df, pred_df)

    subset_correct = 0
    micro_tp = micro_fp = micro_fn = 0
    per_label_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    gold_sets = merged["labels_gold"].apply(_to_set_list).tolist()
    pred_sets = merged["labels_pred"].apply(_to_set_list).tolist()

    for gs, ps in zip(gold_sets, pred_sets):
        if gs == ps:
            subset_correct += 1

        # micro counts
        for _ in ps & gs:
            micro_tp += 1
        for _ in ps - gs:
            micro_fp += 1
        for _ in gs - ps:
            micro_fn += 1

        # per-label counts (needed for macro; emission gated by per_label)
        # EFFICIENT: only touch labels that actually occur in this item
        for y in (gs | ps):
            y = str(y)
            if y in gs and y in ps:
                per_label_counts[y]["tp"] += 1
            elif y in gs:  # and not in ps
                per_label_counts[y]["fn"] += 1
            else:  # y in ps and not in gs
                per_label_counts[y]["fp"] += 1

    n = len(merged)
    subset_acc = subset_correct / n if n else 0.0
    micro_p, micro_r, micro_f1 = _prf(micro_tp, micro_fp, micro_fn)

    # Macro metrics: compute from per-label PRF without necessarily emitting them
    macro_ps, macro_rs, macro_fs = [], [], []
    for y in label_universe:
        c = per_label_counts[str(y)]
        p_, r_, f_ = _prf(c["tp"], c["fp"], c["fn"])
        macro_ps.append(p_)
        macro_rs.append(r_)
        macro_fs.append(f_)

    metrics: Dict[str, float] = {}
    metrics["subset_accuracy"] = subset_acc
    metrics["micro_precision"] = micro_p
    metrics["micro_recall"] = micro_r
    metrics["micro_f1"] = micro_f1
    metrics["macro_precision"] = (sum(macro_ps) / len(label_universe)) if label_universe else 0.0
    metrics["macro_recall"] = (sum(macro_rs) / len(label_universe)) if label_universe else 0.0
    metrics["macro_f1"] = (sum(macro_fs) / len(label_universe)) if label_universe else 0.0

    # Emit per-label details only if requested
    if per_label:
        for y in label_universe:
            y = str(y)
            c = per_label_counts[y]
            p_, r_, f_ = _prf(c["tp"], c["fp"], c["fn"])
            metrics[f"{y}_precision"] = p_
            metrics[f"{y}_recall"] = r_
            metrics[f"{y}_f1"] = f_
            metrics[f"{y}_tp"] = c["tp"]
            metrics[f"{y}_fp"] = c["fp"]
            metrics[f"{y}_fn"] = c["fn"]

    return metrics


# ---------- Single-label evaluator ----------


def _pairwise_counts(gold: List[str], pred: List[str], labels: List[str]) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = Counter()
    for g, p in zip(gold, pred):
        if g in labels and p in labels:
            counts[(g, p)] += 1
        else:
            # unknown classes: ignore in counts; could also raise
            pass
    return counts


def compute_metrics_singlelabel(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    labels: List[str],
    *,
    per_label: bool = False,
) -> Dict[str, float]:
    """
    Compute single-label metrics. Always returns overall metrics; per-label
    metrics are only emitted when per_label=True.
    """
    merged = merge_gold_pred_by_id(gold_df, pred_df)

    # Take first label per item (inputs already normalized to lists; may be empty)
    gold = [(labs[0] if isinstance(labs, list) and labs else "") for labs in merged["labels_gold"].tolist()]
    pred = [(labs[0] if isinstance(labs, list) and labs else "") for labs in merged["labels_pred"].tolist()]

    counts = _pairwise_counts(gold, pred, labels)

    metrics: Dict[str, float] = {}
    supports: Dict[str, int] = {}
    total = len(merged)
    total_tp = 0

    # Compute per-label counts/PRF locally; emit only if per_label=True
    per_label_prf: Dict[str, Tuple[float, float, float]] = {}

    for label in labels:
        tp = counts.get((label, label), 0)
        fp = sum(counts.get((g, label), 0) for g in labels if g != label)
        fn = sum(counts.get((label, p), 0) for p in labels if p != label)
        tn = total - tp - fp - fn

        precision, recall, f1 = _prf(tp, fp, fn)
        per_label_prf[label] = (precision, recall, f1)

        if per_label:
            metrics[f"{label}_precision"] = precision
            metrics[f"{label}_recall"] = recall
            metrics[f"{label}_f1"] = f1
            metrics[f"{label}_tp"] = tp
            metrics[f"{label}_fp"] = fp
            metrics[f"{label}_fn"] = fn
            metrics[f"{label}_tn"] = tn

        supports[label] = tp + fn
        total_tp += tp

    # Overall metrics (always)
    metrics["accuracy"] = (total_tp / total) if total else 0.0
    if labels:
        macro_f1 = sum(per_label_prf[l][2] for l in labels) / len(labels)
    else:
        macro_f1 = 0.0
    metrics["macro_f1"] = macro_f1

    weighted_f1 = sum(per_label_prf[l][2] * supports[l] for l in labels) / total if total else 0.0
    metrics["weighted_f1"] = weighted_f1

    return metrics


# ---------- Dispatcher ----------


def compute_metrics(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    labels: List[str],
    allow_multilabel: bool,
    *,
    per_label: bool = False,
) -> Dict[str, float]:
    """
    Single entry point.
    - Always returns overall metrics.
    - Emits per-label metrics only when per_label=True.
    """
    if allow_multilabel:
        return compute_metrics_multilabel(gold_df, pred_df, labels, per_label=per_label)
    return compute_metrics_singlelabel(gold_df, pred_df, labels, per_label=per_label)
