from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from itertools import combinations

import numpy as np
import pandas as pd

from .utils import parse_labels_cell, canonicalize_label_set, ensure_id_column, raise_if_unknown_labels


def _slug(name: str) -> str:
    """Filesystem/metric-key friendly slug."""
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _alpha_on_subset_generic(
    df_all: pd.DataFrame,
    cols: List[str],
    *,
    alpha_fn: Callable[[pd.DataFrame], float],
) -> float:
    """
    Shared subset-of-coders helper:
      - select coder columns present in df_all
      - require at least 2 coder columns
      - keep rows with ≥2 non-missing coders
      - return alpha_fn(subset) or NaN if insufficient data
    """
    sub = df_all.loc[:, [c for c in cols if c in df_all.columns]]
    if sub.shape[1] < 2:
        return float("nan")
    sub = sub[sub.notna().sum(axis=1) >= 2]
    if sub.empty:
        return float("nan")
    return alpha_fn(sub)


def _alpha_on_subset(df_all: pd.DataFrame, cols: List[str]) -> float:
    """
    Compute Krippendorff's alpha on a subset of coder columns in df_all.
    Keeps rows with at least two non-missing coders in the subset.
    """
    return _alpha_on_subset_generic(df_all, cols, alpha_fn=kripp_alpha_nominal)


# ---------- Load rater-style CSV (parsing centralized via utils) ----------

def load_rater_style_validation(
    csv_path: Path,
    label_universe: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns (df_wide, raters), where df_wide has index=id, columns=raters,
    values=canonical tokens (strings) and NaN for missing.
    """
    df = pd.read_csv(csv_path)

    # Canonicalize id column name
    df = ensure_id_column(df, context="Rater CSV")

    # Identify rater columns: all except 'id'
    raters = [c for c in df.columns if c != "id"]
    if not raters:
        raise ValueError("Rater CSV must contain at least one rater column besides 'id'")

    # Parse cells -> canonical tokens using shared helpers
    rows = []
    for _, row in df.iterrows():
        pid = str(row["id"]).strip()
        out = {"id": pid}
        for r in raters:
            cell = row[r]
            if pd.isna(cell):
                out[r] = np.nan
            else:
                # Multi-label is allowed in ICR.
                labs = parse_labels_cell(cell, allow_multilabel=True)

                # Strict universe validation (kept here by design)
                raise_if_unknown_labels(
                    labs,
                    label_universe,
                    error_builder=lambda unknown: f"Unknown labels not in universe at id={pid}, rater={r}: {unknown}",
                )

                out[r] = canonicalize_label_set(labs)
        rows.append(out)

    df_wide = pd.DataFrame(rows).set_index("id")
    return df_wide, raters


# ---------- LLM predictions -> coder column (canonicalization centralized) ----------

def build_llm_coder(predictions: List[Dict], label_universe: List[str]) -> pd.Series:
    """
    Convert model predictions (list of dicts with 'id' and 'labels') into
    a Series indexed by id with canonical tokens.
    """
    data = {}
    for obj in predictions:
        pid = str(obj.get("id", "")).strip()
        labs = obj.get("labels", []) or []
        labs = [str(x).strip() for x in labs if str(x).strip()]

        # Strict universe validation
        raise_if_unknown_labels(
            labs,
            label_universe,
            error_builder=lambda unknown: f"Prediction for id={pid} includes unknown labels: {unknown}",
        )

        token = canonicalize_label_set(labs)
        data[pid] = token

    s = pd.Series(data, name="LLM")
    s.index.name = "id"
    return s


# ---------- Krippendorff’s alpha (nominal; atomic multi-label) ----------

def kripp_alpha_nominal(codes: pd.DataFrame) -> float:
    """
    Compute Krippendorff's alpha for nominal data.
    'codes' is an items x coders DataFrame of category tokens (strings, ints...) with NaN as missing.
    Implementation follows the standard Do/De formulation and handles missing values.
    """
    if codes.empty:
        return 0.0

    # Observed disagreement Do
    num = 0.0
    den = 0.0
    for _, row in codes.iterrows():
        # cast to str for categorical counting; drop missing
        vals = [str(v) for v in row.tolist() if not pd.isna(v)]
        n_i = len(vals)
        if n_i < 2:
            continue
        # counts per category for this item
        counts: Dict[str, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        # sum_c n_ic (n_i - n_ic)
        num += sum(c * (n_i - c) for c in counts.values())
        den += n_i * (n_i - 1)

    Do = (num / den) if den > 0 else 0.0

    # Expected disagreement De (category marginals across all items/coders)
    flat = codes.values.ravel()
    marg: Dict[str, int] = {}
    N = 0
    for v in flat:
        if pd.isna(v):
            continue
        s = str(v)
        marg[s] = marg.get(s, 0) + 1
        N += 1

    if N < 2:
        return 0.0

    De = sum(n_c * (N - n_c) for n_c in marg.values()) / (N * (N - 1))

    if De == 0.0:
        # no variation; alpha undefined; return 1 if perfect agreement else 0
        return 1.0 if Do == 0.0 else 0.0

    alpha = 1.0 - (Do / De)
    # numerical guard
    if math.isnan(alpha):
        return 0.0
    return float(alpha)


# ---------- Set-based α (general metric form) ----------

def _token_to_set(token: object, empty_token: str = "<EMPTY>"):
    """
    Convert a canonical token back to a Python set of labels.
    Returns None for missing (NaN).
    """
    if pd.isna(token):
        return None
    s = str(token)
    if s == empty_token:
        return frozenset()
    return frozenset(s.split("|"))


def jaccard_distance(S: frozenset, T: frozenset) -> float:
    """
    Jaccard distance between two sets S and T.
    """
    if S == T:
        return 0.0
    union = len(S | T)
    if union == 0:
        return 0.0  # both empty
    inter = len(S & T)
    return 1.0 - (inter / union)


def kripp_alpha_metric(codes: pd.DataFrame, delta: Callable[[object, object], float]) -> float:
    """
    General Krippendorff's alpha for arbitrary distance 'delta' between cell values.
    'codes' is items x coders with values already in the domain of 'delta' (no NaNs).
    Observed disagreement Do: mean delta over unordered coder pairs within items.
    Expected disagreement De: mean delta over unordered pairs drawn from the pooled marginal.
    """
    if codes.empty:
        return 0.0

    # Observed disagreement: accumulate over items
    num_Do = 0.0
    den_Do = 0.0
    for _, row in codes.iterrows():
        vals = [v for v in row.tolist() if v is not None]
        m = len(vals)
        if m < 2:
            continue
        # unordered pairs within the item
        for i in range(m - 1):
            for j in range(i + 1, m):
                num_Do += delta(vals[i], vals[j])
        den_Do += (m * (m - 1) / 2.0)
    Do = (num_Do / den_Do) if den_Do > 0 else 0.0

    # Expected disagreement: from the pooled marginal
    flat_vals: List[object] = []
    for _, row in codes.iterrows():
        for v in row.tolist():
            if v is not None:
                flat_vals.append(v)
    N = len(flat_vals)
    if N < 2:
        return 0.0

    # Compress identical values with frequencies for efficiency
    freq: Dict[object, int] = {}
    for v in flat_vals:
        freq[v] = freq.get(v, 0) + 1

    num_De = 0.0
    keys = list(freq.keys())
    for i in range(len(keys)):
        vi = keys[i]
        ni = freq[vi]
        for j in range(i + 1, len(keys)):
            vj = keys[j]
            nj = freq[vj]
            num_De += (ni * nj) * delta(vi, vj)

    den_De = N * (N - 1) / 2.0
    De = num_De / den_De if den_De > 0 else 0.0

    if De == 0.0:
        return 1.0 if Do == 0.0 else 0.0
    alpha = 1.0 - (Do / De)
    if math.isnan(alpha):
        return 0.0
    return float(alpha)


def kripp_alpha_jaccard(codes_tokens: pd.DataFrame) -> float:
    """
    Krippendorff's alpha with Jaccard distance over sets of labels.
    'codes_tokens' is items x coders with canonical tokens or NaN.
    """
    # Convert tokens -> sets, preserving missing as None (pandas deprecation-safe)
    def to_set_or_none(x):
        return _token_to_set(x)

    codes_sets = codes_tokens.apply(lambda col: col.map(to_set_or_none))

    # Drop items that have fewer than two non-missing coders
    codes_sets = codes_sets[codes_sets.notna().sum(axis=1) >= 2]
    if codes_sets.empty:
        return float("nan")

    return kripp_alpha_metric(codes_sets, jaccard_distance)


# ---------- Binary-incidence α per label (nominal), with summaries ----------

def _binary_table_for_label(codes_tokens: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    For a given label, return an items x coders DataFrame with values in {0,1,NaN}
    indicating absence/presence for that label in each coder's set.
    """
    def present(token) -> object:
        if pd.isna(token):
            return np.nan
        S = _token_to_set(token)
        return 1 if (S is not None and label in S) else 0

    # pandas 2.2+: avoid applymap deprecation
    return codes_tokens.apply(lambda col: col.map(present))


def _binary_incidence_alpha_on_subset(
    df_all: pd.DataFrame,
    cols: List[str],
    label_universe: List[str],
) -> Tuple[float, float]:
    """
    Compute (macro_alpha, prevalence_weighted_alpha) on a subset of coder columns.
    """
    sub = df_all.loc[:, [c for c in cols if c in df_all.columns]]
    if sub.shape[1] < 2:
        return float("nan"), float("nan")

    # Keep items with at least two non-missing coders
    sub = sub[sub.notna().sum(axis=1) >= 2]
    if sub.empty:
        return float("nan"), float("nan")

    alphas: List[float] = []
    weights: List[float] = []  # positives per label (for prevalence weighting)

    for lab in label_universe:
        bin_df = _binary_table_for_label(sub, lab)
        a_lab = kripp_alpha_nominal(bin_df)
        alphas.append(a_lab)

        positives = float(np.nansum(bin_df.to_numpy(dtype=float)))
        weights.append(positives)

    # Macro
    macro = float(np.nanmean(alphas)) if len(alphas) else float("nan")

    # Prevalence-weighted
    total_w = sum(w for w in weights if not math.isnan(w))
    if total_w > 0:
        pw = 0.0
        for a, w in zip(alphas, weights):
            if not math.isnan(a) and w > 0:
                pw += (w / total_w) * a
        pw = float(pw)
    else:
        pw = macro  # fallback

    return macro, pw


# ---------- Public orchestrator for main.py ----------

def compute_icr_alpha(
    rater_csv_path: Path,
    predictions: List[Dict],
    label_universe: List[str],
) -> Dict[str, float]:
    """
    Load the rater-style CSV, add the model as 'LLM' coder, and compute:
      - overall alpha (humans + LLM),
      - humans-only alpha (if >=2 humans),
      - all pairwise human–human alphas,
      - human–LLM alpha per human rater.

    Returns a dict of {metric_name: float}, including:
      - Atomic nominal α (existing keys, unchanged)
      - Binary-incidence α summaries (macro, prevalence-weighted)
      - Jaccard set-based α
    """
    df_raters, raters = load_rater_style_validation(rater_csv_path, label_universe)
    s_llm = build_llm_coder(predictions, label_universe)

    # Join LLM coder; keep only items present in both raters and LLM
    df_all = df_raters.join(s_llm, how="inner")

    # Identify coders
    human_raters = raters  # all columns were humans
    all_coders = human_raters + ["LLM"]

    metrics: Dict[str, float] = {}

    # ===== (1) Atomic multi-label tokens (exact match only) =====
    a_overall = _alpha_on_subset(df_all, all_coders)
    metrics["krippendorff_alpha_overall"] = a_overall

    if len(human_raters) >= 2:
        a_humans = _alpha_on_subset(df_all, human_raters)
        metrics["krippendorff_alpha_humans"] = a_humans

    # Pairwise human–human
    for h1, h2 in combinations(human_raters, 2):
        key = f"krippendorff_alpha_{_slug(h1)}_vs_{_slug(h2)}"
        metrics[key] = _alpha_on_subset(df_all, [h1, h2])

    # Pairwise human–LLM
    for h in human_raters:
        key = f"krippendorff_alpha_{_slug(h)}_vs_llm"
        metrics[key] = _alpha_on_subset(df_all, [h, "LLM"])

    # ===== (2) Binary-incidence α per label (+ macro / prevalence-weighted summaries) =====
    macro_overall, pw_overall = _binary_incidence_alpha_on_subset(df_all, all_coders, label_universe)
    metrics["krippendorff_alpha_binary_overall_macro"] = macro_overall
    metrics["krippendorff_alpha_binary_overall_prevalence_weighted"] = pw_overall

    if len(human_raters) >= 2:
        macro_h, pw_h = _binary_incidence_alpha_on_subset(df_all, human_raters, label_universe)
        metrics["krippendorff_alpha_binary_humans_macro"] = macro_h
        metrics["krippendorff_alpha_binary_humans_prevalence_weighted"] = pw_h

        # Pairwise human–human
        for h1, h2 in combinations(human_raters, 2):
            macro_pair, pw_pair = _binary_incidence_alpha_on_subset(df_all, [h1, h2], label_universe)
            metrics[f"krippendorff_alpha_binary_{_slug(h1)}_vs_{_slug(h2)}_macro"] = macro_pair
            metrics[f"krippendorff_alpha_binary_{_slug(h1)}_vs_{_slug(h2)}_prevalence_weighted"] = pw_pair

    # Pairwise human–LLM
    for h in human_raters:
        macro_hl, pw_hl = _binary_incidence_alpha_on_subset(df_all, [h, "LLM"], label_universe)
        metrics[f"krippendorff_alpha_binary_{_slug(h)}_vs_llm_macro"] = macro_hl
        metrics[f"krippendorff_alpha_binary_{_slug(h)}_vs_llm_prevalence_weighted"] = pw_hl

    # ===== (3) Set-based α with Jaccard distance =====
    a_j_overall = _alpha_on_subset_generic(df_all, all_coders, alpha_fn=kripp_alpha_jaccard)
    metrics["krippendorff_alpha_jaccard_overall"] = a_j_overall

    if len(human_raters) >= 2:
        a_j_humans = _alpha_on_subset_generic(df_all, human_raters, alpha_fn=kripp_alpha_jaccard)
        metrics["krippendorff_alpha_jaccard_humans"] = a_j_humans

    # Pairwise human–human
    for h1, h2 in combinations(human_raters, 2):
        key = f"krippendorff_alpha_jaccard_{_slug(h1)}_vs_{_slug(h2)}"
        metrics[key] = _alpha_on_subset_generic(df_all, [h1, h2], alpha_fn=kripp_alpha_jaccard)

    # Pairwise human–LLM
    for h in human_raters:
        key = f"krippendorff_alpha_jaccard_{_slug(h)}_vs_llm"
        metrics[key] = _alpha_on_subset_generic(df_all, [h, "LLM"], alpha_fn=kripp_alpha_jaccard)

    return metrics
