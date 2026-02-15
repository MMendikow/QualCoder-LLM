#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv

import argparse
import json
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterator, List, Tuple
JSON = Dict[str, Any]

import pandas as pd
from openai import OpenAIError

from .config import PipelineConfig
from .evaluator import compute_metrics, merge_gold_pred_by_id
from .loader import load_hand_coded_validation, load_papers, load_prompt
from .logger import RunLogger
from .llm_client import LLMClient
from .hallu_check import check_hallucinations
from .krippendorf_helper import compute_icr_alpha
from .utils import ensure_id_column


def _chunks(lst: List[JSON], size: int) -> Iterator[List[JSON]]:
    """Yield successive `size`-sized chunks from `lst`.

    This is a generator; the return type is an iterator of lists.
    """
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _detect_icr_mode(csv_path: Path) -> bool:
    """
    Inspect the hand-coded CSV to decide evaluation mode.

    Returns
    -------
    bool
        True  -> ICR mode (rater-style CSV; ≥1 rater columns besides 'id')
        False -> Gold mode (has a 'labels' column)
    """
    df_head = pd.read_csv(csv_path, nrows=1)

    df_head = ensure_id_column(df_head, context="Hand-coded CSV")

    cols = list(df_head.columns)

    if "labels" in cols:
        return False  # gold evaluation

    # rater-style: all columns except 'id' are raters (need at least one)
    rater_cols = [c for c in cols if c != "id"]
    if len(rater_cols) >= 1:
        return True

    raise ValueError(
        "Could not infer evaluation mode from CSV. Either provide a 'labels' column "
        "(gold evaluation) or ≥1 rater columns besides 'id' (ICR mode)."
    )


def _classify_batch_exception(e: Exception) -> Tuple[str, bool]:
    """
    Classify exceptions from the batch classification loop into:
      - (category, retryable)

    Goal: preserve external behavior for successful runs while avoiding wasted
    retries on clearly non-retryable failures (config/programming errors).
    """
    # ---- configuration / environment (non-retryable) ----
    if isinstance(e, EnvironmentError) and "OPENAI_API_KEY" in str(e):
        return "config_missing_api_key", False

    # ---- programming errors (non-retryable; fail fast) ----
    if isinstance(e, (AttributeError, TypeError, NameError, ImportError)):
        return "programming_error", False

    # ---- API / transient IO (retryable) ----
    if isinstance(e, OpenAIError):
        return "openai_api_error", True
    if isinstance(e, (TimeoutError, ConnectionError)):
        return "transient_io_error", True

    # Some transient network issues may surface as OSError; be conservative and only
    # mark as retryable when the message suggests a transient condition.
    if isinstance(e, OSError):
        msg = str(e).lower()
        transient_markers = (
            "timed out",
            "timeout",
            "temporar",  # temporary / temporarily
            "connection reset",
            "connection aborted",
            "network is unreachable",
            "name or service not known",
            "try again",
        )
        if any(m in msg for m in transient_markers):
            return "transient_os_error", True
        return "os_error", False

    # ---- model output parse/validation issues (often retryable) ----
    # In this project, many deterministic-seeming errors are actually due to
    # non-compliant model outputs and can succeed on retry.
    if isinstance(e, (json.JSONDecodeError, ValueError)):
        return "model_output_or_validation_error", True

    # ---- default: unknown exception type; fail fast to avoid masking bugs ----
    return "unknown_error", False


def _prepare_run_paths(cfg: PipelineConfig, ts: str) -> Tuple[Path, Path]:
    # Run directory and canonical cache
    if cfg.canonical_raw_path is None:
        # Normal run: create a new run directory and canonical raw_output.jsonl
        run_dir = cfg.results_dir / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        canonical_raw_path = run_dir / "raw_output.jsonl"
        print(f"[main] Writing canonical cache to: {canonical_raw_path}")
    else:
        # Evaluation-only run: reuse the canonical file specified in the YAML
        canonical_raw_path = cfg.canonical_raw_path
        if not canonical_raw_path.exists():
            raise FileNotFoundError(f"canonical_raw_path does not exist: {canonical_raw_path}")
        run_dir = canonical_raw_path.parent
        print(f"[main] DRY RUN using canonical: {canonical_raw_path}")

    return run_dir, canonical_raw_path


def _load_inputs(cfg: PipelineConfig, *, is_icr: bool) -> Tuple[List[JSON], str, pd.DataFrame | None]:
    # Inputs
    papers = load_papers(cfg.text_dir)
    prompt_template = load_prompt(cfg.prompt_path)

    # Gold labels are only needed in gold mode
    gold_df = None
    if not is_icr:
        gold_df = load_hand_coded_validation(
            cfg.hand_coded_validation_path,
            cfg.allow_multilabel,
            cfg.labels,
        )
    return papers, prompt_template, gold_df


def _read_predictions_from_canonical(canonical_raw_path: Path) -> List[JSON]:
    predictions: List[JSON] = []
    with open(canonical_raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def _classify_and_persist_predictions(
    *,
    cfg: PipelineConfig,
    run_dir: Path,
    canonical_raw_path: Path,
    logger: RunLogger,
    client: LLMClient,
    papers: List[JSON],
    prompt_template: str,
    ts: str,
) -> List[JSON]:
    # NORMAL RUN: classify in batches, with retries, and write canonical incrementally
    predictions: List[JSON] = []

    saved_count = 0

    last_full_prompt: str | None = None

    # Start (or truncate) canonical file early so partial results are still valid JSONL.
    canonical_raw_path.parent.mkdir(parents=True, exist_ok=True)

    with open(canonical_raw_path, "w", encoding="utf-8") as f_out:
        pass

    raw_input_path = run_dir / "raw_input.txt"
    with open(raw_input_path, "w", encoding="utf-8") as f_in:
        f_in.write(f"RAW INPUT PROMPTS — run={ts}\n")
        f_in.write("=" * 80 + "\n\n")

    for batch in _chunks(papers, cfg.batch_size):

        batch_ids = [str(it.get("id", "")).strip() for it in batch]

        attempt = 0

        batch_preds = None

        while attempt <= cfg.max_retries:

            try:

                batch_preds, last_full_prompt = client.classify_batch(
                    prompt_template, batch,
                    allow_multilabel=cfg.allow_multilabel,
                    label_universe=cfg.labels,
                    include_evidence_lines=cfg.include_evidence_lines,
                    include_explanation=cfg.include_explanation,
                )

                with open(raw_input_path, "a", encoding="utf-8") as f_in:
                    f_in.write(
                        f"===== BATCH PROMPT (SUCCESS) | attempt={attempt + 1}/{cfg.max_retries + 1} | ids={batch_ids} =====\n")
                    f_in.write(last_full_prompt)
                    f_in.write("\n===== END BATCH PROMPT =====\n\n")

                break  # success

            except Exception as e:

                category, retryable = _classify_batch_exception(e)
                attempt += 1

                # Log + console warning (preserve existing surface behavior)
                logger.info(
                    f"Batch failed (attempt {attempt}/{cfg.max_retries + 1}) "
                    f"for ids={batch_ids}. Error: {e} [category={category}, retryable={retryable}]"
                )

                maybe_prompt = getattr(client, "_last_full_prompt", None)
                if maybe_prompt:
                    with open(raw_input_path, "a", encoding="utf-8") as f_in:
                        f_in.write(
                            f"===== BATCH PROMPT (FAIL) | attempt={attempt}/{cfg.max_retries + 1} | ids={batch_ids} =====\n")
                        f_in.write(maybe_prompt)
                        f_in.write("\n----- ERROR -----\n")
                        f_in.write(str(e) + "\n")
                        f_in.write("===== END BATCH PROMPT =====\n\n")
                else:
                    # still write a failure block even if no prompt was captured
                    with open(raw_input_path, "a", encoding="utf-8") as f_in:
                        f_in.write(
                            f"===== BATCH PROMPT (FAIL) | attempt={attempt}/{cfg.max_retries + 1} | ids={batch_ids} =====\n")
                        f_in.write("<NO PROMPT CAPTURED>\n")
                        f_in.write("\n----- ERROR -----\n")
                        f_in.write(str(e) + "\n")
                        f_in.write("===== END BATCH PROMPT =====\n\n")

                print(
                    f"[WARN] Batch failed (attempt {attempt}/{cfg.max_retries + 1}) "
                    f"for ids={batch_ids}. Error: {e}"
                )

                # Stop immediately on non-retryable errors, or once retries are exhausted.
                if (not retryable) or (attempt > cfg.max_retries):

                    if not retryable:
                        logger.info(
                            f"Non-retryable error encountered for ids={batch_ids}; aborting without further retries. "
                            f"[category={category}]"
                        )

                    if attempt > cfg.max_retries:
                        # Final failure: warn about partial saved outputs
                        print(
                            f"[WARN] Giving up after {cfg.max_retries + 1} attempts. "
                            f"Saved partial predictions for {saved_count} items; "
                            f"next failing batch starts at id={batch_ids[0] if batch_ids else '<unknown>'}."
                        )
                        logger.info(
                            f"Giving up after {cfg.max_retries + 1} attempts. "
                            f"Saved partial predictions for {saved_count} items; "
                            f"next failing batch starts at id={batch_ids[0] if batch_ids else '<unknown>'}."
                        )

                    # Print the last prompt used (for debugging)
                    if last_full_prompt:
                        print("\n===== FULL PROMPT SENT TO MODEL (FINAL FAILURE) =====")
                        print(last_full_prompt)
                        print("===== END PROMPT =====\n")

                    raise  # re-raise the last exception

        # Successful batch: append to in-memory list and write to canonical file immediately
        if not batch_preds:
            # Defensive: should not happen if success path broke out
            raise RuntimeError(f"Internal error: batch_preds is empty for ids={batch_ids}")

        predictions.extend(batch_preds)

        # Append each object as JSONL so partial progress is preserved even if later batches fail
        with open(canonical_raw_path, "a", encoding="utf-8") as f_out:
            for obj in batch_preds:
                json.dump(obj, f_out, ensure_ascii=False)
                f_out.write("\n")

        saved_count += len(batch_preds)

    print(f"[main] Wrote canonical raw file: {canonical_raw_path.name}")
    return predictions


def _maybe_run_hallucination_check(
    *,
    cfg: PipelineConfig,
    is_icr: bool,
    predictions: List[JSON],
    papers: List[JSON],
    run_dir: Path,
    logger: RunLogger,
) -> None:
    if (not is_icr) and cfg.hallucination_check:
        hallu_log = run_dir / "hallucinations.log"
        hallu_count = check_hallucinations(predictions, papers, hallu_log)
        logger.info(f"Hallucinations (count): {hallu_count}  [details: {hallu_log.name}]")


def _compute_metrics_and_artifacts(
    *,
    cfg: PipelineConfig,
    is_icr: bool,
    predictions: List[JSON],
    gold_df: pd.DataFrame | None,
    run_dir: Path,
) -> Dict[str, float]:
    # ------------------ Compute metrics ------------------
    if is_icr:
        metrics = compute_icr_alpha(
            cfg.hand_coded_validation_path,
            predictions,
            cfg.labels,
        )
        return metrics

    pred_df = pd.DataFrame(predictions)
    metrics = compute_metrics(
        gold_df,
        pred_df,
        cfg.labels,
        cfg.allow_multilabel,
        per_label=not cfg.allow_multilabel,
    )

    # ---------- Crosstab / confusion matrix for F1 (single-label) runs ----------
    if not cfg.allow_multilabel:
        if gold_df is None:
            raise ValueError("Gold labels are required to compute crosstab for F1 runs.")

        merged = merge_gold_pred_by_id(
            gold_df,
            pred_df,
            empty_error=(
                "No overlapping ids between predictions and hand-coded validation "
                "when constructing crosstab."
            ),
        )

        # Flatten to single labels (same convention as compute_metrics_singlelabel)
        gold_labels = [
            labs[0] if isinstance(labs, list) and labs else ""
            for labs in merged["labels_gold"].tolist()
        ]
        pred_labels = [
            labs[0] if isinstance(labs, list) and labs else ""
            for labs in merged["labels_pred"].tolist()
        ]

        # Use configured label universe to fix row/column order
        gold_cat = pd.Categorical(gold_labels, categories=cfg.labels)
        pred_cat = pd.Categorical(pred_labels, categories=cfg.labels)

        crosstab_df = pd.crosstab(
            gold_cat,
            pred_cat,
            rownames=["gold"],
            colnames=["pred"],
            dropna=False,
        )

        crosstab_path = run_dir / "crosstab_log.csv"
        crosstab_df.to_csv(crosstab_path)

    return metrics


def _append_run_history_row(
    *,
    cfg: PipelineConfig,
    ts: str,
    is_icr: bool,
    metrics: Dict[str, float],
) -> None:
    # ------------------ run-history row (NO per-label fields) ------------------
    history_path = cfg.results_dir / "run_history.csv"
    row = OrderedDict()
    cfg_dict = asdict(cfg)

    if is_icr:
        # ---- Krippendorff runs ----

        def _alpha_history_name(metric_key: str) -> str:
            if metric_key.startswith("krippendorff_alpha_binary_"):
                return f"{metric_key} (binary-incidence)"
            if metric_key.startswith("krippendorff_alpha_jaccard_"):
                return f"{metric_key} (jaccard)"
            # atomic nominal keys
            if metric_key.startswith("krippendorff_alpha"):
                return f"{metric_key} (atomic)"
            return metric_key

        # (1) Put the three overall scores first (one per alpha "version")
        overall_atomic = "krippendorff_alpha_overall"
        overall_binary_macro = "krippendorff_alpha_binary_overall_macro"
        overall_jaccard = "krippendorff_alpha_jaccard_overall"

        row[_alpha_history_name(overall_atomic)] = metrics.get(overall_atomic)
        row[_alpha_history_name(overall_binary_macro)] = metrics.get(overall_binary_macro)
        row[_alpha_history_name(overall_jaccard)] = metrics.get(overall_jaccard)

        # Leading config (keep same order as before, but insert new columns after role)
        row["batch_size"] = cfg_dict.get("batch_size")
        row["llm"] = cfg_dict.get("llm")
        row["model_role"] = cfg_dict.get("model_role")
        row["include_explanation"] = cfg_dict.get("include_explanation")
        row["include_evidence_lines"] = cfg_dict.get("include_evidence_lines")

        # Remaining config fields, in original order, excluding specified keys
        exclude_cfg_keys_icr = {
            "labels",
            "allow_multilabel",
            "canonical_raw_path",
            "batch_size",
            "llm",
            "model_role",
            "include_explanation",
            "include_evidence_lines",
        }
        for k, v in cfg_dict.items():
            if k in exclude_cfg_keys_icr:
                continue
            if k in row:
                continue
            row[k] = v

        # (2) Add all ICR metrics that are printed in the log, in log order,
        # except the three "overall" metrics already placed first.
        atomic_keys = sorted(
            k for k in metrics.keys()
            if k.startswith("krippendorff_alpha")
            and "binary" not in k
            and "jaccard" not in k
            and k != overall_atomic
        )
        binary_keys = sorted(k for k in metrics.keys() if k.startswith("krippendorff_alpha_binary_"))
        jaccard_keys = sorted(k for k in metrics.keys() if k.startswith("krippendorff_alpha_jaccard_"))

        # Atomic section order: humans, pairwise humans, human vs llm
        if "krippendorff_alpha_humans" in metrics:
            row[_alpha_history_name("krippendorff_alpha_humans")] = metrics.get("krippendorff_alpha_humans")

        for k in atomic_keys:
            if "_vs_" in k and not k.endswith("_vs_llm"):
                row[_alpha_history_name(k)] = metrics.get(k)
        for k in atomic_keys:
            if k.endswith("_vs_llm"):
                row[_alpha_history_name(k)] = metrics.get(k)

        # Binary-incidence section order:
        # overall (macro) already placed first; now add overall (prevalence-weighted), humans-only,
        # then pairwise humans, then human vs llm (macro and prevalence-weighted already encoded in keys)
        if "krippendorff_alpha_binary_overall_prevalence_weighted" in metrics:
            row[_alpha_history_name("krippendorff_alpha_binary_overall_prevalence_weighted")] = metrics.get(
                "krippendorff_alpha_binary_overall_prevalence_weighted"
            )
        if "krippendorff_alpha_binary_humans_macro" in metrics:
            row[_alpha_history_name("krippendorff_alpha_binary_humans_macro")] = metrics.get(
                "krippendorff_alpha_binary_humans_macro"
            )
        if "krippendorff_alpha_binary_humans_prevalence_weighted" in metrics:
            row[_alpha_history_name("krippendorff_alpha_binary_humans_prevalence_weighted")] = metrics.get(
                "krippendorff_alpha_binary_humans_prevalence_weighted"
            )

        # Pairwise humans (both macro and prevalence-weighted)
        for k in binary_keys:
            if "_vs_" in k and not k.endswith("_vs_llm_macro") and not k.endswith("_vs_llm_prevalence_weighted"):
                # this captures human-vs-human keys
                if "_vs_llm_" in k:
                    continue
                if "_vs_" in k:
                    row[_alpha_history_name(k)] = metrics.get(k)

        # Human vs LLM (both macro and prevalence-weighted)
        for k in binary_keys:
            if "_vs_llm_" in k:
                row[_alpha_history_name(k)] = metrics.get(k)

        # Jaccard section order: humans, pairwise humans, human vs llm
        if "krippendorff_alpha_jaccard_humans" in metrics:
            row[_alpha_history_name("krippendorff_alpha_jaccard_humans")] = metrics.get(
                "krippendorff_alpha_jaccard_humans"
            )

        for k in jaccard_keys:
            if k == overall_jaccard:
                continue
            if "_vs_" in k and not k.endswith("_vs_llm"):
                row[_alpha_history_name(k)] = metrics.get(k)
        for k in jaccard_keys:
            if k.endswith("_vs_llm"):
                row[_alpha_history_name(k)] = metrics.get(k)

    else:
        # ---- F1 score history runs (classification) ----
        # Leading metrics
        row["accuracy"] = metrics.get("accuracy")
        row["macro_f1"] = metrics.get("macro_f1")
        row["weighted_f1"] = metrics.get("weighted_f1")

        # Leading config (insert new columns after role)
        row["batch_size"] = cfg_dict.get("batch_size")
        row["llm"] = cfg_dict.get("llm")
        row["model_role"] = cfg_dict.get("model_role")
        row["include_explanation"] = cfg_dict.get("include_explanation")
        row["include_evidence_lines"] = cfg_dict.get("include_evidence_lines")

        # Remaining config fields, excluding specified keys
        exclude_cfg_keys_f1 = {
            "labels",
            "allow_multilabel",
            "batch_size",
            "llm",
            "model_role",
            "include_explanation",
            "include_evidence_lines",
        }
        for k, v in cfg_dict.items():
            if k in exclude_cfg_keys_f1:
                continue
            if k in row:
                continue
            row[k] = v

        # Additional metrics for multi-label runs that are not yet included
        if cfg.allow_multilabel:
            extra_metric_keys = ["subset_accuracy", "micro_f1"]
        else:
            extra_metric_keys = []
        for key in extra_metric_keys:
            if key not in row:
                row[key] = metrics.get(key)

    row["timestamp"] = ts

    pd.DataFrame([row]).to_csv(
        history_path, mode="a", header=not history_path.exists(), index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    # Only a positional config path; all other behaviour is controlled via YAML.
    parser.add_argument("config", help="Path to config.yaml")

    args = parser.parse_args()
    load_dotenv()  # load OPENAI_API_KEY from a local .env

    cfg = PipelineConfig.from_yaml(args.config)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Decide evaluation mode from the hand-coded CSV shape (auto-ICR detection)
    is_icr = _detect_icr_mode(cfg.hand_coded_validation_path)

    run_dir, canonical_raw_path = _prepare_run_paths(cfg, ts)

    logger = RunLogger(cfg.logs_dir, ts)
    logger.write_config(asdict(cfg))

    papers, prompt_template, gold_df = _load_inputs(cfg, is_icr=is_icr)

    client = LLMClient(cfg.llm, cfg.temperature, cfg.model_role)

    # ------------------ Obtain predictions ------------------
    if cfg.canonical_raw_path is not None:
        # Evaluation-only: read predictions from the canonical file specified in YAML
        predictions = _read_predictions_from_canonical(canonical_raw_path)
    else:
        predictions = _classify_and_persist_predictions(
            cfg=cfg,
            run_dir=run_dir,
            canonical_raw_path=canonical_raw_path,
            logger=logger,
            client=client,
            papers=papers,
            prompt_template=prompt_template,
            ts=ts,
        )

    _maybe_run_hallucination_check(
        cfg=cfg,
        is_icr=is_icr,
        predictions=predictions,
        papers=papers,
        run_dir=run_dir,
        logger=logger,
    )

    metrics = _compute_metrics_and_artifacts(
        cfg=cfg,
        is_icr=is_icr,
        predictions=predictions,
        gold_df=gold_df,
        run_dir=run_dir,
    )

    logger.write_metrics(metrics, per_label=(not is_icr and not cfg.allow_multilabel))

    _append_run_history_row(cfg=cfg, ts=ts, is_icr=is_icr, metrics=metrics)

    print(f"[main] Run completed – log written to {logger.path}")


if __name__ == "__main__":
    main()
