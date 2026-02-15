from __future__ import annotations

"""
Hallucination checker for quoted evidence lines.

This module provides a lightweight post-processing check for runs that request
`evidence_lines` in model outputs. For each prediction, it verifies that every
evidence line appears verbatim in the corresponding source text after
Unicode normalisation and whitespace collapsing.

The checker writes a human-readable log of mismatches to `log_path` and returns
the total number of missing lines. It does not alter predictions or metrics.
"""


import re
import unicodedata
from pathlib import Path

from .types import PredictionItem, TextItem


def _normalise(txt: str) -> str:
    """Unicode-normalize (NFKC), lowercase, and collapse whitespace."""
    txt = unicodedata.normalize("NFKC", txt)
    txt = txt.lower()
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def check_hallucinations(
    predictions: list[PredictionItem],
    papers: list[TextItem],
    log_path: Path,
) -> int:
    """
    Scan predicted 'evidence_lines' for each item and verify that each line
    occurs literally (after normalisation) in the corresponding paper's text.
    """
    # Build id -> unified text mapping (created in loader)
    paper_text = {str(p["id"]): p.get("text", "") for p in papers}
    hallucinations = 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        for pred in predictions:
            pid = str(pred.get("id", "")).strip()
            ev_lines = pred.get("evidence_lines", [])
            if not ev_lines:
                continue  # prompt may not request evidence; skip
            source_text = paper_text.get(pid, "")
            if not source_text:
                continue

            original_norm = _normalise(source_text)
            for line in ev_lines:
                if _normalise(line) not in original_norm:
                    hallucinations += 1
                    log.write(f"[id={pid}] hallucinated evidence: {line}\n")

    return hallucinations
