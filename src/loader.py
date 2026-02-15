# loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import ensure_id_column, parse_labels_cell, raise_if_unknown_labels


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_papers(papers_dir: Path) -> List[Dict]:
    """
    Load list of JSON papers/text-items from a directory.
    Accepts .json files; each file must contain at least an identifier plus content.

    Normalization
    -------------
    - If 'id' is missing, it defaults to the filename stem.
    - 'id' is normalized to a stripped string.
    - If 'text' is missing or None, it is normalized to the empty string "".

    Returns a list of dicts.
    """
    papers: List[Dict] = []
    for p in sorted(papers_dir.glob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        if "id" not in obj:
            obj["id"] = p.stem

        obj["id"] = str(obj["id"]).strip()

        if "text" not in obj or obj["text"] is None:
            obj["text"] = ""

        papers.append(obj)
    return papers


def load_hand_coded_validation(
    csv_path: Path,
    allow_multilabel: bool,
    label_universe: List[str],
) -> pd.DataFrame:
    """
    Load a hand-coded validation CSV with at minimum columns: id, labels.

    The 'labels' cell must be either a plain label string or a JSON array string.

    Returns a DataFrame with labels parsed to a list[str] column.
    """
    df = pd.read_csv(csv_path)

    df = ensure_id_column(df, context="Validation CSV")
    if "labels" not in df.columns:
        raise ValueError(
            f"Validation CSV must contain a 'labels' column. Found: {list(df.columns)}"
        )

    df["id"] = df["id"].astype(str).str.strip()

    df["labels"] = df["labels"].apply(lambda cell: parse_labels_cell(cell, allow_multilabel=allow_multilabel))

    # Universe validation
    for pid, labs in zip(df["id"].tolist(), df["labels"].tolist()):
        raise_if_unknown_labels(
            labs,
            label_universe,
            error_builder=lambda unknown: f"Unknown labels not in universe at id={pid}: {unknown}",
        )

    return df
