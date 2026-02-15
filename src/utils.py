# utils.py
from __future__ import annotations
import json
from typing import Callable, List
import pandas as pd

def parse_labels_cell(cell: object, allow_multilabel: bool) -> List[str]:
    """
    Parse a CSV cell containing labels.

    Accepted input forms
    --------------------
    Multi-label tasks (allow_multilabel=True):
      - JSON array string, e.g. ["method","dataset"]
      - Plain string (single-label special case), e.g. method

    Single-label tasks (allow_multilabel=False):
      - Plain string, e.g. method
      - JSON array string of length 1, e.g. ["method"]

    Missing/empty cells become [].
    """
    if pd.isna(cell):
        return []

    if isinstance(cell, list):
        labs = [str(x).strip() for x in cell if str(x).strip()]
    else:
        s = str(cell).strip()
        if not s:
            return []

        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                labs = [str(x).strip() for x in parsed if str(x).strip()]
            else:
                labs = [s]
        else:
            labs = [s]

    labs = [x for x in labs if x]

    if not allow_multilabel and len(labs) > 1:
        raise ValueError(f"Single-label task but multiple labels found: {labs}")

    return labs

def canonicalize_label_set(labels: List[str], empty_token: str = "<EMPTY>") -> str:
    """
    Turn a list of labels into a canonical atomic token for Krippendorff's alpha:
    - sort unique labels
    - join with '|'
    - represent empty as <EMPTY>
    """
    labs = [str(x).strip() for x in labels if str(x).strip()]
    uniq = sorted(set(labs))
    if not uniq:
        return empty_token
    return "|".join(uniq)

def trim_dedup_str_list(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        xs = str(x).strip()
        if not xs:
            continue
        if xs not in seen:
            seen.add(xs)
            out.append(xs)
    return out


def ensure_id_column(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Ensure a DataFrame has a canonical 'id' column.

    - If 'id' is missing but 'ID' exists, rename 'ID' -> 'id'.
    - If 'id' is still missing, raise a ValueError with a context-specific message.

    This helper centralizes the project's recurring CSV schema normalization.
    """
    if "id" not in df.columns and "ID" in df.columns:
        df = df.rename(columns={"ID": "id"})
    if "id" not in df.columns:
        raise ValueError(f"{context} must contain an 'id' column. Found: {list(df.columns)}")
    return df


def raise_if_unknown_labels(
    labels: List[str],
    label_universe: List[str],
    *,
    error_builder: Callable[[List[str]], str],
) -> None:
    """Raise a ValueError if any labels are not present in the label universe.

    This helper centralizes the project's repeated label-membership validation.
    Callers supply an error_builder to preserve existing, context-specific error
    messages.
    """
    unknown = [x for x in labels if x not in label_universe]
    if unknown:
        raise ValueError(error_builder(unknown))
