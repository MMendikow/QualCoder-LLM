from __future__ import annotations

from typing import List, TypedDict

__all__ = ["TextItem", "PredictionItem"]


class TextItemBase(TypedDict):
    """
    Normalized paper/text record produced by loader.load_papers().

    Guaranteed by loader:
      - id: present and normalized to a stripped string
      - text: present ("" if missing/None)
    """
    id: str
    text: str


class TextItem(TextItemBase, total=False):
    """
    Optional fields may or may not be present in the JSON input files.
    """
    title: str | None
    sections: dict[str, str]


class PredictionItemBase(TypedDict):
    """
    Normalized prediction item produced by LLMClient.classify_batch() and written to the canonical JSONL.
    """
    id: str
    labels: List[str]


class PredictionItem(PredictionItemBase, total=False):
    """
    Optional fields appear only when requested by configuration/prompting.
    """
    evidence_lines: List[str]
    explanation: str
