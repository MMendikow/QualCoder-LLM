from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


def _infer_repo_root(config_dir: Path) -> Path:
    """Infer the repository root given the directory containing the YAML config.

    Heuristic
    ---------
    - If the config directory itself contains `data/`, treat it as the root.
    - Else if its parent contains `data/`, treat the parent as the root
      (common when configs live in `configs/`).
    - Else fall back to the config directory.
    """
    if (config_dir / "data").is_dir():
        return config_dir
    if (config_dir.parent / "data").is_dir():
        return config_dir.parent
    return config_dir


def _resolve_path(raw_value: str | Path, *, base_dir: Path, repo_root: Path) -> Path:
    """Resolve a user-supplied path with a key-specific default base directory.

    Rules
    -----
    - Absolute paths are used as-is.
    - Relative paths beginning with a top-level repo folder (`data/`, `outputs/`,
      `configs/`, `src/`) are interpreted relative to the repository root.
    - All other relative paths are interpreted relative to `base_dir`.
    """
    p = Path(str(raw_value)).expanduser()
    if p.is_absolute():
        return p.resolve()

    if p.parts and p.parts[0] in {"data", "outputs", "configs", "src"}:
        return (repo_root / p).resolve()

    return (base_dir / p).resolve()


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Runtime configuration loaded from a single flat YAML file.

    The YAML schema is intentionally strict: only the flat key set documented in
    the example config is accepted. This avoids user confusion around multiple
    competing config layouts and aliases.
    """

    # Outputs (derived from project_name)
    project_name: str
    logs_dir: Path
    results_dir: Path

    # Inputs
    prompt_path: Path
    text_dir: Path
    hand_coded_validation_path: Path | None
    labels_txt_path: Path

    # Labels
    labels: List[str]

    # OpenAI model parameters
    llm: str
    temperature: float
    model_role: str

    # Runtime
    batch_size: int
    max_retries: int

    # Task policy
    validation_mode: bool
    allow_multilabel: bool

    # Optional helpers
    include_evidence_lines: bool = False
    include_explanation: bool = False
    hallucination_check: bool = False

    # Optional: evaluation-only mode (no API calls).
    canonical_raw_path: Path | None = None

    @staticmethod
    def _load_labels_txt(path: Path) -> List[str]:
        """Load labels from a txt file (one label per line).

        Duplicates are removed while preserving order. The label universe is
        extended with "n/a" if it is not present.
        """
        lines = path.read_text(encoding="utf-8").splitlines()
        parts = [ln.strip() for ln in lines if ln.strip()]

        seen: set[str] = set()
        labels: List[str] = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                labels.append(p)

        if "n/a" not in seen:
            labels.append("n/a")

        if not labels:
            raise ValueError(f"No labels found in labels txt: {path}")
        return labels

    @staticmethod
    def from_yaml(path: str | Path) -> "PipelineConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if not isinstance(raw, dict):
            raise ValueError("Config YAML must decode to a mapping (key/value object).")

        allowed_keys = {
            "project_name",
            "prompt_path",
            "text_dir",
            "hand_coded_validation_path",
            "labels_txt_path",
            "llm",
            "temperature",
            "model_role",
            "batch_size",
            "max_retries",
            "validation_mode",
            "allow_multilabel",
            "include_evidence_lines",
            "include_explanation",
            "hallucination_check",
            "canonical_raw_path",
        }
        extra = set(raw.keys()) - allowed_keys
        if extra:
            raise ValueError(
                "Unsupported config keys (only the documented flat schema is accepted): "
                + ", ".join(sorted(extra))
            )

        required_keys = {
            "project_name",
            "prompt_path",
            "text_dir",
            "labels_txt_path",
            "llm",
            "batch_size",
            "max_retries",
            "allow_multilabel",
        }
        missing = required_keys - set(raw.keys())
        if missing:
            raise ValueError(f"Missing required config keys: {sorted(missing)}")

        project_name = str(raw["project_name"]).strip()
        if not project_name:
            raise ValueError("project_name must be a non-empty string")

        repo_root = _infer_repo_root(config_path.parent)
        data_root = repo_root / "data"
        outputs_root = repo_root / "outputs"

        prompt_path = _resolve_path(raw["prompt_path"], base_dir=data_root / "prompts", repo_root=repo_root)
        text_dir = _resolve_path(raw["text_dir"], base_dir=data_root / "text", repo_root=repo_root)
        labels_txt_path = _resolve_path(raw["labels_txt_path"], base_dir=data_root / "labels", repo_root=repo_root)

        validation_mode = bool(raw.get("validation_mode", True))
        if validation_mode and "hand_coded_validation_path" not in raw:
            raise ValueError("hand_coded_validation_path is required when validation_mode is true")

        hand_coded_validation_raw = raw.get("hand_coded_validation_path")
        hand_coded_validation_path = (
            _resolve_path(
                hand_coded_validation_raw,
                base_dir=data_root / "validation",
                repo_root=repo_root,
            )
            if hand_coded_validation_raw
            else None
        )

        labels = PipelineConfig._load_labels_txt(labels_txt_path)

        # Outputs derived solely from project_name: outputs/<project_name>/{logs,results}
        project_root = outputs_root / project_name
        logs_dir = project_root / "logs"
        results_dir = project_root / "results"

        llm_value = str(raw["llm"]).strip()
        if not llm_value:
            raise ValueError("llm must be a non-empty string")

        batch_size = int(raw["batch_size"])
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        max_retries = int(raw["max_retries"])
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        allow_multilabel = bool(raw["allow_multilabel"])

        include_evidence_lines = bool(raw.get("include_evidence_lines", False))
        include_explanation = bool(raw.get("include_explanation", False))
        hallucination_check = bool(raw.get("hallucination_check", False))

        canonical_raw = raw.get("canonical_raw_path")
        canonical_raw_path = (
            _resolve_path(canonical_raw, base_dir=repo_root, repo_root=repo_root) if canonical_raw else None
        )

        if not validation_mode and canonical_raw_path is not None:
            raise ValueError("canonical_raw_path cannot be used when validation_mode is false")

        if not validation_mode and hallucination_check:
            raise ValueError("hallucination_check cannot be enabled when validation_mode is false")

        return PipelineConfig(
            project_name=project_name,
            logs_dir=logs_dir,
            results_dir=results_dir,
            prompt_path=prompt_path,
            text_dir=text_dir,
            hand_coded_validation_path=hand_coded_validation_path,
            labels_txt_path=labels_txt_path,
            labels=labels,
            llm=llm_value,
            temperature=float(raw.get("temperature", 0.0)),
            model_role=str(raw.get("model_role", "")),
            batch_size=batch_size,
            max_retries=max_retries,
            validation_mode=validation_mode,
            allow_multilabel=allow_multilabel,
            include_evidence_lines=include_evidence_lines,
            include_explanation=include_explanation,
            hallucination_check=hallucination_check,
            canonical_raw_path=canonical_raw_path,
        )