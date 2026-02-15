from pathlib import Path
from typing import Dict

import yaml


class RunLogger:
    def __init__(self, logs_dir: Path, ts: str) -> None:
        self.path = logs_dir.joinpath(f"{ts}.log")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_config(self, cfg: Dict) -> None:
        clean_cfg = {}
        for k, v in cfg.items():
            if k in {"labels"}:  # what to exclude from being logged
                continue
            # Render Paths as strings for YAML readability
            clean_cfg[k] = str(v) if hasattr(v, "__fspath__") else v

        with open(self.path, "w", encoding="utf-8") as f:
            f.write("# Active configuration\n")
            yaml.safe_dump(
                clean_cfg,
                f,
                sort_keys=False,
                default_flow_style=False,
                width=1_000_000,
            )
            f.write("=" * 60 + "\n")

    def write_metrics(self, metrics: Dict[str, float], per_label: bool = False) -> None:
        """
        Write metrics to the run log.
        - Always write the overall block.
        - Only write the 'Per-label metrics' section if per_label=True.
        """
        # Detect ICR-related keys
        any_alpha = any(k.startswith("krippendorff_alpha") for k in metrics.keys())

        if any_alpha:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n# Intercoder reliability (Krippendorff's α)\n")

                def _write_scalar(label: str, key: str) -> None:
                    if key in metrics and isinstance(metrics[key], (int, float)):
                        f.write(f"{label:35s}: {metrics[key]:.4f}\n")

                def _write_pairwise_atomic_like(keys: list[str], *, key_prefix: str) -> None:
                    # Pairwise human–human
                    for k in sorted(keys):
                        if k.startswith(key_prefix) and "_vs_" in k and not k.endswith("_vs_llm"):
                            human_pair = k.removeprefix(key_prefix).replace("_", " ")
                            f.write(f"pairwise humans ({human_pair}): {metrics[k]:.4f}\n")

                    # Pairwise human–LLM
                    for k in sorted(keys):
                        if k.startswith(key_prefix) and k.endswith("_vs_llm"):
                            human = (
                                k.removeprefix(key_prefix)
                                .removesuffix("_vs_llm")
                                .replace("_", " ")
                            )
                            f.write(f"human vs LLM ({human} vs LLM): {metrics[k]:.4f}\n")

                # ---------- Atomic (nominal) ----------
                # One-line description as requested:
                # Atomic multi-label tokens (exact match only)
                atomic_keys = [
                    k for k in metrics.keys()
                    if k.startswith("krippendorff_alpha")
                    and "binary" not in k
                    and "jaccard" not in k
                ]
                if atomic_keys:
                    f.write("\nAtomic multi-label tokens (exact match only)\n")

                    # Preferred order (same as before)
                    _write_scalar("overall (humans + LLM)", "krippendorff_alpha_overall")
                    _write_scalar("humans only", "krippendorff_alpha_humans")

                    _write_pairwise_atomic_like(atomic_keys, key_prefix="krippendorff_alpha_")

                # ---------- Binary-incidence per label ----------
                # One-line description as requested:
                # Binary-incidence α per label (+ macro / prevalence-weighted summaries)
                bin_keys = [k for k in metrics.keys() if k.startswith("krippendorff_alpha_binary_")]
                if bin_keys:
                    f.write("\nBinary-incidence α per label (+ macro / prevalence-weighted summaries)\n")

                    _write_scalar("overall (macro)", "krippendorff_alpha_binary_overall_macro")
                    _write_scalar(
                        "overall (prevalence-weighted)",
                        "krippendorff_alpha_binary_overall_prevalence_weighted",
                    )
                    _write_scalar("humans only (macro)", "krippendorff_alpha_binary_humans_macro")
                    _write_scalar(
                        "humans only (prevalence-weighted)",
                        "krippendorff_alpha_binary_humans_prevalence_weighted",
                    )

                    # Pairwise comparisons (computed by krippendorf_helper.compute_icr_alpha)
                    pair_vals: Dict[str, Dict[str, float]] = {}
                    for k in bin_keys:
                        if k.endswith("_macro"):
                            base = k.removesuffix("_macro")
                            pair_vals.setdefault(base, {})["macro"] = metrics[k]
                        elif k.endswith("_prevalence_weighted"):
                            base = k.removesuffix("_prevalence_weighted")
                            pair_vals.setdefault(base, {})["prevalence_weighted"] = metrics[k]

                    def _write_binary_pairwise(pair_map: Dict[str, Dict[str, float]]) -> None:
                        # Pairwise human–human
                        for base in sorted(pair_map):
                            if base in {"krippendorff_alpha_binary_overall", "krippendorff_alpha_binary_humans"}:
                                continue
                            if (
                                base.startswith("krippendorff_alpha_binary_")
                                and "_vs_" in base
                                and not base.endswith("_vs_llm")
                            ):
                                human_pair = base.removeprefix("krippendorff_alpha_binary_").replace("_", " ")
                                if "macro" in pair_map[base]:
                                    f.write(
                                        f"pairwise humans ({human_pair}) (macro): {pair_map[base]['macro']:.4f}\n"
                                    )
                                if "prevalence_weighted" in pair_map[base]:
                                    f.write(
                                        f"pairwise humans ({human_pair}) (prevalence-weighted): "
                                        f"{pair_map[base]['prevalence_weighted']:.4f}\n"
                                    )

                        # Pairwise human–LLM
                        for base in sorted(pair_map):
                            if base.startswith("krippendorff_alpha_binary_") and base.endswith("_vs_llm"):
                                human = (
                                    base.removeprefix("krippendorff_alpha_binary_")
                                    .removesuffix("_vs_llm")
                                    .replace("_", " ")
                                )
                                if "macro" in pair_map[base]:
                                    f.write(
                                        f"human vs LLM ({human} vs LLM) (macro): {pair_map[base]['macro']:.4f}\n"
                                    )
                                if "prevalence_weighted" in pair_map[base]:
                                    f.write(
                                        f"human vs LLM ({human} vs LLM) (prevalence-weighted): "
                                        f"{pair_map[base]['prevalence_weighted']:.4f}\n"
                                    )

                    _write_binary_pairwise(pair_vals)

                # ---------- Set-based Jaccard ----------
                # One-line description as requested:
                # Set-based α with Jaccard distance
                jac_keys = [k for k in metrics.keys() if k.startswith("krippendorff_alpha_jaccard_")]
                if jac_keys:
                    f.write("\nSet-based α with Jaccard distance\n")

                    _write_scalar("overall (humans + LLM)", "krippendorff_alpha_jaccard_overall")
                    _write_scalar("humans only", "krippendorff_alpha_jaccard_humans")

                    _write_pairwise_atomic_like(jac_keys, key_prefix="krippendorff_alpha_jaccard_")

            return

        # Otherwise: standard metrics (single-label or multi-label)
        if "subset_accuracy" in metrics:
            overall_keys = {
                "subset_accuracy",
                "micro_precision", "micro_recall", "micro_f1",
                "macro_precision", "macro_recall", "macro_f1",
            }
        else:
            overall_keys = {"accuracy", "macro_f1", "weighted_f1"}

        overall = {k: v for k, v in metrics.items() if k in overall_keys or "_" not in k}
        per_label_dict = {k: v for k, v in metrics.items() if k not in overall}

        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n# Metrics\n")
            for k in sorted(overall):
                v = overall[k]
                if isinstance(v, (int, float)):
                    f.write(f"{k:25s}: {v:.4f}\n")
                else:
                    f.write(f"{k:25s}: {v}\n")

            # Only emit per-label metrics when explicitly requested
            if per_label and per_label_dict:
                f.write("\n# Per-label metrics\n")
                grouped: Dict[str, Dict[str, float]] = {}
                for key, val in per_label_dict.items():
                    if "_" not in key:
                        continue
                    label, metric = key.split("_", 1)
                    grouped.setdefault(label, {})[metric] = val

                for label in sorted(grouped):
                    f.write(f"\n[{label}]\n")
                    for mkey, mval in sorted(grouped[label].items()):
                        if isinstance(mval, (int, float)):
                            f.write(f"  {mkey:15s}: {mval:.4f}\n")
                        else:
                            f.write(f"  {mkey:15s}: {mval}\n")

    def info(self, message: str) -> None:
        """Append a free-form note line (not part of metrics)."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n# Notes\n")
            f.write(f"NOTE: {message}\n")
