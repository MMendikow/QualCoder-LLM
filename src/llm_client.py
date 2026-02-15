from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from openai import OpenAI, OpenAIError

from .utils import raise_if_unknown_labels, trim_dedup_str_list

# Single canonical JSON-only instruction (kept short and unambiguous).
JSON_CONTRACT_BASE = (
    "Output only a single valid JSON array (from `[` to `]`) and nothing else (no markdown, no prose).\n"
)


def _build_json_contract(
    *,
    batch_size: int,
    allow_multilabel: bool,
    label_universe: List[str],
    include_evidence_lines: bool = False,
    include_explanation: bool = False,
) -> str:
    """
    Build strict format instructions + examples.
    Only affects prompting; does not change parsing/validation behavior.
    """

    # ---- element template line (dynamic) ----
    element_parts = ['"id": "<id>"', '"labels": ["<label>", ...]']
    if include_evidence_lines:
        element_parts.append('"evidence_lines": ["<verbatim quote from input>", ...]')
    if include_explanation:
        element_parts.append('"explanation": "<brief rationale for label choice>"')
    element_template = "Each element must be {" + ", ".join(element_parts) + "}."

    # ---- batch rule ----
    if batch_size <= 1:
        batch_rule = "The batch has 1 item, so output a JSON array with exactly 1 object."
    else:
        batch_rule = (
            f"The batch has {batch_size} items. Output exactly {batch_size} objects, "
            "one per id. Include every id exactly once (no missing / extra ids)."
        )

    # ---- label cardinality rule ----
    if allow_multilabel:
        label_rule = "Multi-label allowed: labels may contain multiple allowed labels."
    else:
        label_rule = "Single-label mode: labels must contain exactly 1 label."

    # ---- fallback rule (universal) ----
    na_rule = 'If none of the allowed labels fit the given text, output labels: ["n/a"].'

    # ---- allowed labels ----
    allowed = "Allowed labels (use exactly these spellings):\n" + "\n".join(f"- {x}" for x in label_universe)

    # ---- examples (structure-only; placeholders) ----
    def _example_obj(id_placeholder: str, labels_expr: str) -> str:
        obj = f'{{"id":"{id_placeholder}","labels":{labels_expr}'
        if include_evidence_lines:
            obj += ',"evidence_lines":["<VERBATIM_QUOTE_1_FROM_INPUT>","<VERBATIM_QUOTE_2_FROM_INPUT>"]'
        if include_explanation:
            obj += ',"explanation":"<BRIEF_RATIONALE_FOR_LABEL_CHOICE>"'
        obj += "}"
        return obj

    ex_single = "[" + _example_obj("<ID_FROM_BATCH>", '["<ALLOWED_LABEL>"]') + "]"
    ex_multi = "[" + _example_obj("<ID_FROM_BATCH>", '["<ALLOWED_LABEL_1>","<ALLOWED_LABEL_2>"]') + "]"

    if batch_size > 1:
        ex_batch = (
            "["
            + _example_obj("<ID_1_FROM_BATCH>", '["<ALLOWED_LABEL>"]')
            + ","
            + _example_obj("<ID_2_FROM_BATCH>", '["<ALLOWED_LABEL>"]')
            + "]"
        )
    else:
        ex_batch = ex_single

    examples_lines: List[str] = [
        "Examples (placeholders only; copy the structure, not the placeholder content):",
        f"- Single-label structure: {ex_single}",
    ]
    if allow_multilabel:
        examples_lines.append(f"- Multi-label structure: {ex_multi}")
    if batch_size > 1:
        examples_lines.append(f"- Multi-item batch structure: {ex_batch}")

    examples = "\n".join(examples_lines)

    extra_rules: List[str] = []
    if include_evidence_lines:
        extra_rules.append(
            "evidence_lines must contain short verbatim quotes copied exactly from the input text that justify the chosen label(s)."
        )
    if include_explanation:
        extra_rules.append(
            "explanation must be a brief rationale explaining why you selected the label(s) for this item, grounded in the provided input."
        )

    if allow_multilabel:
        extra_rules.append(
            'Only output labels: ["n/a"] if no other allowed label applies; do not combine "n/a" with any other label.'
        )

    return "\n".join(
        [
            JSON_CONTRACT_BASE.rstrip(),
            element_template,
            batch_rule,
            label_rule,
            "Each label must be copied verbatim from the Allowed labels list; do not rephrase.",
            *extra_rules,
            na_rule,
            allowed,
            examples,
        ]
    )


def _extract_json_array(text: str) -> str:
    """
    Best-effort cleanup for free-text responses:
    - Strip common Markdown code fences (```json ... ```, ``` ... ```)
    - Return the outermost JSON array substring [ ... ]
    Raises ValueError if no plausible array is found.
    """
    s = text.strip()

    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()

    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No top-level JSON array found in response.")
    return s[start : end + 1].strip()


def _normalize_prediction_item(
    item: Dict,
    *,
    allow_multilabel: bool,
    label_universe: List[str],
    require_evidence_lines: bool = False,
    require_explanation: bool = False,
) -> Dict:
    if "id" not in item:
        raise ValueError("Prediction item missing required 'id'")
    pid = str(item["id"]).strip()

    labels = item.get("labels", [])
    if labels is None:
        labels = []
    elif isinstance(labels, str):
        labels = [labels]
    elif not isinstance(labels, (list, tuple)):
        labels = [str(labels)]

    cleaned = trim_dedup_str_list(labels)

    if not cleaned:
        cleaned = ["n/a"]

    if "n/a" in cleaned and len(cleaned) > 1:
        raise ValueError(f'id={pid}: "n/a" must not be combined with other labels: {cleaned}')

    if not allow_multilabel and len(cleaned) > 1:
        raise ValueError(f"Single-label mode violation for id={pid}: {cleaned}")

    raise_if_unknown_labels(
        cleaned,
        label_universe,
        error_builder=lambda unknown: f"id={pid}: unknown labels not in universe: {unknown}",
    )

    ev = item.get("evidence_lines", []) or []
    if isinstance(ev, str):
        ev = [ev]
    ev = trim_dedup_str_list(ev)

    explanation_raw = item.get("explanation")
    explanation = None
    if explanation_raw is not None:
        explanation = str(explanation_raw).strip()

    if require_evidence_lines and not ev:
        raise ValueError(f"id={pid}: missing required evidence_lines")
    if require_explanation and not explanation:
        raise ValueError(f"id={pid}: missing required explanation")

    out: Dict[str, Any] = {
        "id": pid,
        "labels": cleaned,
    }
    if ev:
        out["evidence_lines"] = ev
    if explanation:
        out["explanation"] = explanation
    return out


@dataclass(frozen=True, slots=True)
class LLMClient:
    """
    Thin wrapper around OpenAI chat completions, plus strict JSON parsing/validation.

    IMPORTANT (bug fix):
    Because this dataclass uses slots=True, any "private" attributes must be
    declared as fields; otherwise object.__setattr__ will raise AttributeError.
    """
    model: str
    temperature: float
    model_role: str

    # ---- private cached state (must be declared with slots=True) ----
    _client: OpenAI | None = field(default=None, init=False, repr=False)
    _last_full_prompt: str | None = field(default=None, init=False, repr=False)

    @property
    def client(self) -> OpenAI:
        c = self._client
        if c is not None:
            return c

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not provided")

        c = OpenAI(api_key=api_key)
        object.__setattr__(self, "_client", c)
        return c

    def _build_batch_payload(self, batch: List[Dict]) -> List[Dict]:
        """
        Build the per-item payload sent to the model.
        We keep only id/title and either sections or text.
        """
        payload: List[Dict] = []
        for it in batch:
            pid = str(it.get("id", "")).strip()
            title = (it.get("title") or "").strip() if it.get("title") else None

            entry: Dict[str, Any] = {"id": pid}
            if title:
                entry["title"] = title

            if it.get("sections"):
                entry["sections"] = it["sections"]
            else:
                entry["text"] = it.get("text") or ""

            payload.append(entry)
        return payload

    def classify_batch(
        self,
        prompt_template: str,
        batch: List[Dict],
        *,
        allow_multilabel: bool,
        label_universe: List[str],
        include_evidence_lines: bool = False,
        include_explanation: bool = False,
    ) -> Tuple[List[Dict], str]:
        """
        Returns (normalized_predictions, full_prompt_used).
        """
        payload = self._build_batch_payload(batch)
        expected_ids = {x["id"] for x in payload}

        batch_text = json.dumps(payload, ensure_ascii=False, indent=None)

        json_contract = _build_json_contract(
            batch_size=len(payload),
            allow_multilabel=allow_multilabel,
            label_universe=label_universe,
            include_evidence_lines=include_evidence_lines,
            include_explanation=include_explanation,
        )

        full_prompt = (
            f"{prompt_template}\n\n{json_contract}\n"
            f"INPUT (analyse and code only the items below):\n{batch_text}"
        )

        # frozen + slots => must use object.__setattr__
        object.__setattr__(self, "_last_full_prompt", full_prompt)

        messages: List[Dict[str, str]] = []
        if self.model_role:
            messages.append({"role": "system", "content": self.model_role})
        messages.append({"role": "user", "content": full_prompt})

        # Prefer strict JSON via json_schema; fall back if unsupported
        response_format_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "batch_labels",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "labels": {"type": "array", "items": {"type": "string"}},
                            "evidence_lines": {"type": "array", "items": {"type": "string"}},
                            "explanation": {"type": "string"},
                        },
                        "required": ["id", "labels"],
                        "additionalProperties": True,
                    },
                },
                "strict": True,
            },
        }

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format=response_format_schema,
            )
        except OpenAIError:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )

        content = resp.choices[0].message.content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            candidate = _extract_json_array(content)
            parsed = json.loads(candidate)
            content = candidate

        if not isinstance(parsed, list):
            raise ValueError("Top-level JSON must be a list")

        returned_ids = [str(x.get("id", "")).strip() for x in parsed]
        if len(returned_ids) != len(set(returned_ids)):
            raise ValueError("Duplicate ids in model output")
        if set(returned_ids) != expected_ids:
            raise ValueError(
                f"Response ids mismatch. expected={sorted(expected_ids)} got={sorted(set(returned_ids))}"
            )

        for x in parsed:
            id_str = str(x.get("id", "")).strip()
            if id_str not in expected_ids:
                raise ValueError(f"Unexpected id in model output: {id_str!r}")

        normed = [
            _normalize_prediction_item(
                x,
                allow_multilabel=allow_multilabel,
                label_universe=label_universe,
                require_evidence_lines=include_evidence_lines,
                require_explanation=include_explanation,
            )
            for x in parsed
        ]
        return normed, full_prompt
