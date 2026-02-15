# QualCoder-LLM

QualCoder-LLM is a lightweight, reproducible pipeline for **LLM-assisted qualitative coding** of short texts stored as JSON files. It supports:

* **Single-label** or **multi-label** coding with an OpenAI model.
* **Gold (ground-truth) evaluation** against an explicitly labeled validation dataset.
* **Intercoder reliability (ICR)** using **Krippendorff’s α** on rater-style validation datasets (including the LLM as an additional coder).
* **Experiment tracking via `run_history.csv`**, which records key configuration settings and headline metrics for every run—making it straightforward to compare prompts, models, label sets, and options over time.

Everything is controlled via a **single flat YAML config file**. Outputs are written to `outputs/<project_name>/...`.

---

## Table of contents

* [Repository structure](#repository-structure)
* [What the pipeline does](#what-the-pipeline-does)
* [Requirements](#requirements)
* [Installation](#installation)
* [Quick start (included examples)](#quick-start-included-examples)
* [Preparing your inputs](#preparing-your-inputs)

  * [1) Label universe](#1-label-universe)
  * [2) Prompt template](#2-prompt-template)
  * [3) Text items (JSON)](#3-text-items-json)
  * [4) Validation file (CSV)](#4-validation-file-csv)
  * [How evaluation mode is inferred](#how-evaluation-mode-is-inferred)
* [Configuration (YAML)](#configuration-yaml)

  * [Path resolution](#path-resolution)
* [Running the pipeline](#running-the-pipeline)
* [Outputs](#outputs)

  * [Run directory](#run-directory)
  * [`run_history.csv` (why it matters)](#run_historycsv-why-it-matters)
* [Optional features](#optional-features)

  * [Evidence lines](#evidence-lines)
  * [Explanation](#explanation)
  * [Hallucination check](#hallucination-check)
  * [Evaluation-only mode (no API calls)](#evaluation-only-mode-no-api-calls)
* [Metrics (with formulas)](#metrics-with-formulas)

  * [Single-label (gold mode)](#single-label-gold-mode)
  * [Multi-label (gold mode)](#multi-label-gold-mode)
  * [ICR (Krippendorff’s α)](#icr-krippendorffs-α)
* [Troubleshooting](#troubleshooting)
* [Minimal checklist (recommended workflow)](#minimal-checklist-recommended-workflow)

---

## Repository structure

```
configs/          # YAML configs (recommended location)
data/
  prompts/        # prompt templates (.txt)
  text/           # text items as .json files
  labels/         # label sets (.txt)
  validation/     # validation datasets (.csv): gold OR rater-style
outputs/          # auto-generated logs and results
src/              # pipeline source code
.env
.env.example
requirements.txt
run.py
```

---

## What the pipeline does

At a high level, the pipeline performs four steps:

1. **Load inputs**

   * label universe from a `.txt` file
   * prompt template from a `.txt` file
   * text items from a directory of `.json` files
   * validation data from a `.csv` file

2. **Call the LLM to code items** (unless you run in evaluation-only mode)

   * batching is controlled by `batch_size`
   * strict output format is enforced (JSON array with `id` and `labels`)

3. **Evaluate results**

   * **Gold mode** if the validation CSV contains a `labels` column
   * **ICR mode** otherwise (≥1 rater columns besides `id`)

4. **Write outputs**

   * per-run artifacts into `outputs/<project_name>/results/run_<timestamp>/...`
   * a cumulative `run_history.csv` for cross-run comparison

---

## Requirements

* Python **3.10+**
* An OpenAI API key

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your `.env` file for the OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Quick start (included examples)

This repository includes two example YAML configs:

* `example_singlelabel_groundtruth.yaml` (single-label + gold evaluation)
* `example_multilabel.yaml` (multi-label + gold evaluation)

Run one example:

```bash
python run.py configs/example_singlelabel_groundtruth.yaml
```

or:

```bash
python run.py configs/example_multilabel.yaml
```

Equivalent entrypoint:

```bash
python -m src.main configs/example_singlelabel_groundtruth.yaml
```

---

## Preparing your inputs

This section is intentionally detailed: most runtime errors come from small input-format mistakes.

### 1) Label universe

**File:** `data/labels/<your_labels>.txt`
**Format:** one label per line (exact spelling is important)

Example:

```
economic_strain
care_responsibilities
mental_wellbeing
trust_in_institutions
```

Rules enforced by the pipeline:

* Duplicate lines are removed (order preserved).
* `"n/a"` is automatically appended if missing.
* Model outputs and validation labels are validated against this universe; unknown labels raise an error.

---

### 2) Prompt template

**File:** `data/prompts/<your_prompt>.txt`
**Format:** plain English instructions + a codebook (recommended), written for a human coder.

A practical prompt typically includes:

* a short definition of the task (what counts as evidence for each code)
* inclusion/exclusion criteria per label
* guidance for ambiguous cases
* explicit guidance on when to choose `"n/a"`

#### What the pipeline appends to your prompt (default model instructions)

You do **not** need to specify output formatting rules yourself. At runtime, the pipeline appends a strict “JSON contract” to every batch prompt to enforce:

* **Output must be a single valid JSON array and nothing else** (no Markdown, no prose).
* Each element must include:

  * `"id": "<id>"`
  * `"labels": ["<label>", ...]`
* **Batch integrity**

  * output exactly one object per input item
  * each `id` appears exactly once
* **Label policy**

  * labels must match the allowed label list *verbatim*
  * **single-label mode:** exactly 1 label per item
  * **multi-label mode:** multiple labels allowed
  * `"n/a"` must not be combined with other labels
* The **full allowed label list** is included in the appended contract.
* The **INPUT payload** is appended as a JSON array of items (each item contains `id` and either `text` or `sections`, and optionally `title`).

This design keeps your prompt readable while still forcing machine-checkable outputs.

---

### 3) Text items (JSON)

**Directory:** `data/text/<your_text_dir>/`
**Format:** one `.json` file per item.

Minimum required schema:

```json
{
  "id": "optional-string-id",
  "text": "the excerpt to code"
}
```

Normalization performed by the loader:

* If `"id"` is missing, it defaults to the filename stem.
* `"id"` is normalized to a stripped string.
* If `"text"` is missing or `null`, it becomes `""`.

Optional fields (supported):

* `"title": "..."` (string or null)
* `"sections": { "Section A": "...", "Section B": "..." }`

If `sections` exists and is non-empty, the pipeline sends `sections` to the model; otherwise it sends `text`.

---

### 4) Validation file (CSV)

A **CSV** (“comma-separated values”) file is a plain-text table format. You can create it in:

* Excel / LibreOffice / Numbers → *Save As* or *Export* → **CSV**
* Google Sheets → *File → Download → Comma-separated values (.csv)*

Important export advice:

* Prefer **comma-delimited** CSV (`,`), because the pipeline loads with standard CSV defaults.
* Prefer **UTF-8** encoding.
* If your region exports semicolon-delimited CSV (`;`), re-export with commas (or convert delimiters) before running.

There are two supported validation layouts.

#### A) Gold (ground-truth) CSV — explicit labels

Gold evaluation is used when (and only when) your CSV contains a **`labels`** column.

Required columns:

* `id` (or `ID`)
* `labels`

**Single-label gold**:

```csv
id,labels
001,economic_strain
002,trust_in_institutions
```

**Multi-label gold** (labels stored as a JSON array *string*):

```csv
id,labels
001,"[""economic_strain"",""mental_wellbeing""]"
002,"[""n/a""]"
```

How to enter multi-label cells safely:

* In spreadsheet cells, write valid JSON like:

  * `["economic_strain","mental_wellbeing"]`
* When exported to CSV, quoting often becomes:

  * `"[""economic_strain"",""mental_wellbeing""]"`
    (this is normal CSV escaping)

Parsing rules enforced by the pipeline:

* `labels` may be a plain string or a JSON array string.
* Empty / missing cells become `[]`.
* If `allow_multilabel: false`, cells with more than one label raise an error.
* All labels are validated against the label universe.

#### B) ICR (rater-style) CSV — no explicit gold labels

ICR mode is used when the CSV **does not** contain a `labels` column.

Required columns:

* `id` (or `ID`)
* **at least one rater column** (any name is fine), e.g. `rater_1`

Example:

```csv
id,rater_1,rater_2
001,economic_strain,"[""economic_strain"",""mental_wellbeing""]"
002,trust_in_institutions,trust_in_institutions
```

Key points:

* **ICR mode works even with only one rater column.** In that case, agreement is computed between that rater and the LLM (because the pipeline adds the LLM as an additional coder). “Humans-only” reliability requires ≥2 human raters.
* Rater cells may be single labels or JSON arrays (multi-label is always allowed in ICR parsing).
* Missing cells are treated as missing coder decisions (not empty label sets).

---

### How evaluation mode is inferred

This rule is central:

* If your validation CSV contains a **`labels`** column → **Gold evaluation**.
* Otherwise, if it contains **≥1 rater columns besides `id`** → **ICR evaluation**.

In practice: **adding or removing the column name `labels` is the switch** between gold mode and ICR mode.

---

## Configuration (YAML)

The config is a single flat YAML file (only documented keys are accepted).

### Required keys

* `project_name`
  Output root: `outputs/<project_name>/{logs,results}`

* Inputs

  * `prompt_path`
  * `text_dir`
  * `hand_coded_validation_path`
  * `labels_txt_path`

* Model / runtime

  * `llm`
  * `batch_size`
  * `max_retries`

* Task policy

  * `allow_multilabel` (`true` or `false`)

### Optional keys

* `temperature` (default: `0.0`)
* `model_role` (default: `""`) — optional system message
* `include_evidence_lines` (default: `false`)
* `include_explanation` (default: `false`)
* `hallucination_check` (default: `false`)
* `canonical_raw_path` (default: `null`) — evaluation-only mode

Example config:

```yaml
project_name: my_run

prompt_path: my_prompt.txt
text_dir: my_texts
hand_coded_validation_path: my_validation.csv
labels_txt_path: my_labels.txt

llm: gpt-4.1-nano
temperature: 0.0
model_role: "You are a careful qualitative coder."

batch_size: 5
max_retries: 2

allow_multilabel: true

include_evidence_lines: false
include_explanation: false
hallucination_check: false
# canonical_raw_path: outputs/my_run/results/run_YYYY-MM-DDTHH-MM-SS/raw_output.jsonl
```

---

## Path resolution

If you use relative paths, they default to:

* `prompt_path` → `data/prompts/`
* `text_dir` → `data/text/`
* `hand_coded_validation_path` → `data/validation/`
* `labels_txt_path` → `data/labels/`

So this:

```yaml
prompt_path: my_prompt.txt
```

means:

```
data/prompts/my_prompt.txt
```

Paths starting with `data/`, `outputs/`, `configs/`, or `src/` are treated as repo-root relative.

---

## Running the pipeline

Run with:

```bash
python run.py configs/<your_config>.yaml
```

Alternative entrypoint:

```bash
python -m src.main configs/<your_config>.yaml
```

---

## Outputs

### Run directory

Each run writes to:

```
outputs/<project_name>/results/run_<timestamp>/
```

Typical files:

* `raw_output.jsonl`
  Canonical predictions, one JSON object per line:

  ```json
  {"id":"001","labels":["economic_strain"]}
  ```

  If enabled, items may additionally include:

  * `evidence_lines: [...]`
  * `explanation: "..."`

* `raw_input.txt`
  The exact prompt text sent to the model per batch (including the appended JSON contract and batch payload). This is the primary audit/debug artifact.

* `crosstab_log.csv`
  Only for **single-label gold** runs: a confusion-matrix-style cross-tabulation.

* `hallucinations.log`
  Only when `hallucination_check: true` (and only in gold mode).

Logs are written to:

```
outputs/<project_name>/logs/<timestamp>.log
```

---

### `run_history.csv` (why it matters)

`run_history.csv` is stored at:

```
outputs/<project_name>/results/run_history.csv
```

It appends one row per run containing:

* key overall metrics (gold metrics or ICR α metrics)
* the most relevant configuration parameters (e.g., model, role, batch size, evidence/explanation flags)

This enables a practical “experiment loop”:

* adjust prompt/codebook → rerun → compare results
* swap models → rerun → compare results
* change label sets → rerun → compare results
* enable/disable evidence and hallucination checks → rerun → compare outcomes and auditability

---

## Optional features

### Evidence lines

If `include_evidence_lines: true`, the model is required to return:

```json
"evidence_lines": ["<verbatim quote from input>", "..."]
```

Intended use:

* improve auditability (“what in the text supports this code?”)
* support qualitative review and coder training

The pipeline validates formatting and (optionally) checks literal presence (see hallucination check).

---

### Explanation

If `include_explanation: true`, the model is required to return:

```json
"explanation": "<brief rationale>"
```

This is a short justification, useful for interpretability and manual review.

---

### Hallucination check

If `hallucination_check: true` (gold mode only), the pipeline verifies that every `evidence_lines` entry appears **verbatim** in the source text after:

* Unicode normalization (NFKC),
* lowercasing,
* whitespace collapsing.

Mismatches are counted and written to `hallucinations.log`. This check **does not change predictions or metrics**; it is purely an audit signal.

---

### Evaluation-only mode (no API calls)

If you already have predictions and want to recompute metrics:

```yaml
canonical_raw_path: outputs/<project_name>/results/run_.../raw_output.jsonl
```

Then run as usual:

```bash
python run.py configs/<your_config>.yaml
```

The pipeline reads `raw_output.jsonl` and re-evaluates it. This is useful for:

* changing the validation CSV
* re-running ICR calculations
* re-computing metrics after modifying the label universe
* regenerating artifacts like the single-label crosstab

---

## Metrics (with formulas)

### Single-label (gold mode)

Let there be (N) items and a label set (\mathcal{L}). Each item has exactly one gold label (y_i \in \mathcal{L}) and one predicted label (\hat{y}_i \in \mathcal{L}).

**Accuracy**
[
\text{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[y_i=\hat{y}_i]
]

For each label (\ell \in \mathcal{L}):

* (TP_\ell): items where (y_i=\ell) and (\hat{y}_i=\ell)
* (FP_\ell): items where (y_i\neq\ell) and (\hat{y}_i=\ell)
* (FN_\ell): items where (y_i=\ell) and (\hat{y}_i\neq\ell)

**Precision / Recall / F1**
[
P_\ell=\frac{TP_\ell}{TP_\ell+FP_\ell},\quad
R_\ell=\frac{TP_\ell}{TP_\ell+FN_\ell},\quad
F1_\ell=\frac{2P_\ell R_\ell}{P_\ell+R_\ell}
]

**Macro-F1**
[
\text{Macro-F1}=\frac{1}{|\mathcal{L}|}\sum_{\ell\in\mathcal{L}}F1_\ell
]

**Weighted-F1** (support-weighted by gold frequency)
[
\text{Weighted-F1}=\sum_{\ell\in\mathcal{L}}\frac{\text{support}*\ell}{N},F1*\ell
\quad\text{where}\quad \text{support}*\ell=TP*\ell+FN_\ell
]

---

### Multi-label (gold mode)

Each item (i) has a gold label set (G_i \subseteq \mathcal{L}) and predicted set (P_i \subseteq \mathcal{L}).

**Subset accuracy** (exact set match)
[
\text{SubsetAcc}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[G_i=P_i]
]

Define micro counts aggregated across items:

[
TP=\sum_{i=1}^{N}|G_i\cap P_i|,\quad
FP=\sum_{i=1}^{N}|P_i\setminus G_i|,\quad
FN=\sum_{i=1}^{N}|G_i\setminus P_i|
]

**Micro precision / recall / F1**
[
P_{\text{micro}}=\frac{TP}{TP+FP},\quad
R_{\text{micro}}=\frac{TP}{TP+FN},\quad
F1_{\text{micro}}=\frac{2P_{\text{micro}}R_{\text{micro}}}{P_{\text{micro}}+R_{\text{micro}}}
]

For macro metrics, compute per-label (TP_\ell, FP_\ell, FN_\ell) by treating each label as a binary decision (“present” vs “absent”), then:

[
P_{\text{macro}}=\frac{1}{|\mathcal{L}|}\sum_{\ell}P_\ell,\quad
R_{\text{macro}}=\frac{1}{|\mathcal{L}|}\sum_{\ell}R_\ell,\quad
F1_{\text{macro}}=\frac{1}{|\mathcal{L}|}\sum_{\ell}F1_\ell
]

---

### ICR (Krippendorff’s α)

Krippendorff’s alpha is reported in several variants. The general form is:

[
\alpha = 1 - \frac{D_o}{D_e}
]

* (D_o): observed disagreement
* (D_e): expected disagreement by chance

Interpretation:

* (\alpha = 1): perfect agreement
* (\alpha = 0): chance-level agreement
* (\alpha < 0): systematic disagreement

#### 1) Atomic nominal α (exact set token match)

Each coder’s label set is converted into a canonical token:

* sort unique labels
* join with `|`
* empty set becomes `<EMPTY>`

Example: `["b","a","a"] → "a|b"`

Nominal α then follows the standard observed/expected disagreement formulation using category counts per item and global marginals.

#### 2) Binary-incidence α (per-label presence/absence)

For each label (\ell), each coder decision is converted into a binary value:

* 1 if (\ell) is present
* 0 if (\ell) is absent
* missing remains missing

Compute (\alpha_\ell) per label and report:

**Macro binary-incidence α**
[
\alpha_{\text{macro}}=\frac{1}{|\mathcal{L}|}\sum_{\ell\in\mathcal{L}}\alpha_\ell
]

**Prevalence-weighted binary-incidence α**

Let (w_\ell) be the number of positive assignments for label (\ell) across all items/coders:

[
\alpha_{\text{pw}}=\frac{\sum_{\ell} w_\ell \alpha_\ell}{\sum_{\ell} w_\ell}
]

#### 3) Jaccard set-based α (partial overlap)

This variant treats decisions as sets and uses Jaccard distance:

[
\delta(S,T)=1-\frac{|S\cap T|}{|S\cup T|}
]

Krippendorff’s (\alpha) is computed using the general metric form with (\delta) for observed and expected disagreement over coder pairs.

---

## Troubleshooting

* **The run is evaluated as ICR, but you expected gold evaluation**

  * Ensure the validation CSV contains a column named exactly `labels`.

* **The run is evaluated as gold, but you expected ICR**

  * Remove/rename the `labels` column and use rater columns instead.

* **CSV loads as a single column**

  * Your CSV is likely semicolon-delimited (`;`). Export as comma-delimited CSV.

* **Unknown label errors**

  * Ensure every label in the CSV is present in the label universe file (`labels_txt_path`).

* **No overlapping ids**

  * Ensure the `id` values in JSON files match the `id` column in the validation CSV (or omit `id` in JSON and rely on filename stems).

* **Model output parsing errors**

  * Keep `temperature: 0.0`, reduce `batch_size`, and increase `max_retries`. The pipeline enforces strict JSON-array output.

---

## Minimal checklist (recommended workflow)

1. Define labels in `data/labels/`
2. Write a prompt in `data/prompts/`
3. Add text JSON files in `data/text/`
4. Add a validation CSV in `data/validation/` (gold or rater-style)
5. Create one YAML config in `configs/`
6. Run `python run.py configs/<your_config>.yaml`
7. Inspect `outputs/<project_name>/logs/` and `outputs/<project_name>/results/`
