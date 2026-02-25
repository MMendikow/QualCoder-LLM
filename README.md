<p align="center">
  <img alt="QualCoder-LLM: LLM-assisted qualitative coding" src="assets/qualcoder_banner.svg" width="900" />
</p>

QualCoder-LLM is a **ready-to-use** pipeline for **LLM-assisted qualitative coding** of text documents. It supports:

* **Single-label** and **multi-label** coding.
* **Automated evaluation**, including:
  * **Ground-truth evaluation** against an expert-labeled validation dataset using standard classification metrics (e.g., accuracy and F1 variants).
  * **Intercoder reliability (ICR)** computing **Krippendorff’s α** treating the LLM as an additional coder alongside one or more human coders. 
* **Experiment tracking via `run_history.csv`**, which records key configuration settings and headline metrics for every run—making it straightforward to compare prompts, models, label sets, and options over time.
* Optional **evidence lines** (verbatim quotes from the source text supporting the assigned label(s)) with a **hallucination audit** that flags evidence lines that differ from the original input text.
* **Robust execution with configurable retries**: single failed LLM coding calls are retried up to a user-defined limit, while successful calls are saved so completed work is never lost.

Everything is controlled via a **single flat YAML configuration file**.


## Table of contents

- [Repository layout](#repository-layout)
- [What the pipeline does](#what-the-pipeline-does)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start (included examples)](#quick-start-included-examples)
- [Preparing your inputs](#preparing-your-inputs)
- [Validation file (CSV)](#4-validation-file-csv)
- [Configuration (YAML)](#configuration-yaml)
- [Path resolution](#path-resolution)
- [Running the pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Optional features](#optional-features)
- [Metrics (with formulas)](#metrics-with-formulas)
- [Troubleshooting](#troubleshooting)
- [Minimal checklist (recommended workflow)](#minimal-checklist-recommended-workflow)


---

## Repository layout

```text
configs/          # YAML configs 
data/
  prompts/        # prompt templates (.txt): instructions + codebook given to the LLM
  text/           # text items (.json): the documents/excerpts to be coded
  labels/         # label sets (.txt): your label universe
  validation/     # validation datasets (.csv): either ground-truth labels OR multi-rater coding
outputs/          # auto-generated logs and results (grouped by project_name)
src/              # pipeline source code
.env              # your local environment variables (e.g., OPENAI_API_KEY);
.env.example      # example template 
requirements.txt  # needed installations
run.py            # convenience entrypoint (wraps the main pipeline)
```

---

## What the pipeline does

At a high level, the pipeline performs four steps:

1. Load inputs  
   - label universe from a `.txt` file (list of the allowed labels for a task)  
   - prompt template from a `.txt` file (instruction for the LLM)  
   - text items from a directory of `.json` files (documents you want to classify)  
   - validation data from a `.csv` file (human coders results or expert classification)

2. Call the LLM to code items (unless you run in evaluation-only mode)  
   - batching is controlled by `batch_size` (how many papers should be given at a single LLM call)  
   - strict output format is enforced (JSON array with id and labels, to ensure save interpretation of Llm Coding results)

3. Evaluate results  
   - Groundtruth mode if the validation CSV contains a single `labels` column  
   - ICR mode otherwise (≥1 human rater columns)

4. Write outputs  
   - per-run artifacts into (all the information , raw data and the calculated evaluation metrics can be found here) `outputs/<project_name>/results/run_<timestamp>/...`  
   - a cumulative `run_history.csv` for cross-run comparison (for every new started project there will be a new run history started)

---

## Requirements

- **Python 3.10+**
- **OpenAI API key**  
  The pipeline is currently optimised for OpenAI’s API (configured via `OPENAI_API_KEY`).  
  Support for additional LLM providers may be added in future versions. Advanced users can adapt the modular client component to already integrate other providers.

---

## Installation

Install dependencies:

```bash id="btxvkh"
pip install -r requirements.txt
```

Create a `.env` file containing your OpenAI API key:

```bash id="17zuwj"
cp .env.example .env
```

Edit `.env` and set:

```text id="il38lu"
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Quick start (included examples)

This repository already includes two example YAML configs in `configs/`:

- `example_singlelabel_groundtruth.yaml` (single-label + ground-truth evaluation)
- `example_multilabel.yaml` (multi-label + ICR evaluation)

From the repository root (i.e., the directory containing `run.py`), run:

```bash id="dam42o"
python run.py configs/example_singlelabel_groundtruth.yaml
```

or:

```bash id="2ytrlk"
python run.py configs/example_multilabel.yaml
```

---

## Preparing your inputs

This section is intentionally detailed: most runtime errors come from small input-format mistakes.

### 1) Label universe

File: `data/labels/<your_labels>.txt`  
Format: one label per line (exact spelling is important)

Example:

```text id="u2w9dx"
economic_strain
care_responsibilities
mental_wellbeing
trust_in_institutions
```

Rules enforced by the pipeline:

- Duplicate lines are removed (order preserved). 
- `"n/a"` is automatically appended if missing as an label in the label universe. 
- Model outputs and validation labels are validated against this universe; unknown labels raise an error.

### 2) Prompt template

File: `data/prompts/<your_prompt>.txt`  
Format: Plain-English instructions plus an optional codebook.

A practical prompt could include:

- a concise definition of the task
- inclusion/exclusion criteria per label
- guidance for ambiguous cases
- explicit guidance on when to choose `"n/a"`
- short, correctly coded examples (few-shot prompting)

#### What the pipeline appends to your prompt (default model instructions)

You do not need to specify output-formatting rules yourself. At runtime, the pipeline appends a strict JSON contract to every batch prompt. This keeps your human-readable prompt clean while enforcing machine-checkable outputs. 

```text id="hxumws"
Output only a single valid JSON array (from `[` to `]`) and nothing else (no markdown, no prose).

Each element must be {"id": "<id>", "labels": ["<label>", ...]}.
# If enabled by configuration, each element may also include:
# - "evidence_lines": ["<verbatim quote from input>", ...]
# - "explanation": "<brief rationale for label choice>"

The batch has <BATCH_SIZE> items. Output exactly <BATCH_SIZE> objects, one per id.
Include every id exactly once (no missing / extra ids).

<Single-label mode: labels must contain exactly 1 label.>
<Multi-label mode: labels may contain multiple allowed labels.>

Each label must be copied verbatim from the Allowed labels list; do not rephrase.

# If evidence_lines is enabled:
# evidence_lines must contain short verbatim quotes copied exactly from the input text that justify the chosen label(s).

# If explanation is enabled:
# explanation must be a brief rationale explaining why you selected the label(s) for this item, grounded in the provided input.

# If multi-label mode is enabled:
# Only output labels: ["n/a"] if no other allowed label applies; do not combine "n/a" with any other label.

If none of the allowed labels fit the given text, output labels: ["n/a"].

Allowed labels (use exactly these spellings):
- <LABEL_1>
- <LABEL_2>
- ...
- <LABEL_N>

Examples (placeholders only; copy the structure, not the placeholder content):
- Single-label structure: [{"id":"<ID_FROM_BATCH>","labels":["<ALLOWED_LABEL>"]}]
- Multi-label structure:  [{"id":"<ID_FROM_BATCH>","labels":["<ALLOWED_LABEL_1>","<ALLOWED_LABEL_2>"]}]
- Multi-item batch structure:
  [{"id":"<ID_1_FROM_BATCH>","labels":["<ALLOWED_LABEL>"]},{"id":"<ID_2_FROM_BATCH>","labels":["<ALLOWED_LABEL>"]}]
```

Finally, the pipeline appends the INPUT payload as a JSON array of items. Each item includes id, and either text or seperated sections (so it may include title ecetera if present in your project).

### 3) Text items (JSON)

Directory: `data/text/<your_text_dir>/`  
Format: one `.json` file per item.

Minimum required schema:

```json id="ug2mjb"
{
  "id": "optional-string-id",
  "text": "the excerpt to code"
}
```

Normalization performed by the loader:
- If `"id"` is missing, it defaults to the filename stem (filename without `.json`).
- If `"text"` is missing or null, it becomes `""`.

Optional fields (supported):
- `"title": "..."` (string or null)
- `"sections": { "Section A": "...", "Section B": "..." }`

If sections exists and is non-empty, the pipeline sends sections to the model; otherwise it sends text.

Example (file without id in the JSON):  
Suppose you have a file named: `data/text/interview_07.json`  
with the following contents:

```json id="bv7d2t"
{
  "title": "Interview excerpt",
  "text": "I’m juggling two jobs and still worried about bills."
}
```

Because `"id"` is missing, the loader assigns:
- `id = "interview_07"` (from the filename stem)

So the item is treated internally as if it were:

```json id="hmhxle"
{
  "id": "interview_07",
  "title": "Interview excerpt",
  "text": "I’m juggling two jobs and still worried about bills."
}
```

---

## 4) Validation file (CSV)

A CSV (“comma-separated values”) file is a plain-text table format. You can create it in:
- Excel / LibreOffice / Numbers → Save As / Export → CSV
- Google Sheets → File → Download → Comma-separated values (.csv)

Export advice (recommended):
- Prefer comma-delimited CSV (,) because the pipeline loads CSVs using standard defaults.
- Prefer UTF-8 encoding.
- If your system exports semicolon-delimited CSV (;), re-export with commas (or convert delimiters) before running.

There are two supported validation layouts:

### A) Ground-truth CSV (explicit expert labels)

Ground-truth evaluation is used when (and only when) your CSV contains a column named `labels`.

Required columns:
- `id` (or `ID`)
- `labels`

Single-label ground truth (one label per row)

```csv id="vmmnht"
id,labels
001,economic_strain
002,trust_in_institutions
```

Multi-label ground truth (labels stored as a JSON array string)

```csv id="pwj49k"
id,labels
001,"[""economic_strain"",""mental_wellbeing""]"
002,"[""n/a""]"
```

How to enter multi-label cells safely:
- In your spreadsheet cell, enter valid JSON such as:
  - `["economic_strain","mental_wellbeing"]`
- When exported to CSV, many tools escape quotes and you may see:
  - `"[""economic_strain"",""mental_wellbeing""]"`

This is normal CSV escaping and is supported by the parser.

Parsing rules enforced by the pipeline:
- labels may be a plain string (e.g., `economic_strain`) or a JSON array string (e.g., `["economic_strain","mental_wellbeing"]`).
- Empty or missing cells are parsed as `[]`.
- If `allow_multilabel: false`, any cell containing more than one label raises an error.
- All labels are validated against the configured label universe; unknown labels raise an error.

### B) ICR (rater-style) CSV (no explicit ground-truth labels)

ICR mode is used when the CSV does not contain a `labels` column and instead contains one or more rater columns (with arbitrary names).

Required columns:
- `id` (or `ID`)
- at least one rater column (any name is acceptable), e.g. `rater_1`

Example:

```csv id="a4w8n0"
id,rater_1,rater_2
001,economic_strain,"[""economic_strain"",""mental_wellbeing""]"
002,trust_in_institutions,trust_in_institutions
```

Key points:
- ICR mode works with one rater column: reliability is then computed between that rater and the LLM (because the pipeline adds the LLM as an additional coder). “Humans-only” reliability requires ≥ 2 human raters.
- Rater cells may contain single labels or JSON arrays (multi-label is permitted in ICR input).
- Missing cells are treated as missing coder decisions: blank/NaN rater entries are excluded from agreement calculations, and any item must have at least two non-missing coders (within the coder subset being evaluated) to contribute to Krippendorff’s α.

### How evaluation mode is inferred

This rule is central:
- If your validation CSV contains a `labels` column → ground-truth evaluation.
- Otherwise, if it contains ≥ 1 rater columns besides `id` → ICR evaluation.

In practice, adding or removing the column name `labels` is the switch between ground-truth mode and ICR mode.

---

## Configuration (YAML)

The pipeline is configured through a single YAML file with a strict schema (only documented keys are accepted; unknown keys raise an error).

### Required keys

- `project_name`  
  Name of the run group; outputs are written under `outputs/<project_name>/{logs,results}`.

**Inputs**
- `prompt_path`  
  Path to the prompt template (.txt). If given as a simple relative filename, it is resolved under `data/prompts/`.
- `text_dir`  
  Directory containing input items (.json files). If relative, it is resolved under `data/text/`.
- `hand_coded_validation_path`  
  Validation CSV path (ground-truth or rater-style). If relative, it is resolved under `data/validation/`.
- `labels_txt_path`  
  Label universe file (.txt, one label per line). If relative, it is resolved under `data/labels/`.

**Model / runtime**
- `llm`  
  The OpenAI model identifier used for coding.
- `batch_size`  
  Number of items included in a single LLM call.
- `max_retries`  
  Maximum number of retries for a failed batch call (robust execution).

**Task policy**
- `allow_multilabel`  
  `true` = multiple labels per item allowed; `false` = exactly one label per item (enforced in parsing and evaluation).

### Optional keys

- `temperature` (default: 0.0)  
  Sampling temperature for the model call.
- `model_role` (default: "")  
  Optional system message (role) prepended to the chat.
- `include_evidence_lines` (default: false)  
  Requests evidence_lines in the model output (verbatim quotes supporting the assigned label(s)).
- `include_explanation` (default: false)  
  Requests a brief explanation field in the model output.
- `hallucination_check` (default: false)  
  Runs an evidence-line audit that flags quoted evidence not found in the original text.
- `canonical_raw_path` (default: null)  
  Evaluation-only mode: skips LLM calls and instead reads predictions from an existing canonical JSONL file (typically a previous run’s raw_output.jsonl).

Example config:

```yaml id="5y2aed"
project_name: my_run

# Inputs (these relative paths resolve under data/prompts, data/text, data/validation, data/labels by default)
prompt_path: my_prompt.txt
text_dir: my_texts
hand_coded_validation_path: my_validation.csv
labels_txt_path: my_labels.txt

# Model / runtime
llm: gpt-4.1-nano
temperature: 0.0
model_role: "You are a careful qualitative coder."

batch_size: 5
max_retries: 2

# Task policy
allow_multilabel: true

# Optional output fields / checks
include_evidence_lines: false
include_explanation: false
hallucination_check: false

# Evaluation-only mode (uncomment to reuse an existing run’s predictions)
# canonical_raw_path: outputs/my_run/results/run_YYYY-MM-DDTHH-MM-SS/raw_output.jsonl
```

---

## Path resolution

If you use relative paths, they default to:
- `prompt_path` → `data/prompts/`
- `text_dir` → `data/text/`
- `hand_coded_validation_path` → `data/validation/`
- `labels_txt_path` → `data/labels/`

So this:

```yaml id="h1u279"
prompt_path: my_prompt.txt
```

means:

`data/prompts/my_prompt.txt`

Paths starting with `data/`, `outputs/`, `configs/`, or `src/` are treated as repo-root relative.

---

## Running the pipeline

Run with:

```bash id="20masa"
python run.py configs/<your_config>.yaml
```

---

## Outputs

### Run directory

Each run writes to:

`outputs/<project_name>/results/run_<timestamp>/`

Typical files include:

- **`raw_output.jsonl`** — canonical predictions (JSONL; one JSON object per line), e.g.  
  `{"id":"001","labels":["economic_strain"]}`  
  If enabled, items may additionally include:
  - `evidence_lines: [...]`
  - `explanation: "..."`

- **`raw_input.txt`** — the exact prompt text sent to the model *per batch*, including your prompt template, the appended JSON contract, and the batch payload. This is the primary audit/debug artifact.

- **`crosstab_log.csv`** — written only for **single-label ground-truth runs**; a confusion-matrix-style cross-tabulation of ground-truth vs. predicted labels.

- **`hallucinations.log`** — written only when `hallucination_check: true`; reports evidence lines that do not occur in the original input text.

### Logs

Logs (including the active configuration and automatically computed metrics) are written to:

`outputs/<project_name>/logs/<timestamp>.log`

### Run history

`run_history.csv` is stored at:

`outputs/<project_name>/results/run_history.csv`

It appends **one row per run**, containing:

- key overall metrics (ground-truth classification metrics or ICR Krippendorff’s α metrics)
- the most relevant configuration parameters (e.g., model, role, batch size, evidence/explanation flags)
- a timestamp for the run

The overview enabled by Run History enables a practical “experiment loop”:

- adjust prompt/codebook → rerun → compare results  
- swap models → rerun → compare results  
- change label sets → rerun → compare results  
- enable/disable evidence and hallucination checks → rerun → compare results  

---

## Optional features

### Evidence lines

If `include_evidence_lines: true`, the model is required to return:

```json id="3ofe37"
"evidence_lines": ["<verbatim quote from input>", "..."]
```

Intended use:
- improve auditability (“what in the text supports this code?”)
- support qualitative review and coder training

The pipeline validates formatting and (optionally) checks literal presence (see hallucination check).

### Explanation

If `include_explanation: true`, the model is required to return:

```json id="rgs1vd"
"explanation": "<brief rationale>"
```

This is a short justification, useful for interpretability and manual review (of the codes selected by the LLM).

### Hallucination check

If `hallucination_check: true` the pipeline verifies that every `evidence_lines` entry appears verbatim in the source text after:
- Unicode normalization (NFKC),
- lowercasing,
- whitespace collapsing.

Mismatches are counted and written to `hallucinations.log`. This check does not change predictions or metrics; it is purely an audit signal.

### Evaluation-only mode (no API calls)

If you already have predictions and want to recompute metrics:

```yaml id="9p4ulf"
canonical_raw_path: outputs/<project_name>/results/run_.../raw_output.jsonl
```

Then run as usual:

```bash id="y73qs9"
python run.py configs/<your_config>.yaml
```

The pipeline reads `raw_output.jsonl` and re-evaluates it. This is useful for:
- changing the validation CSV
- re-running ICR calculations
- re-computing metrics after modifying the label universe
- regenerating artifacts like the single-label crosstab

---

## Metrics (with formulas)

GitHub can render LaTeX-style mathematics directly in `README.md`. Use inline math with `$...$` and block math with `$$...$$` or a fenced `math` block (recommended for readability).

### Single-label (ground-truth mode)

Let there be $N$ items and a label set $\mathcal{L}$. Each item has exactly one ground-truth label $y_i\in\mathcal{L}$ and one predicted label $\hat{y}_i\in\mathcal{L}$.

**Accuracy**

```math id="7e50ds"
\text{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[y_i=\hat{y}_i]
```

For each label $\ell\in\mathcal{L}$, define:

- $TP_{\ell}$: items where $y_i=\ell$ and $\hat{y}_i=\ell$
- $FP_{\ell}$: items where $y_i
eq\ell$ and $\hat{y}_i=\ell$
- $FN_{\ell}$: items where $y_i=\ell$ and $\hat{y}_i
eq\ell$

**Precision / Recall / F1**

```math id="o3tkd6"
P_{\ell}=\frac{TP_{\ell}}{TP_{\ell}+FP_{\ell}},\quad
R_{\ell}=\frac{TP_{\ell}}{TP_{\ell}+FN_{\ell}},\quad
F1_{\ell}=\frac{2P_{\ell}R_{\ell}}{P_{\ell}+R_{\ell}}
```

**Macro-F1**

```math id="b2k2nh"
\text{Macro-F1}=\frac{1}{|\mathcal{L}|}\sum_{\ell\in\mathcal{L}} F1_{\ell}
```

**Weighted-F1** (support-weighted by ground-truth frequency)

```math id="9jti3h"
\text{Weighted-F1}=\sum_{\ell\in\mathcal{L}}\frac{\text{support}_{\ell}}{N}\,F1_{\ell},
\qquad \text{support}_{\ell}=TP_{\ell}+FN_{\ell}
```

---

### Multi-label (ground-truth mode)

Each item $i$ has a ground-truth label set $G_i\subseteq\mathcal{L}$ and a predicted label set $\hat{G}_i\subseteq\mathcal{L}$.

**Subset accuracy** (exact set match)

```math id="whgqmq"
\text{SubsetAcc}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[G_i=\hat{G}_i]
```

Define micro counts aggregated across items:

```math id="kvwivz"
TP=\sum_{i=1}^{N}|G_i\cap \hat{G}_i|,\quad
FP=\sum_{i=1}^{N}|\hat{G}_i\setminus G_i|,\quad
FN=\sum_{i=1}^{N}|G_i\setminus \hat{G}_i|
```

**Micro precision / recall / F1**

```math id="qewxni"
P_{\text{micro}}=\frac{TP}{TP+FP},\quad
R_{\text{micro}}=\frac{TP}{TP+FN},\quad
F1_{\text{micro}}=\frac{2P_{\text{micro}}R_{\text{micro}}}{P_{\text{micro}}+R_{\text{micro}}}
```

For macro metrics, compute per-label $TP_{\ell},FP_{\ell},FN_{\ell}$ by treating each label as a binary decision (“present” vs “absent”), then:

```math id="gz1ldz"
P_{\text{macro}}=\frac{1}{|\mathcal{L}|}\sum_{\ell\in\mathcal{L}} P_{\ell},\quad
R_{\text{macro}}=\frac{1}{|\mathcal{L}|}\sum_{\ell\in\mathcal{L}} R_{\ell},\quad
F1_{\text{macro}}=\frac{1}{|\mathcal{L}|}\sum_{\ell\in\mathcal{L}} F1_{\ell}
```

---

### ICR (Krippendorff’s α)

Krippendorff’s alpha is reported in several variants. The general form is:

$$
\alpha = 1 - \frac{D_o}{D_e}
$$

- $D_o$: observed disagreement  
- $D_e$: expected disagreement by chance  

Interpretation:

- $\alpha=1$: perfect agreement  
- $\alpha=0$: chance-level agreement  
- $\alpha<0$: systematic disagreement  

#### 1) Atomic nominal α (exact set token match)

Each coder’s label set is converted into a canonical token:

- sort unique labels
- join with `|`
- empty set becomes `<EMPTY>`

Example: `["b","a","a"] → "a|b"`

Nominal $\alpha$ then follows the standard observed/expected disagreement formulation using category counts per item and global marginals.

#### 2) Binary-incidence α (per-label presence/absence)

For each label $\ell$, each coder decision is converted into a binary value:

- 1 if $\ell$ is present
- 0 if $\ell$ is absent
- missing remains missing

Compute $\alpha_{\ell}$ per label and report:

**Macro binary-incidence α**

```math id="zr7qtk"
\alpha_{\text{macro}}=\frac{1}{|\mathcal{L}|}\sum_{\ell\in\mathcal{L}}\alpha_{\ell}
```

**Prevalence-weighted binary-incidence α**

Let $w_{\ell}$ be the number of positive assignments for label $\ell$ across all items and coders:

```math id="ps8wol"
\alpha_{\text{pw}}=\frac{\sum_{\ell\in\mathcal{L}} w_{\ell}\,\alpha_{\ell}}{\sum_{\ell\in\mathcal{L}} w_{\ell}}
```

#### 3) Jaccard set-based α (partial overlap)

This variant treats decisions as sets and uses Jaccard distance:

```math id="s0cgab"
\delta(S,T)=1-\frac{|S\cap T|}{|S\cup T|}
```

Krippendorff’s $\alpha$ is computed using the general metric form with $\delta$ for observed and expected disagreement over coder pairs.

---

## Troubleshooting

- **The run is evaluated as ICR, but you expected ground-truth evaluation**
  - Ensure the validation CSV contains a column named exactly `labels`.

- **The run is evaluated as ground-truth, but you expected ICR**
  - Remove or rename the `labels` column and use one or more rater columns instead.

- **The CSV loads as a single column**
  - Your CSV is likely semicolon-delimited (`;`). Re-export as a comma-delimited CSV (`,`), ideally with UTF-8 encoding.

- **Unknown label errors**
  - Ensure every label appearing in the validation CSV is present in your label universe file (`labels_txt_path`) and matches spelling exactly.

- **No overlapping ids**
  - Ensure the `id` values in your text JSON files match the `id` column in the validation CSV (or omit `id` in the JSON files and rely on filename stems).

- **Model output parsing / validation errors**
  - Keep `temperature: 0.0`, reduce `batch_size`, and increase `max_retries`. The pipeline requires a single valid JSON array with exactly one object per input id.

---

## Minimal checklist (recommended workflow)

1. Define labels in `data/labels/`.
2. Write a prompt in `data/prompts/`.
3. Add text JSON files in `data/text/`.
4. Add a validation CSV in `data/validation/` (ground-truth or rater-style).
5. Create a YAML config in `configs/`.
6. Run: `python run.py configs/<your_config>.yaml`
7. Inspect `outputs/<project_name>/logs/` and `outputs/<project_name>/results/`.
