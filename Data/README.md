# NER Training Data

This directory contains all training, evaluation, and reference data for the Mongolian NER model (`Nomio4640/ner-mongolian` on HuggingFace Hub).

**None of these files are used at runtime by the web app.** They are only used for model training and evaluation.

## Labeling Formats

Two different labeling formats are used across this project. Both produce identical BERT token inputs after tokenization — the format does not affect model quality.

### CoNLL (BIO tags) — used in `data/`

Token-level format. One word per line with BIO label, sentences separated by blank lines.

```
Батболд O O B-PER
гишүүн  O O O

Улаанбаатар O O B-LOC
хотод       O O O
```

> `.txt` and `.conll` file extensions are both CoNLL format — there is no technical difference.

### JSONL (character offsets) — used in `datav2/`

Sentence-level format. One JSON object per line with character-position labels.

```json
{"text": "Батболд гишүүн", "labels": [[0, 7, "PER"]]}
{"text": "Улаанбаатар хотод", "labels": [[0, 10, "LOC"]]}
```

> `.json` = one big JSON array/object wrapping everything. `.jsonl` = one independent JSON object per line (easier to stream and append).

### Why two formats?

- **CoNLL** was used for v1 (manual annotation + silver labeling). Standard format for NER datasets.
- **JSONL** was used for v2 (synthetic data generation). Character offsets are easier to produce programmatically — no need to pre-tokenize.
- Both get converted to BERT subword token labels during training. The Colab training code has separate tokenizers for each format (`tokenize_conll_data()` and `tokenize_json_data()`), but the model sees the same tensors.

## Shared Validation & Test Sets

`data/valid.txt` and `data/test.txt` are used by **both** v1 and v2 training pipelines. This is intentional — using the same evaluation set allows fair comparison of model performance across different training approaches.

## Directory Structure

### `data/` — CoNLL Training Data (v1 Pipeline)

| File | Size | Description |
|------|------|-------------|
| `train.txt` | 2.5MB | Original manually-annotated gold training data |
| `train_final.txt` | 6.0MB | **Final training file** — train.txt + auto-labeled silver data with label fixes. This is what gets uploaded to Colab |
| `valid.txt` | 275KB | Validation split (shared across v1 and v2, used for early stopping) |
| `test.txt` | 307KB | Test split (shared across v1 and v2, used by `eval/` scripts) |

### `datav2/` — JSONL Training Data (v2 Pipeline)

| File | Description |
|------|-------------|
| `generate_training_data.py` | **Data generation script** (run locally). Reads NER-dataset/ reference files and produces synthetic training sentences with Mongolian case suffixes, politician names, companies, etc. |
| `training_v2_cells.py` | **Model training code** (copy-paste into Google Colab cells). Loads train_v2_merged.jsonl for training and data/valid.txt for validation. Handles character-offset to BERT subword alignment |
| `train_v2_merged.jsonl` | **Final v2 training file** (20,696 sentences). All generated data merged and shuffled |

Intermediate per-entity JSONL files (per_names.jsonl, org_*.jsonl, etc.) are gitignored. Regenerate with:
```bash
cd Data/datav2 && python generate_training_data.py
```

### `NER-dataset/` — Reference Data for Data Generation

Source datasets used by `generate_training_data.py` to create synthetic training examples. Not used directly for training or at runtime.

| File | Description |
|------|-------------|
| `NER_v1.0.json.gz` | Base NER dataset (10,162 sentences, gzip-compressed JSON). `.gz` = gzip compression — Python reads directly with `gzip.open()` |
| `locations.json` | All Mongolian administrative locations (279 entries: аймаг, сум, дүүрэг, хороо, хороолол) with parent hierarchy and coordinates |
| `districts.csv` | Flat list of 353 sums/districts with parent aimag. Supplements locations.json with 280 additional sums |
| `mongolian_abbreviations.csv` | Organization abbreviations (526 entries, e.g. МҮОХ → Монголын Үндэсний Олимпийн Хороо) |
| `countries.csv` | Country names for LOC entity generation |
| `mongolian_news_demo.csv` | Demo news articles (500 rows) for testing |

Large compressed files (mongolian_personal_names.csv.gz, mongolian_company_names.csv.gz, mongolian_clan_names.csv.gz) are gitignored — keep locally if needed for regenerating datav2.

## Training Pipelines

### v1 (CoNLL)
```
train.txt (manual annotation)
    +
silver data (auto-labeled by model, high confidence)
    |
    v  [merge_train.py → fix_labels.py]
train_final.txt
    |
    v  Upload to Google Colab
    |
    v  Fine-tune BERT model
    |
    v  Push to HuggingFace Hub: Nomio4640/ner-mongolian
```

### v2 (JSONL)
```
NER-dataset/ reference data (names, locations, abbreviations)
    |
    v  [generate_training_data.py] (run locally)
    |
intermediate JSONL files (gitignored, regenerable)
    |
    v  [merged + shuffled]
train_v2_merged.jsonl
    |
    v  Upload to Google Colab
    |
    v  [training_v2_cells.py] Fine-tune BERT model
    |
    v  Push to HuggingFace Hub: Nomio4640/ner-mongolian
```

Both pipelines use `data/valid.txt` for validation and `data/test.txt` for evaluation.

## How to Regenerate Files

```bash
# Regenerate v2 intermediate JSONL files from reference data
cd Data/datav2 && python generate_training_data.py

# v1 intermediate files (merge_train.py, fix_labels.py) were removed from repo
# but can be recovered from git history if needed:
# git log --all --oneline -- scripts/
```
