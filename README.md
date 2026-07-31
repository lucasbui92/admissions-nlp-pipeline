
## Data Preparation

Run these once before using the pipeline.

**1. Build course descriptions** — populates the `combined_description` column in `course_descriptions.xlsx`:

```
python -m prep.build_course_descriptions
```

**2. Populate subject indices** — looks up each course title in `course_mappings.xlsx` and writes the corresponding subject index into the `subject` column of your input file. Modifies the file in-place.

- **Sample mode:** pass `--input data/sample/sample_personal_statements.xlsx` to populate the built-in sample file, or pass `--input <path>` to populate your own file (which can also be placed inside `data/sample/`).
- **Restricted mode:** `--input <path>` is required as the production dataset must be explicitly specified.

```
python prep/add_subject_index.py --input <path>
```

Both scripts must be run from the project root.

## Running the Pipeline

```bash
python run.py --mode <mode> --output_name <name> [--input <path>] [--output_path <dir>] [--metric <metric>] [--include_matches] [--stage <stage>] [--export_unassigned]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--mode` | Yes | `sample` (for external use) or `restricted` (internal only — requires access to the production dataset) |
| `--output_name` | Yes | Label for the output folder (e.g. `trial1`) |
| `--input` | Only in `restricted` mode | Path to your `.xlsx` input file |
| `--output_path` | Only in `sample` mode | Directory where CSV/Excel outputs are written |
| `--metric` | No | Single metric to compute: `chunk_semantic`, `doc_semantic`, `grammar`, `readability`, `topic_modelling`. Defaults to all metrics when omitted. |
| `--include_matches` | No | Only valid with `--metric grammar` — adds per-rule grammar match details to the Excel export |
| `--stage` | Only when `--metric topic_modelling` | `candidates` or `scoring` — see [Topic Modelling](#topic-modelling) |
| `--export_unassigned` | No | Only valid with `--stage scoring`. Also exports a CSV of sentences that matched no topic — see [Topic Modelling](#topic-modelling) |

### Modes

**`sample`** — Uses the built-in file at `data/sample/sample_personal_statements.xlsx`. No `--input` needed, but `--output_path` is required for CSV/Excel outputs.

```bash
python run.py --mode sample --output_name trial1 --output_path C:\path\to\output
```

**`restricted`** — For internal use only. Requires access to the production dataset, which is not publicly available. If you are an external user, use `sample` mode instead.

### Output

Results are saved to `output/<mode>/<output_name>_<YYYYMMDD>/`:

| File | Contents |
|---|---|
| `grammar.json` | Grammar scoring records |
| `readability.json` | Readability scoring records |
| `doc_semantic.json` | Document-level semantic alignment records |
| `chunk_semantic.json` | Chunk-level semantic alignment records |
| `<output_name>.xlsx` | Combined Excel report |

Only the files for the computed metrics are written. If `--metric` is used, only the relevant JSON file(s) are produced.

The Excel report is written to `--output_path`.

`topic_modelling` is the exception — it doesn't write JSON or feed the Excel report. It writes CSVs to `--output_path` instead; see [Topic Modelling](#topic-modelling).

### Optional: Grammar Match Details

Add `--include_matches` to include per-rule grammar match details as extra columns in the Excel export. This only affects the grammar metric — it has no effect on any other metric:

```bash
python run.py --mode sample --output_name trial1 --include_matches
```

## Topic Modelling

Topic modelling is one of the metrics computed by the pipeline. It scores each personal statement against a fixed set of topics using seed-keyword matching, not clustering — every statement is scored against every topic, none are discovered dynamically.

Unlike the other metrics, running it (`--metric topic_modelling`) requires `--stage candidates` or `--stage scoring`.

### Stage 1: `candidates`

```bash
python run.py --mode sample --output_name trial1 --metric topic_modelling --stage candidates
```

Scans statements for sentences containing a seed keyword (from `data/reference/topics_keywords_seed.txt`), then extracts and ranks the other words/phrases in those sentences as candidate keywords for each topic. Writes a CSV (`topic`, `rank`, `keyword`, `frequency`) to `<output_path>\<output_name>_<YYYYMMDD>.csv`. Review it and manually curate the approved keywords into `data/reference/topics_keywords_final.txt` (same `Topic: keyword, keyword, ...` format as the seed file) before running the scoring stage.

> **Note:** the candidates CSV and the scoring CSV (below) resolve to the same filename pattern. Use a different `--output_name` for each stage, or move/rename the candidates CSV after review, so scoring doesn't overwrite it.

### Stage 2: `scoring`

```bash
python run.py --mode sample --output_name trial1 --metric topic_modelling --stage scoring [--export_unassigned]
```

Requires `data/reference/topics_keywords_final.txt` to exist (fails otherwise). Tokenizes each statement into sentences, then scores each sentence against every topic in two phases:

1. **Lemma matching** — counts lemmatized keyword/phrase matches per topic.
2. **Semantic fallback** — for sentences that score below threshold in phase 1, embeds the sentence and adds a fractional score for any topic whose keywords are similar enough (cosine similarity).

Each sentence is assigned to its highest-scoring topic(s) (ties split evenly); a statement's final score per topic is the proportion of its sentences assigned to that topic. Writes a CSV of per-statement topic proportions to `<output_path>\<output_name>_<YYYYMMDD>.csv` — one column per topic, values between 0 and 1.

Sentence tokenization is cached in `data/cache/` and reused on subsequent runs against the same input file.

`--export_unassigned` additionally writes a diagnostic CSV of sentences that matched no topic at all, to `<output_path>\<output_name>_<YYYYMMDD>_unassigned_sentences.csv`. This is a one-time diagnostic, not meant for every run.
