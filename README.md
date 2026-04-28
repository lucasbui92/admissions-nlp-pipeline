
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
python run.py --mode <mode> --output_name <name> [--input <path>] [--include_matches]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--mode` | Yes | `sample` (for external use) or `restricted` (internal only — requires access to the production dataset) |
| `--output_name` | Yes | Label for the output folder (e.g. `trial1`) |
| `--input` | Only in `restricted` mode | Path to your `.xlsx` input file |
| `--metric` | No | Single metric to compute: `chunk_semantic`, `doc_semantic`, `grammar`, `readability`. Defaults to all metrics when omitted. |
| `--include_matches` | No | Adds grammar match details to the Excel export |

### Modes

**`sample`** — Uses the built-in file at `data/sample/sample_personal_statements.xlsx`. No `--input` needed.

```bash
python run.py --mode sample --output_name trial1
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

### Optional: Grammar Match Details

Add `--include_matches` to include per-rule grammar match details as extra columns in the Excel export:

```bash
python run.py --mode sample --output_name trial1 --include_matches
```
