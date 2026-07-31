from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import yaml

with open(Path("config") / "settings.yml") as f:
    MODE_SETTINGS = yaml.safe_load(f)["modes"]

DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("output")

COURSES_FILE = DATA_ROOT / "reference" / "course_descriptions.xlsx"
COURSE_MAPPINGS_FILE = DATA_ROOT / "reference" / "course_mappings.xlsx"
KEYWORDS_DIR = DATA_ROOT / "keywords"
TOPIC_KEYWORDS_FILE = KEYWORDS_DIR / "topics_keywords_seed.txt"
TOPIC_KEYWORDS_FINAL_FILE = KEYWORDS_DIR / "topics_keywords_final.txt"
EXCLUDED_WORDS_FILE = KEYWORDS_DIR / "excluded_words.txt"

VALID_MODES = {"sample", "restricted"}


@dataclass
class Paths:
    input_file: Path
    reference_file: Path
    output_dir: Path
    grammar_output_file: Path
    readability_output_file: Path
    doc_semantic_output_file: Path
    chunk_semantic_output_file: Path
    topic_scoring_csv: Path
    topic_unassigned_csv: Path
    sentences_tokenized_pkl: Path
    topic_candidates_file: Path
    data_source_type: str
    derived_dir: Path


def resolve_paths(mode, output_name):
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")

    mode_cfg = MODE_SETTINGS.get(mode, {})
    input_path = mode_cfg.get("input_path")
    input_name = mode_cfg.get("input_name")
    output_path = mode_cfg.get("output_path")

    if not input_path or not input_name:
        raise ValueError(
            f"settings.yml is missing input_path/input_name for mode '{mode}'. "
            f"Fill these in under modes.{mode} before running."
        )
    if not output_path:
        raise ValueError(
            f"settings.yml is missing output_path for mode '{mode}'. "
            f"Fill this in under modes.{mode} before running."
        )

    input_file = Path(input_path) / input_name
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    derived_dir = Path(output_path)

    data_source_type = mode

    if not COURSES_FILE.exists():
        raise FileNotFoundError(f"Reference file not found: {COURSES_FILE}")

    today = datetime.now().strftime("%Y%m%d")
    output_dir = OUTPUT_ROOT / mode / f"{output_name}_{today}"

    return Paths(
        input_file=input_file,
        reference_file=COURSES_FILE,
        output_dir=output_dir,
        grammar_output_file=output_dir / "grammar.json",
        readability_output_file=output_dir / "readability.json",
        doc_semantic_output_file=output_dir / "doc_semantic.json",
        chunk_semantic_output_file=output_dir / "chunk_semantic.json",
        topic_scoring_csv=derived_dir / f"{output_name}_{today}.csv",
        topic_unassigned_csv=derived_dir / f"{output_name}_{today}_unassigned_sentences.csv",
        sentences_tokenized_pkl=input_file.parent / f"{input_file.stem}_sentences_tokenized.pkl",
        topic_candidates_file=derived_dir / f"{output_name}_{today}.csv",
        data_source_type=data_source_type,
        derived_dir=derived_dir,
    )
