from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("output")
DERIVED_DIR = Path(r"B:\derived")
RAW_DIR = Path(r"B:\raw")
CACHE_DIR = DATA_ROOT / "cache"

SAMPLE_INPUT_FILE = DATA_ROOT / "sample" / "sample_personal_statements.xlsx"
COURSES_FILE = DATA_ROOT / "reference" / "course_descriptions.xlsx"
COURSE_MAPPINGS_FILE = DATA_ROOT / "reference" / "course_mappings.xlsx"
TOPIC_KEYWORDS_FILE = DATA_ROOT / "reference" / "topics_keywords_seed.txt"
TOPIC_KEYWORDS_FINAL_FILE = DATA_ROOT / "reference" / "topics_keywords_final.txt"

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
    topic_embeddings_cache: Path
    sentences_tokenized_pkl: Path
    topic_candidates_file: Path
    data_source_type: str


def resolve_paths(mode, external_input, output_name):
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")

    if mode == "sample":
        input_file = SAMPLE_INPUT_FILE
        data_source_type = "sample"
    else:
        if not external_input:
            raise ValueError("Restricted mode requires --input")
        input_file = Path(external_input)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        data_source_type = "restricted"

    if not COURSES_FILE.exists():
        raise FileNotFoundError(f"Reference file not found: {COURSES_FILE}")

    today = datetime.now().strftime("%Y%m%d")
    output_dir = OUTPUT_ROOT / mode / f"{output_name}_{today}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return Paths(
        input_file=input_file,
        reference_file=COURSES_FILE,
        output_dir=output_dir,
        grammar_output_file=output_dir / "grammar.json",
        readability_output_file=output_dir / "readability.json",
        doc_semantic_output_file=output_dir / "doc_semantic.json",
        chunk_semantic_output_file=output_dir / "chunk_semantic.json",
        topic_scoring_csv=DERIVED_DIR / f"{output_name}_{today}.csv",
        topic_embeddings_cache=CACHE_DIR / f"{input_file.stem}_topic_embeddings.npy",
        sentences_tokenized_pkl=(RAW_DIR if mode == "restricted" else CACHE_DIR) / f"{input_file.stem}_sentences_tokenized.pkl",
        topic_candidates_file=DERIVED_DIR / f"{output_name}_{today}.csv",
        data_source_type=data_source_type,
    )
